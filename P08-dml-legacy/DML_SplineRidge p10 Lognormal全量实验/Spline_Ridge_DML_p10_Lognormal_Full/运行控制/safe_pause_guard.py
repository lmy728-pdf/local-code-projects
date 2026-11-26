# -*- coding: utf-8 -*-
"""运行中的实验定时安全收尾：不改估计代码，不强杀进程。"""
from __future__ import annotations
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from contextlib import closing

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def task_digest(tasks):
    return hashlib.sha256(json.dumps(sorted(tasks), separators=(",", ":")).encode()).hexdigest()

def row_digest(db):
    h = hashlib.sha256()
    for table in ("manifest", "packets", "attempts"):
        h.update(table.encode())
        ordering = "1" if table == "manifest" else "1,2,3,4"
        for row in db.execute("SELECT * FROM " + table + " ORDER BY " + ordering):
            h.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()

def collect_frame(module, frame=None):
    if frame is None:
        frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is module.collect.__code__ and frame.f_globals is module.__dict__:
            return frame
        frame = frame.f_back
    return None

def verified_backup(module, root, config, loaded, expected, deadline, export_tables=True):
    """仅在当前队列已全部收齐、原 record_batch 已提交之后调用。"""
    root = Path(root)
    stamp = datetime.fromtimestamp(deadline).strftime("%Y%m%d_%H%M%S")
    destination = root / "运行控制" / ("暂停备份_" + stamp)
    destination.mkdir(parents=True, exist_ok=True)
    source = root / "checkpoints" / "replications.sqlite3"
    target = destination / "replications.sqlite3"
    temporary = target.with_suffix(".sqlite3.tmp")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
        assert db.execute("PRAGMA quick_check").fetchone() == ("ok",)
        source_hash = row_digest(db)
        with closing(sqlite3.connect(temporary)) as copied:
            db.backup(copied)
    with temporary.open("r+b") as f:
        os.fsync(f.fileno())
    os.replace(temporary, target)
    with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert source_hash == row_digest(backup), "备份内容与已提交数据库不一致"
        manifest = backup.execute("SELECT config_hash FROM manifest").fetchall()
        assert manifest == [(config["config_hash"],)]
        restored = module.load_packets(backup, config)
        assert set(restored) == set(loaded) == set(expected), "有已派发任务未保存"
        for task in restored:
            assert restored[task][0] == loaded[task][0]
            assert module.clean_json(restored[task][1]) == module.clean_json(loaded[task][1])
        method_n = sum(len(packet["rows"]) for _, packet in restored.values())
        assert method_n == len(restored) * len(module.METHODS)
        if export_tables:
            raw, failures = module.flatten_results(backup, restored, config)
            summary, nuisance = module.summarize_results(raw)
            paired = module.pair_results(raw, summary)
            for name, table in (("replication_results", raw), ("summary", summary),
                                ("failures", failures), ("nuisance_diagnostics", nuisance),
                                ("paired_comparisons", paired)):
                module.atomic_csv(destination / ("paused_" + name + ".csv"), table)
    # 保存源码和配置副本；原实验文件保持不变。
    for name in ("configuration.json", "source_manifest.json", "spline_p10_lognormal_worker.py"):
        file = root / name
        if file.exists():
            shutil.copy2(file, destination / name)
    for file in root.parent.glob("*_代码.ipynb"):
        shutil.copy2(file, destination / file.name)
    receipt = dict(passed=True, state="saved_and_verified", completed=len(restored),
                   method_records=method_n, config_hash=config["config_hash"],
                   task_digest=task_digest(restored), database_content_sha256=source_hash,
                   backup=str(target), saved_at=datetime.now().astimezone().isoformat(),
                   all_dispatched_results_saved=True, export_tables=export_tables)
    atomic_json(destination / "保存核验.json", receipt)
    atomic_json(root / "运行控制" / "pause_receipt.json", receipt)
    return receipt

def install(module, root, deadline, expected_config_hash, require_active=True, export_tables=True):
    root = Path(root).resolve()
    control = root / "运行控制"
    state_path = control / "pause_guard_status.json"
    deadline = float(deadline)
    existing = getattr(module, "_scheduled_pause_guard", None)
    if existing and existing["status"] in ("armed", "draining"):
        assert existing["deadline_epoch"] == deadline, "已有其他暂停时间，需明确更新"
        return existing
    frames = [collect_frame(module, f) for f in sys._current_frames().values()]
    active = next((f for f in frames if f is not None), None)
    if require_active:
        assert active is not None, "没有找到正在执行的 collect，未挂接"
        assert Path(active.f_locals["destination"]).resolve() == root
        assert active.f_locals["config"]["config_hash"] == expected_config_hash
        assert active.f_locals["config"]["mode"] == "full"
    state = dict(status="armed", deadline_epoch=deadline,
                 deadline_local=datetime.fromtimestamp(deadline).astimezone().isoformat(),
                 pid=os.getpid(), config_hash=expected_config_hash, root=str(root),
                 armed_at=datetime.now().astimezone().isoformat())
    original_wait, original_save = module.wait, module.save_progress
    expected = set()
    def publish():
        atomic_json(state_path, state)
    def restore():
        module.wait = original_wait
        module.save_progress = original_save
    def guarded_wait(*args, **kwargs):
        frame = sys._getframe(1)
        answer = original_wait(*args, **kwargs)
        if state["status"] == "armed" and time.time() >= deadline:
            assert collect_frame(module, frame) is frame
            local = frame.f_locals
            assert Path(local["destination"]).resolve() == root
            assert local["config"]["config_hash"] == expected_config_hash
            # 消耗未派发的任务迭代器。已有 futures 由原 collect 完成并保存。
            expected.update(local["loaded"])
            expected.update(local["futures"].values())
            collections.deque(local["source"], maxlen=0)
            state.update(status="draining", draining_at=datetime.now().astimezone().isoformat(),
                         expected_saved_datasets=len(expected),
                         in_flight=len(local["futures"]), task_digest=task_digest(expected))
            publish()
            module.append_log(root, "17点安全暂停：停止派发新任务，等待已派发结果全部写入断点。")
        return answer
    def guarded_save(*args, **kwargs):
        progress = original_save(*args, **kwargs)
        frame = collect_frame(module)
        if state["status"] != "draining" or frame is None or frame.f_locals["futures"]:
            return progress
        local = frame.f_locals
        loaded, config = local["loaded"], local["config"]
        if len(loaded) == progress["total"]:
            state["status"] = "all_computations_finished"
            publish()
            restore()
            return progress
        # save_progress 之前原 record_batch 已经完整提交，batch 已清空。
        assert not local["batch"], "不能在未提交的批次上宣称保存完成"
        try:
            receipt = verified_backup(module, root, config, loaded, expected,
                                      deadline, export_tables=export_tables)
            progress.update(status="paused", eta_seconds=None,
                            pause_reason="scheduled_17_00", backup=receipt["backup"],
                            updated=datetime.now().astimezone().isoformat(timespec="seconds"))
            module.atomic_json(root / "progress.json", progress)
            state.update(status="paused_saved_verified", completed=len(loaded),
                         method_records=receipt["method_records"], backup=receipt["backup"],
                         paused_at=datetime.now().astimezone().isoformat())
            publish()
            module.append_log(root, f'按计划安全暂停：{len(loaded)} 组全部提交，备份完整性及逐条数据核验通过。')
        except Exception:
            state.update(status="save_verification_failed", error=traceback.format_exc())
            publish()
            restore()
            raise
        restore()
        # 使用正常的 Python 中断展开上下文，原 executor 会关闭空闲 workers。
        # 本次队列已经全部完成；没有强杀进程，没有丢弃已计算的结果。
        raise KeyboardInterrupt("按计划安全暂停：全部已派发结果已保存并核验。之后重新运行 Full 即可续跑。")
    module.wait = guarded_wait
    module.save_progress = guarded_save
    module._scheduled_pause_guard = state
    publish()
    return state

def status(root):
    root = Path(root)
    result = {}
    for filename in ("pause_guard_status.json", "pause_receipt.json"):
        path = root / "运行控制" / filename
        if path.exists():
            result[filename] = json.loads(path.read_text(encoding="utf-8"))
    for filename in ("progress.json", "completed.json"):
        path = root / filename
        if path.exists():
            result[filename] = json.loads(path.read_text(encoding="utf-8"))
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status(args.root), ensure_ascii=False, indent=2))


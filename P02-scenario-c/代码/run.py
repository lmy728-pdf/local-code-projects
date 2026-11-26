"""Run one cell, or the full authorized sequence. Checkpoints are per stage and cell."""
from science import SIZES, METHODS, run_replication
import argparse
import hashlib
import io
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import os
import platform
import sqlite3
import sys
import time
import warnings
import zlib

import numpy as np
import scipy
from scipy.optimize import OptimizeWarning
import psutil
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"smoke": 5, "pilot": 100, "main": 2000}
LIMIT_CONTEXT = None

def start_worker():
    global LIMIT_CONTEXT
    LIMIT_CONTEXT = threadpool_limits(1)
    warnings.filterwarnings("ignore", message=r"Unrecognized options detected: .*threads.*parallel.*",
                            category=OptimizeWarning)

def fingerprint():
    checksum = hashlib.sha256()
    for name in ("science.py", "methods.py", "legacy_core.py"):
        checksum.update((ROOT/"代码"/name).read_bytes())
    checksum.update((ROOT/"sources/Rank DML.tex").read_bytes())
    checksum.update(f"{sys.version}|{np.__version__}|{scipy.__version__}".encode())
    return checksum.hexdigest()

def db_path(stage, case, n):
    return ROOT/"checkpoints"/stage/f"{case}_n{n}.sqlite"

def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for retry in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if retry == 9:
                raise
            time.sleep(.05*(retry+1))

def pack(data):
    if data is None:
        return None
    buf = io.BytesIO()
    np.savez(buf, fold=data[0], rare=data[1], errors=data[2])
    return zlib.compress(buf.getbuffer(), level=1)

def persist(db, pending):
    with db:
        db.executemany("INSERT INTO results VALUES(?,?,?)", pending)
    pending.clear()

def get_workers(requested):
    try:
        available = len(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.Error):
        available = os.cpu_count() or 1
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        count, period = quota.read_text().split()
        if count != "max":
            available = min(available, max(1, int(count)//int(period)))
    return min(12, requested, available)

def rss_tree():
    parent = psutil.Process()
    total = 0
    for process in [parent]+parent.children(recursive=True):
        try:
            total += process.memory_info().rss
        except psutil.Error:
            pass
    return total/2**20

def run_cell(stage, case, n, reps, workers, resume):
    path = db_path(stage, case, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    pending = []
    started = time.perf_counter()
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS config (signature TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS results (rep INTEGER PRIMARY KEY, record TEXT NOT NULL, arrays BLOB)")
        row = db.execute("SELECT signature FROM config").fetchone()
        if row is None:
            db.execute("INSERT INTO config VALUES (?)", (fingerprint(),))
            db.commit()
        elif row[0] != fingerprint():
            raise RuntimeError(f"Code or numerical environment changed for {path}")
        done = {row[0] for row in db.execute("SELECT rep FROM results")}
        if done and not resume:
            raise RuntimeError("Existing checkpoint. Use --resume.")
        remaining = [r for r in range(reps) if r not in done]
        if not remaining:
            print(f"SKIP {stage} {case} n={n} complete {reps}", flush=True)
            return
        progress = ROOT/"logs"/f"{stage}_{case}_n{n}.json"
        workers = min(workers, len(remaining))
        committed = reps-len(remaining)
        invalid, peak = 0, rss_tree()
        print(f"START {stage} {case} n={n} R={reps} workers={workers} committed={committed}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=start_worker) as pool:
            todo = iter(remaining)
            active = {}
            def submit():
                for _ in range(workers-len(active)):
                    rep = next(todo, None)
                    if rep is not None:
                        active[pool.submit(run_replication, (stage, case, n, rep))] = rep
            submit()
            last = 0.
            current = committed
            with tqdm(total=reps, initial=committed, desc=f"{stage} {case} {n}", ascii=True,
                      mininterval=2, unit="rep") as bar:
                while active:
                    ready, _ = wait(active, timeout=4, return_when=FIRST_COMPLETED)
                    for job in ready:
                        rep = active.pop(job)
                        record, cache = job.result()
                        assert record["rep"] == rep
                        invalid += sum(not r["valid_interval"] for r in record["methods"])
                        pending.append((rep, json.dumps(record, allow_nan=False), pack(cache)))
                        current += 1
                        bar.update()
                    if len(pending) >= 12:
                        persist(db, pending)
                    elapsed = time.perf_counter()-started
                    if elapsed-last >= .8 or not active:
                        last = elapsed
                        peak = max(peak, rss_tree())
                        try:
                            save_json(progress, dict(stage=stage, case=case, n=n, target=reps,
                                completed=current, previously_done=committed, workers=workers,
                                invalid_this_session=invalid, wall_seconds=elapsed, peak_tree_rss_mib=peak))
                        except OSError as exc:
                            print(f"Progress snapshot unavailable: {exc}", flush=True)
                    submit()
        persist(db, pending)
        elapsed = time.perf_counter()-started
        perf = ROOT/"logs"/"cell_timings.jsonl"
        with perf.open("a", encoding="utf-8") as out:
            out.write(json.dumps(dict(stage=stage, case=case, n=n, target=reps, workers=workers,
                previously_done=committed, new=len(remaining), wall_seconds=elapsed, peak_tree_rss_mib=peak,
                platform=platform.platform()))+"\n")
        print(f"DONE {stage} {case} n={n} R={reps} wall={elapsed:.2f}s", flush=True)
    finally:
        persist(db, pending)
        db.close()

def run(args):
    lock = ROOT/"logs/running.lock"
    try:
        with lock.open("x", encoding="utf-8") as dst:
            dst.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Run lock exists: {lock}. Verify PID is stopped before removing a stale lock.") from exc
    try:
        workers = get_workers(args.workers)
        cases = [args.case] if args.case else ("C1", "C2")
        sizes = [args.n] if args.n else SIZES
        if args.stage == "smoke" and not args.case and not args.n:
            cells = [("C1",500,5), ("C1",8000,2), ("C2",500,5)]
        else:
            cells = [(case,n,args.reps or LIMITS[args.stage]) for case in cases for n in sizes]
        for case, n, reps in cells:
            if psutil.disk_usage(str(ROOT)).free < 2*2**30:
                raise RuntimeError("At least 2 GiB free disk are needed")
            run_cell(args.stage, case, n, reps, workers, args.resume)
    finally:
        lock.unlink(missing_ok=True)

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(LIMITS), required=True)
    parser.add_argument("--case", choices=["C1", "C2"])
    parser.add_argument("--n", type=int, choices=SIZES)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--workers", type=int, default=12, choices=range(1,13))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.reps is not None and (not args.case or not args.n or not 1 <= args.reps <= LIMITS[args.stage]):
        parser.error("--reps requires one explicit cell and must lie within the stage limit")
    run(args)


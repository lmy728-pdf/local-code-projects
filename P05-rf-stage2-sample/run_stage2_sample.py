"""Select saved RF replications per condition and compute only Stage 2."""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import defaultdict
from contextlib import closing
import csv
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "Rank_DML_Scenario_A_Trees"
CACHE = "nuisance_cache/main/rf.sqlite"
DEFAULT_SEED = 20261123
PER_CELL = 200
FILES = ("run.py", "report.py", "experiment_manifest.json", "baseline/main_seed_registry.csv.gz",
         "spline_core/methods.py", "spline_core/legacy_core.py", "spline_core/theory.json", CACHE)


def backup_path(value):
    if value is not None:
        return value.resolve()
    choices = sorted(ROOT.parent.glob(f"{PROJECT}_Backup_*.zip"))
    if len(choices) != 1:
        raise ValueError("Use --backup to select the saved ZIP when it is absent or ambiguous.")
    return choices[0]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_ids(db, runner, seed):
    pool = defaultdict(list)
    for n, law, rep in db.execute("SELECT n,law,rep FROM records ORDER BY n,law,rep"):
        pool[n,law].append(rep)
    all_keys = []
    cells = []
    for n in runner.MANIFEST["sample_sizes"]:
        for law in runner.methods.LAWS:
            available = pool[n,law]
            size = min(PER_CELL, len(available))
            rng = runner.methods.rng("main", n, law, 0, f"backup-rf-stage2-draw/{seed}")
            selected = sorted(int(rep) for rep in rng.choice(available, size=size, replace=False)) if size else []
            all_keys.extend((n, law, rep) for rep in selected)
            cells.append(dict(n=n, law=law, saved=len(available), sampled=size,
                              status="full_sample" if size==PER_CELL else ("under_200" if size else "no_data")))
    return all_keys, cells


def make_checkpoint(out, identity):
    with out:
        out.execute("CREATE TABLE IF NOT EXISTS sample_config(id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT)")
        out.execute("CREATE TABLE IF NOT EXISTS records(n INTEGER,law TEXT,rep INTEGER,"
                    "metadata TEXT,payload BLOB,PRIMARY KEY(n,law,rep))")
        old = out.execute("SELECT identity FROM sample_config WHERE id=1").fetchone()
        if old is None:
            out.execute("INSERT INTO sample_config VALUES(1,?)", (identity,))
        elif old[0] != identity:
            raise ValueError("Existing checkpoint belongs to a different backup, seed or sample. Use --output another folder.")


def jobs(db, keys):
    for key in keys:
        raw, payload = db.execute("SELECT metadata,payload FROM records WHERE n=? AND law=? AND rep=?", key).fetchone()
        yield json.loads(raw), payload


def save_report(root, tables, out, cells, seed, archive, reporter):
    import pandas as pd

    records = []
    for raw, in out.execute("SELECT metadata FROM records ORDER BY n,law,rep"):
        row = json.loads(raw)
        for method in row["rows"]:
            meta = method["metadata"]
            records.append(dict(n=row["n"], law=row["law"], rep=row["rep"], learner="RF",
                                **{k: v for k, v in method.items() if k != "metadata"},
                                selected_L=meta.get("selected_L"), failure=meta.get("failure", "")))
    frame = pd.DataFrame(records)
    assert len(frame) == sum(cell["sampled"] for cell in cells) * len(reporter.methods.METHODS)
    assert not frame.duplicated(["n", "law", "rep", "method"]).any()
    summary = reporter.summarize(frame)
    expected = {(c["n"], c["law"]): c["sampled"] for c in cells if c["sampled"]}
    assert {(row.n, row.law): row.R for row in summary.itertuples()} == expected
    frame.to_csv(tables/"RF_Stage2_逐次结果.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(tables/"RF_Stage2_抽样汇总.csv", index=False, encoding="utf-8-sig")
    select = frame[frame.method=="Rank-Ad"].groupby(
        ["n","law","selected_L"], dropna=False).size().rename("count").reset_index()
    select.to_csv(tables/"RF_Stage2_RankAd选阶.csv", index=False, encoding="utf-8-sig")
    note = root/"结果_MD"
    note.mkdir(exist_ok=True)
    figs = root/"图"
    figs.mkdir(exist_ok=True)
    n = 1000 if 1000 in set(frame.n) else min(frame.n)
    save_coverage_figure(summary, n, figs/"RF_Stage2_抽样覆盖率.pdf", reporter)
    comparison = comparison_table(summary, n, reporter)
    (note/"RF_Stage2_结果.md").write_text(
        f"# Scenario A · RF 第二阶段抽样结果\n\n"
        f"来源为 {archive.name}。独立抽样种子为 {seed}。"
        f"从 {len(expected)} 个有缓存的条件共抽样 {sum(expected.values()):,} 次。"
        "每组在所有已保存 ID 中不放回随机抽样，抽样上限为 200，"
        "本表用于基于现有缓存的抽样比较。\n\n"
        "区间失败计为未覆盖，另列有效区间条件下的覆盖率。"
        "t(1.5) 和 Cauchy 的准确性指标以 MedAE 为主。\n\n"
        f"本次共输出 {len(frame):,} 条方法结果，其中无效区间 {int((~frame.valid_interval).sum()):,} 条。\n\n"
        + comparison + "\n\n"
        + "- [二阶段汇总表](../汇总表_CSV/RF_Stage2_抽样汇总.csv)\n"
        "- [条件抽样计划](../汇总表_CSV/条件抽样计划.csv)\n"
        "- [抽样 ID](../汇总表_CSV/抽样ID.csv)\n"
        f"- [覆盖率 PDF](../图/RF_Stage2_抽样覆盖率.pdf) 展示 n={n:,}，含 95% 参考值和 Monte Carlo 不确定性。\n",
        encoding="utf-8",
    )


def comparison_table(summary, n, reporter):
    view = summary[summary.n == n].set_index(["law", "method"])
    lines = [f"## n = {n:,} 的 OLS 与 Rank-Ad", "",
             "| 分布 | 指标 | OLS 误差 | Rank-Ad 误差 | Rank-Ad Wald | Rank-Ad max-t |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for law in reporter.methods.LAWS:
        if law not in view.index.get_level_values("law"):
            continue
        heavy = law in ("t15", "cauchy")
        metric = "median_absolute_error" if heavy else "rmse"
        label = "MedAE" if heavy else "RMSE"
        ols, ad = (view.loc[(law, name)] for name in ("OLS", "Rank-Ad"))
        lines.append(f"| {reporter.LAWS[law]} | {label} | {ols[metric]:.4f} | "
                     f"{ad[metric]:.4f} | {ad.coverage:.1%} | {ad.max_t_coverage:.1%} |")
    return "\n".join(lines)


def save_coverage_figure(summary, n, dest, reporter):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    view = summary[summary.n == n].set_index(["law", "method"])
    methods = list(reporter.methods.METHODS)
    laws = [law for law in reporter.methods.LAWS if law in view.index.get_level_values("law")]
    data = np.array([[view.loc[(law, method), "coverage"] for method in methods] for law in laws])
    vmin = max(0., min(.8, float(data.min()) - .02))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=.95, vmax=1.)
    fig, ax = plt.subplots(figsize=(10.35, 6.3), layout="constrained")
    image = ax.pcolormesh(data, cmap="RdBu", norm=norm, edgecolors="white", linewidth=.65)
    for i, law in enumerate(laws):
        for j, method in enumerate(methods):
            row = view.loc[(law, method)]
            r, g, b, _ = image.cmap(norm(row.coverage))
            light_background = .2126*r + .7152*g + .0722*b >= .46
            ax.text(j+.5, i+.5, f"{row.coverage:.1%}\n±{100*1.96*row.coverage_mc_se:.1f}",
                    ha="center", va="center", fontsize=9,
                    color="#162438" if light_background else "white")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(methods))+.5, methods, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(laws))+.5, [reporter.LAWS[law] for law in laws])
    ax.tick_params(axis="both", length=0)
    ticks = [0, .25, .5, .75, .85, .90, .95, 1] if vmin < .75 else [.80, .85, .90, .95, 1]
    ticks = [tick for tick in ticks if tick >= vmin]
    cb = fig.colorbar(image, ax=ax, ticks=ticks, fraction=.036, pad=.025, format=PercentFormatter(1))
    cb.solids.set_rasterized(False)
    cb.ax.axhline(.95, color="#162438", lw=1.25)
    ax.set_title("RF Wald coverage  |  Scenario A", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(f"n = {n:,}  ·  95% reference  ·  ±1.96 Monte Carlo SE (percentage points)", labelpad=12)
    fig.savefig(dest)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, help="Server backup ZIP")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, choices=range(1,13), default=12)
    parser.add_argument("--plan-only", action="store_true", help="Verify and select IDs without computing estimates")
    args = parser.parse_args()
    archive = backup_path(args.backup)
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    tables = target/"汇总表_CSV"
    tables.mkdir(exist_ok=True)

    with FileLock(target/"stage2_sample.lock").acquire(timeout=0), TemporaryDirectory(prefix="rf_backup_stage2_") as temp:
        scratch = Path(temp)
        with ZipFile(archive) as z:
            for name in FILES:
                remote = f"{PROJECT}/{name}"
                local = scratch/name
                local.parent.mkdir(exist_ok=True, parents=True)
                with z.open(remote) as source, local.open("wb") as dest:
                    shutil.copyfileobj(source, dest, length=2**20)
        sys.path.insert(0, str(scratch))
        runner = importlib.import_module("run")
        dbpath = scratch/CACHE
        with runner.read_db(dbpath,"main","rf",1) as db:
            keys, cells = sample_ids(db, runner, args.seed)
            if not keys:
                raise ValueError("There are no saved RF replications to sample.")
            with dbpath.open("rb") as f:
                identity = json.dumps(dict(
                    source_sha256=hashlib.file_digest(f, "sha256").hexdigest(),
                    signature=runner.science_signature(), seed=args.seed, keys=keys
                ), separators=(",",":"))
            checkpoint = target/"stage2_sample.sqlite"
            if checkpoint.is_file() and checkpoint.stat().st_size:
                with closing(sqlite3.connect(checkpoint.as_uri()+"?mode=ro", uri=True)) as prior:
                    stored = prior.execute("SELECT identity FROM sample_config WHERE id=1").fetchone()
                    if stored != (identity,):
                        raise ValueError("This output folder already belongs to a different sample.")
            write_csv(tables/"条件抽样计划.csv", cells)
            write_csv(tables/"抽样ID.csv", [dict(n=n, law=law, rep=rep) for n,law,rep in keys])
            print(f"{len(keys):,} rows drawn from {sum(bool(c['sampled']) for c in cells)} conditions.", flush=True)
            if args.plan_only:
                return
            with closing(sqlite3.connect(checkpoint)) as out:
                make_checkpoint(out, identity)
                done = set(out.execute("SELECT n,law,rep FROM records"))
                if not done.issubset(keys):
                    raise ValueError("Unexpected replication ID in the subsample checkpoint.")
                pending = [key for key in keys if key not in done]
                if pending:
                    workers = runner.effective_workers(args.workers)
                    runner.dispatch(out, runner.stage2, jobs(db,pending), keys, len(done), workers, "RF Stage 2")
                else:
                    print("All selected replications are saved. Regenerating reports.", flush=True)
                reporter = importlib.import_module("report")
                save_report(target, tables, out, cells, args.seed, archive, reporter)
    print(f"Saved results to {target}")


if __name__ == "__main__":
    main()

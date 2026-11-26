"""Export the formal Scenario A RF report from a complete Stage 2 checkpoint."""

import argparse
from contextlib import closing
from hashlib import file_digest
import json
from pathlib import Path
from shutil import copyfileobj
import sqlite3
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import report
import run
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def load_results(path, expected_keys):
    rows = []
    registry = run.seed_registry()
    expected_methods = set(run.methods.METHODS)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("Stage 2 SQLite integrity check failed")
        config = db.execute("SELECT signature, mode, learner, stage FROM config").fetchall()
        if len(config) != 1 or config[0][1:] != ("main", "rf", 2):
            raise ValueError("This is not a formal RF Stage 2 checkpoint")
        keys = set(db.execute("SELECT n, law, rep FROM records"))
        if keys != expected_keys:
            raise ValueError(f"Expected {len(expected_keys):,} formal IDs, found {len(keys):,}; "
                             f"missing {len(expected_keys - keys):,}, unexpected {len(keys - expected_keys):,}")
        for n, law, rep, raw in db.execute("SELECT n, law, rep, metadata FROM records"):
            record = json.loads(raw)
            key = (n, law, rep)
            if (record["n"], record["law"], record["rep"]) != key or record["seed_key"] != registry[key]:
                raise ValueError(f"Mismatched replication ID or seed: {key}")
            if record["learner"] != "rf" or record["mode"] != "main":
                raise ValueError(f"Mismatched learner or mode: {key}")
            estimates = record["rows"]
            if len(estimates) != len(expected_methods) or {m["method"] for m in estimates} != expected_methods:
                raise ValueError(f"Missing or duplicate estimator: {key}")
            for estimate in estimates:
                metadata = estimate["metadata"]
                rows.append(dict(learner="RF", n=n, law=law, rep=rep,
                                 **{k: v for k, v in estimate.items() if k != "metadata"},
                                 selected_L=metadata.get("selected_L"), failure=metadata.get("failure", "")))
    return pd.DataFrame(rows), config[0][0]


def save_figure(primary, dest, n, coverage, scale_limit):
    laws = list(run.methods.LAWS)
    methods = list(run.methods.METHODS)
    ordered = primary.loc[pd.MultiIndex.from_product([laws, methods])]
    if coverage:
        values = ordered.coverage.to_numpy().reshape(len(laws), len(methods))
        margin = np.repeat("", values.size).astype(object)
        for i, (value, se) in enumerate(zip(values.flat, ordered.coverage_mc_se)):
            margin[i] = f"{value:.1%}\n±{100 * 1.96 * se:.1f}"
        labels = margin.reshape(values.shape)
        low = scale_limit
        ticks = [low, (low + .95) / 2, .95, 1.0]
        norm = TwoSlopeNorm(vmin=low, vcenter=.95, vmax=1.0)
        cmap, title = "RdBu", "RF Wald coverage  |  Scenario A"
        note = f"n = {n:,}  ·  95% reference  ·  ±1.96 Monte Carlo SE (percentage points)"
    else:
        heavy = ordered.index.get_level_values(0).isin(report.HEAVY)
        values = np.where(heavy, ordered.relative_medae_ols, ordered.relative_rmse_ols)
        values = values.reshape(len(laws), len(methods))
        errors = np.where(heavy, ordered.median_absolute_error, ordered.rmse).reshape(values.shape)
        labels = np.array([[f"{error:.4f}" for error in row] for row in errors])
        high = scale_limit
        norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=high)
        ticks = sorted({0, .5, 1.0, high})
        cmap, title = "RdBu_r", "RF estimation error  |  Scenario A"
        note = f"n = {n:,}  ·  Cell values show RMSE, or MedAE for t(1.5) and Cauchy"

    fig, ax = plt.subplots(figsize=(9.1, 6.4), layout="constrained")
    plot = ax.pcolormesh(values, cmap=cmap, norm=norm, edgecolors="white", linewidth=.6)
    for i, j in np.ndindex(values.shape):
        r, g, b, _ = plot.cmap(plot.norm(values[i, j]))
        color = "white" if .2126*r + .7152*g + .0722*b < .46 else "#17202a"
        ax.text(j + .5, i + .5, labels[i, j], ha="center", va="center", fontsize=9, color=color)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(methods)) + .5, methods, rotation=26, ha="right")
    ax.set_yticks(np.arange(len(laws)) + .5, [report.LAWS[law] for law in laws])
    ax.tick_params(axis="both", length=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_xlabel(note, labelpad=10)
    colorbar = fig.colorbar(plot, ax=ax, ticks=ticks, fraction=.047, pad=.032,
                            format=PercentFormatter(1, decimals=0) if coverage else None)
    colorbar.solids.set_rasterized(False)
    colorbar.ax.axhline(.95 if coverage else 1.0, color="#17202a", lw=1.1)
    if not coverage:
        colorbar.set_label("Error / OLS")
    fig.savefig(dest)
    plt.close(fig)


def comparison_markdown(primary, n):
    lines = [
        f"## n = {n:,} 时 OLS 与 Rank-Ad", "",
        "| 分布 | 指标 | OLS 绝对误差 | Rank-Ad 相对误差 | OLS Wald | Rank-Ad Wald | Rank-Ad max-t |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    better = 0
    for law in run.methods.LAWS:
        ols = primary.loc[(law, "OLS")]
        rank = primary.loc[(law, "Rank-Ad")]
        if law in report.HEAVY:
            metric, raw, ratio = "MedAE", ols.median_absolute_error, rank.relative_medae_ols
        else:
            metric, raw, ratio = "RMSE", ols.rmse, rank.relative_rmse_ols
        better += ratio < 1
        lines.append(f"| {report.LAWS[law]} | {metric} | {raw:.4f} | {ratio:.3f} | "
                     f"{ols.coverage:.2%} | {rank.coverage:.2%} | {rank.max_t_coverage:.2%} |")
    rank_ad = primary.xs("Rank-Ad", level="method")
    lines.extend([
        "", f"Rank-Ad 的主要点估计误差有 {better} 种低于同分布 OLS。"
        f"普通 Wald 的经验覆盖率为 {rank_ad.coverage.min():.1%}–{rank_ad.coverage.max():.1%}，"
        f"max-t 为 {rank_ad.max_t_coverage.min():.1%}–{rank_ad.max_t_coverage.max():.1%}。", "",
    ])
    return lines


def export_results(results, source, signature, dest):
    tables, figures, notes = (dest / name for name in ("汇总表_CSV", "图", "结果_MD"))
    for path in (tables, figures, notes):
        path.mkdir(parents=True, exist_ok=True)

    summary = report.summarize(results)
    summary.to_csv(tables / "RF_正式汇总.csv", index=False, encoding="utf-8-sig")
    selected = results[results.method == "Rank-Ad"].groupby(
        ["n", "law", "selected_L"], dropna=False).size().rename("count").reset_index()
    selected.to_csv(tables / "RF_RankAd选阶.csv", index=False, encoding="utf-8-sig")

    plotted_sizes = (1000, 2000)
    plotted = summary[summary.n.isin(plotted_sizes)]
    common_low = min(.85, np.floor(plotted.coverage.min() * 20) / 20)
    relative_error = np.where(plotted.law.isin(report.HEAVY),
                              plotted.relative_medae_ols, plotted.relative_rmse_ols)
    common_high = max(1.5, np.ceil(np.nanmax(relative_error) * 2) / 2)
    invalid = int((~results.valid_interval).sum())
    R = int(summary.R.iloc[0])
    with source.open("rb") as stream:
        checksum = file_digest(stream, "sha256").hexdigest()
    lines = [
        "# Scenario A · RF 正式结果", "",
        f"每个条件 {R:,} 次重复，覆盖三个样本量和九种误差分布。"
        f"共核验 {len(results) // len(run.methods.METHODS):,} 个正式重复 ID 和随机种子，"
        f"得到 {len(results):,} 条方法结果，其中 {invalid:,} 条区间无效。", "",
        "覆盖率以全部重复为分母，无效区间计为未覆盖。CSV 也提供仅按有效区间计算的覆盖率。"
        "覆盖率图用 95% 作参考，标注的误差范围为 ±1.96 倍 Monte Carlo 标准误。"
        "前七种分布报告 RMSE，t(1.5) 和 Cauchy 报告 MedAE。", "",
    ]
    for n in plotted_sizes:
        primary = plotted[plotted.n == n].set_index(["law", "method"])
        save_figure(primary, figures / f"RF_Wald覆盖率_n{n}.pdf", n, coverage=True, scale_limit=common_low)
        save_figure(primary, figures / f"RF_估计误差_n{n}.pdf", n, coverage=False, scale_limit=common_high)
        lines.extend(comparison_markdown(primary, n))

    lines.extend([
        "## 图表索引", "",
        "- [正式汇总 CSV](../汇总表_CSV/RF_正式汇总.csv) 包含全部三个样本量、九种分布、九种方法。",
        "- [Rank-Ad 选阶 CSV](../汇总表_CSV/RF_RankAd选阶.csv) 包含每个条件的选阶频数。",
        "- n = 1,000 · [Wald 覆盖率](../图/RF_Wald覆盖率_n1000.pdf) · [估计误差](../图/RF_估计误差_n1000.pdf)",
        "- n = 2,000 · [Wald 覆盖率](../图/RF_Wald覆盖率_n2000.pdf) · [估计误差](../图/RF_估计误差_n2000.pdf)", "",
        "同一指标在两个样本量下共用色标。估计误差图的数字是绝对 RMSE 或 MedAE，颜色表示相对于同分布 OLS 的误差。", "",
        "## 数据来源", "",
        f"正式 RF 第二阶段检查点 `{source.name}`。备份文件 SHA-256 为 `{checksum}`。"
        f"SQLite 中登记的实验签名为 `{signature}`。", "",
    ])
    (notes / "RF_正式结果.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Stage 2 rf.sqlite or a ZIP with rf.sqlite")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    keys = set(run.keys_for("main"))
    if args.checkpoint.suffix.lower() == ".zip":
        with TemporaryDirectory(prefix="rf-report-") as folder, ZipFile(args.checkpoint) as archive:
            path = Path(folder) / "rf.sqlite"
            source, = [name for name in archive.namelist() if not name.endswith("/") and
                       Path(name).name == "rf.sqlite"]
            with archive.open(source) as compressed, path.open("xb") as extracted:
                copyfileobj(compressed, extracted, length=2**20)
            results, signature = load_results(path, keys)
    else:
        results, signature = load_results(args.checkpoint, keys)
    export_results(results, args.checkpoint, signature, args.output)
    print(f"Verified {len(keys):,} replications. Report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()

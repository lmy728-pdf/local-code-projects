"""Build the three Scenario A learner figures and the comparison tables."""
import argparse
import json

from run import ROOT, LEARNERS, db_path, keys_for, read_db, methods
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ORDER = ("Spline Ridge", "RF", "GBDT")
NAMES = {"rf": "RF", "gbdt": "GBDT"}
HEAVY = ("t15", "cauchy")
LAWS = dict(zip(methods.LAWS, ("Gaussian", "Logistic", "t(3)", "Contam. normal",
                              "Bimodal", "Lognormal (.75,.2)", "Lognormal (1,.5)", "t(1.5)", "Cauchy")))
DIAGNOSTIC_FIELDS = ("fold", "status", "n_train", "n_valid", "outer_seeds", "outer_iterations",
                   "training_observed_rmse", "validation_observed_rmse", "validation_truth_rmse",
                   "huber_scale", "huber_inner", "component", "failure", "reference_rmse_m",
                   "reference_rmse_ell", "alignment_rmse", "alignment_location", "alignment_product",
                   "root_n_alignment_product", "cov_b_a")


def load_records(mode, learner, phase):
    path = db_path(mode, learner, phase)
    if not path.is_file():
        return
    with read_db(path, mode, learner, phase) as db:
        for text, in db.execute("SELECT metadata FROM records ORDER BY n,law,rep"):
            yield json.loads(text)


def flatten(mode):
    results, diagnostics, stage1 = [], [], []
    for learner in LEARNERS:
        name = NAMES[learner]
        for record in load_records(mode, learner, 2):
            base = {k: record.get(k) for k in ("n", "law", "rep", "sample_sha256", "fold_sha256")}
            for row in record["rows"]:
                meta = row["metadata"]
                results.append(dict(learner=name, **base, **{k: v for k, v in row.items() if k != "metadata"},
                                    selected_L=meta.get("selected_L"), failure=meta.get("failure", "")))
        for record in load_records(mode, learner, 1):
            base = {k: record.get(k) for k in ("n", "law", "rep", "sample_sha256", "fold_sha256")}
            stage1.append(dict(learner=name, **base, status=record["status"],
                               failure=record.get("failure", ""), seconds=record["nuisance_seconds"],
                               seed_key=json.dumps(record.get("seed_key"))))
            for detail in record["diagnostics"]:
                row = {}
                for key in DIAGNOSTIC_FIELDS:
                    value = detail.get(key)
                    row[key] = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                diagnostics.append(dict(learner=name, **base, **row))
    return pd.DataFrame(results), pd.DataFrame(diagnostics), pd.DataFrame(stage1)


def median(series):
    valid = series.dropna()
    return valid.median() if len(valid) else np.nan


def summarize(results):
    summary = []
    for (learner, n, law, method), g in results.groupby(["learner", "n", "law", "method"]):
        R = len(g)
        point = g.loc[g.valid_estimator, "estimate"].to_numpy(float)
        intervals = g[g.valid_interval]
        cov = float(g.covered.mean())
        if method == "Rank-Ad":
            max_n = int(g.max_t_covered.notna().sum())
            max_cov = g.max_t_covered.eq(True).sum() / R
            max_valid_cov = g.max_t_covered.eq(True).sum() / max_n if max_n else np.nan
        else:
            max_cov, max_valid_cov = np.nan, np.nan
        sd = np.diff(np.percentile(point, [25, 75]))[0] / 1.348979500392 if len(point) > 1 else np.nan
        median_se = median(intervals.se)
        summary.append(dict(learner=learner, n=n, law=law, method=method, R=R,
                            bias=float(np.mean(point-1)) if len(point) else np.nan,
                            rmse=float(np.sqrt(np.mean((point-1)**2))) if len(point) else np.nan,
                            median_absolute_error=float(np.median(abs(point-1))) if len(point) else np.nan,
                            coverage=cov, coverage_mc_se=np.sqrt(cov*(1-cov)/R),
                            coverage_valid_only=intervals.covered.mean(),
                            median_interval_length=median(intervals.length), median_se=median_se,
                            iqr_empirical_sd=sd, calibration_ratio=median_se/sd if sd > 0 else np.nan,
                            valid_estimators=len(point), valid_intervals=len(intervals),
                            failure_rate=1-len(intervals)/R, mean_seconds=g.seconds.mean(),
                            median_seconds=g.seconds.median(), max_t_coverage=max_cov,
                            max_t_coverage_valid_only=max_valid_cov,
                            max_t_coverage_mc_se=np.sqrt(max_cov*(1-max_cov)/R),
                            max_t_median_length=median(g.max_t_length),
                            infinite_variance_stress_test=law in HEAVY))
    s = pd.DataFrame(summary)
    base = s[s.method == "OLS"].set_index(["learner", "n", "law"])
    for metric, relative in (("rmse", "relative_rmse_ols"),
                             ("median_absolute_error", "relative_medae_ols")):
        baseline = pd.Series([base[metric].get((r.learner, r.n, r.law), np.nan)
                              for r in s.itertuples()], index=s.index)
        s[relative] = s[metric] / baseline.where(baseline > 0)
    return s


def pairing(stage1):
    if stage1.empty:
        return pd.DataFrame(columns=["n", "law", "shared_ids", "matched_hashes", "hash_conflicts", "unverifiable"])
    keys = ["n", "law", "rep"]
    rf = stage1[stage1.learner == "RF"]
    gb = stage1[stage1.learner == "GBDT"]
    both = rf.merge(gb, on=keys, suffixes=("_rf", "_gb"), validate="one_to_one")
    rows = []
    for (n, law), g in both.groupby(["n", "law"]):
        good = g[["sample_sha256_rf", "sample_sha256_gb", "fold_sha256_rf", "fold_sha256_gb"]].notna().all(axis=1)
        matches = good & g.sample_sha256_rf.eq(g.sample_sha256_gb) & g.fold_sha256_rf.eq(g.fold_sha256_gb)
        rows.append(dict(n=n, law=law, shared_ids=len(g), matched_hashes=int(matches.sum()),
                         hash_conflicts=int((good & ~matches).sum()), unverifiable=int((~good).sum())))
    return pd.DataFrame(rows, columns=["n", "law", "shared_ids", "matched_hashes", "hash_conflicts", "unverifiable"])


def save_matrix(summary, n, figure_dir, is_coverage):
    sample = summary[summary.n == n]
    learners = [name for name in ORDER if name in set(sample.learner)]
    laws = [law for law in methods.LAWS if law in set(sample.law)]
    fig, axes = plt.subplots(1, len(learners), squeeze=False,
                             figsize=(6*len(learners)+1, max(4, .65*len(laws)+2)), layout="constrained")
    if is_coverage:
        norm = TwoSlopeNorm(vmin=0, vcenter=.95, vmax=1)
        cmap = "RdBu"
    else:
        norm, cmap = None, "viridis_r"
    for ax, learner in zip(axes[0], learners):
        subset = sample[sample.learner == learner].set_index(["law", "method"])
        vals = np.full((len(laws), len(methods.METHODS)), np.nan)
        labels = {}
        for i, law in enumerate(laws):
            for j, method in enumerate(methods.METHODS):
                if (law, method) not in subset.index:
                    continue
                entry = subset.loc[(law, method)]
                metric = "median_absolute_error" if law in HEAVY else "rmse"
                if is_coverage:
                    vals[i, j] = entry.coverage
                    labels[i, j] = f"{entry.coverage:.1%}\n±{100*1.96*entry.coverage_mc_se:.1f}"
                elif np.isfinite(entry[metric]):
                    vals[i, j] = entry[metric]
                    labels[i, j] = f"{entry[metric]:.3f}"
        if not is_coverage:
            # Use one scale per law shared by all learner panels. Text shows absolute error.
            for i, law in enumerate(laws):
                metric = "median_absolute_error" if law in HEAVY else "rmse"
                maximum = sample.loc[sample.law == law, metric].max()
                if maximum > 0:
                    vals[i] /= maximum
        image = ax.pcolormesh(vals, cmap=cmap, norm=norm, vmin=None if norm else 0,
                              vmax=None if norm else 1)
        ax.invert_yaxis()
        for (i, j), label in labels.items():
            r, g, b, _ = image.cmap(image.norm(vals[i, j]))
            text_color = "white" if .2126*r+.7152*g+.0722*b < .46 else "#17202a"
            ax.text(j + .5, i + .5, label, ha="center", va="center", fontsize=8, color=text_color)
        ax.set_xticks(np.arange(len(methods.METHODS)) + .5, methods.METHODS, rotation=38, ha="right")
        ax.set_yticks(np.arange(len(laws)) + .5, [LAWS[law] for law in laws])
        ax.set_title(learner, loc="left", fontweight="bold")
    if is_coverage:
        colorbar = fig.colorbar(image, ax=list(axes[0]), ticks=[0, .25, .50, .75, .90, .95, 1.0],
                                fraction=.025, format=PercentFormatter(1))
        colorbar.solids.set_rasterized(False)
        colorbar.ax.axhline(.95, color="#152935", lw=1.1)
        title = f"Wald coverage at n={n:,}  |  95% reference; ±1.96 Monte Carlo SE (pp)"
        name = "A_Learners_Coverage.pdf"
    else:
        title = f"Absolute error at n={n:,}  |  RMSE; MedAE for t(1.5) and Cauchy"
        name = "A_Learners_Accuracy.pdf"
    fig.suptitle(title, fontweight="bold")
    fig.savefig(figure_dir/name)
    plt.close(fig)


def save_nuisance_figure(nuisance, n, figure_dir):
    sample = nuisance[nuisance.n == n]
    if sample.empty:
        return
    metrics = ("reference_rmse_m", "reference_rmse_ell", "root_n_alignment_product")
    labels = ("Treatment RMSE", "Outcome RMSE", "Root-n alignment product")
    laws = [k for k in methods.LAWS if k in set(sample.law)]
    fig, axes = plt.subplots(1, 3, figsize=(14, max(3.6, .37*len(laws)+1.8)), sharey=True, layout="constrained")
    for ax, metric, label in zip(axes, metrics, labels):
        for learner, c, marker in zip(ORDER, ("#426c9c", "#bc5c32", "#088275"), ("o", "s", "^")):
            subset = sample[sample.learner == learner].set_index("law")
            if subset.empty:
                continue
            ax.scatter(subset[metric].reindex(laws), np.arange(len(laws)), label=learner, color=c, marker=marker)
        ax.set_xscale("symlog", linthresh=.1)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=.25)
    axes[0].set_yticks(range(len(laws)), [LAWS[k] for k in laws])
    axes[0].invert_yaxis()
    axes[2].legend(frameon=False)
    fig.suptitle(f"Reference-grid nuisance diagnostics at n={n:,}  |  fold medians", fontweight="bold")
    fig.savefig(figure_dir/"A_Learners_Nuisance.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke","pilot","main"))
    mode = parser.parse_args().mode
    results, diagnostics, stage1 = flatten(mode)
    if results.empty:
        parser.error(f"No {mode} Stage 2 results. Run the chosen learner and stage first.")
    out, figs = ROOT/"results"/mode, ROOT/"figures"/mode
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    s = summarize(results)
    select = results[results.method == "Rank-Ad"].groupby(
        ["learner", "n", "law", "selected_L"], dropna=False).size().rename("count").reset_index()
    metrics = ("reference_rmse_m", "reference_rmse_ell", "alignment_rmse",
               "root_n_alignment_product", "cov_b_a")
    if not diagnostics.empty:
        complete = diagnostics.dropna(subset=["reference_rmse_m"])
        nuisance = complete.groupby(["learner", "n", "law"]).agg(
            **{k: (k, "median") for k in metrics}, reference_folds=("fold", "count")).reset_index()
    else:
        diagnostics = pd.DataFrame(columns=["learner", "n", "law", "rep", *DIAGNOSTIC_FIELDS])
        nuisance = pd.DataFrame(columns=["learner", "n", "law", *metrics, "reference_folds"])
    if mode == "main":
        base = pd.read_csv(ROOT/"baseline"/"main_results.csv").assign(learner="Spline Ridge")
        ols = base[base.method == "OLS"].set_index(["n","law"]).median_absolute_error
        base["relative_medae_ols"] = [r.median_absolute_error / ols.loc[(r.n,r.law)] for r in base.itertuples()]
        base["max_t_coverage_mc_se"] = np.sqrt(base.max_t_coverage*(1-base.max_t_coverage)/base.R)
        s = pd.concat([s, base], ignore_index=True)
        base_sel = pd.read_csv(ROOT/"baseline"/"rank_ad_selection.csv").assign(learner="Spline Ridge")
        select = pd.concat([select, base_sel], ignore_index=True)
        base_n = pd.read_csv(ROOT/"baseline"/"nuisance_summary.csv").assign(learner="Spline Ridge")
        nuisance = pd.concat([nuisance, base_n], ignore_index=True)
    paired = pairing(stage1)
    outputs = dict(main_results=s, replications=results, rank_ad_selection=select,
                   nuisance_diagnostics=diagnostics, nuisance_summary=nuisance,
                   stage1_status=stage1, sample_pairing=paired)
    for name, frame in outputs.items():
        frame.to_csv(out/f"{name}.csv", index=False, encoding="utf-8-sig")
    available_n = set(results.n)
    n = 1000 if 1000 in available_n else min(available_n)
    save_matrix(s, n, figs, True)
    save_matrix(s, n, figs, False)
    save_nuisance_figure(nuisance, n, figs)
    sizes = results.groupby("learner").size().div(len(methods.METHODS)).astype(int)
    all_main = mode == "main" and all(sizes.get(name,0) == len(keys_for(mode)) for name in NAMES.values())
    both_matches = (int(paired.matched_hashes.sum()), int(paired.hash_conflicts.sum()), int(paired.unverifiable.sum()))
    failures = int((~results.valid_interval).sum())
    diagnostic_misses = int((stage1.status == "failed").sum())
    lines = [f"# Scenario A 三学习器比较（{mode}）", "",
             f"正式进度为{'完整' if all_main else '当前已提交范围'}。"]
    for name in NAMES.values():
        lines.append(f"- {name} 完成 {sizes.get(name, 0):,} 次 Stage 2。")
    lines += [f"- RF 与 GBDT 的共同 ID 中，样本及折哈希匹配 {both_matches[0]:,} 次，"
              f"冲突 {both_matches[1]:,} 次，无法核验 {both_matches[2]:,} 次。",
              f"- 共有 {failures:,} 条无效区间和 {diagnostic_misses:,} 次带 Stage 1 异常的重复。详情见 CSV。",
              "", "Spline Ridge 主表沿用历史正式聚合结果。树学习器沿用已记录的 DGP 和分折种子；"
              "历史实验未存完整 X、D、Y 与折的联合哈希，三学习器表按聚合结果比较。",
              "", "有限方差分布以 RMSE 为主要准确性指标，t(1.5) 和 Cauchy 以 MedAE 和 IQR 稳健尺度为主。"
              "覆盖率分母含无效区间，CSV 另有有效区间覆盖率。参考网格误差采用已知 nuisance 真值；"
              "训练与外层验证的观测误差另存，极端分布的风险仅作压力测试诊断。",
              "", f"三张 PDF 展示 n={n:,}。CSV 包含所有已完成样本量及分布。", "",
              "- [主表](main_results.csv) · [选阶频数](rank_ad_selection.csv) · [样本配对核验](sample_pairing.csv)",
              "- [覆盖率](../../figures/{0}/A_Learners_Coverage.pdf) · [绝对误差](../../figures/{0}/A_Learners_Accuracy.pdf)".format(mode)]
    if (figs/"A_Learners_Nuisance.pdf").is_file():
        lines.append(f"- [第一阶段诊断](../../figures/{mode}/A_Learners_Nuisance.pdf)")
    (out/"A_Learners_结果.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Tables: {out}\nFigures: {figs}")


if __name__ == "__main__":
    main()

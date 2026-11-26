"""Summarize immutable pilot task records without confusing failures with coverage.

Usage: python summarize_pilot.py results/RUN_ID
Only derived reports are replaced. No simulation or model training occurs here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev

METHODS = ("original_mean_dml", "standard_score", "dukes_dr_gcm")
METHOD_LABELS = dict(zip(METHODS, ("A: original Mean-DML", "B: standard score", "C: Dukes DR-GCM (Python)")))
SCENARIO_LABELS = {"gaussian": "Gaussian", "t2.1": "Student-t(2.1)", "lognormal2": "Centered lognormal(2)"}
ESTIMATE_FIELDS = ("theta_hat", "theta_se", "ci_low", "ci_high", "ci_length")
SUMMARY_FIELDS = ["scenario", "method", "coverage_type", "planned", "attempted", "pending", "valid", "failed", "covered_count", "coverage_valid", "failure_rate", "failure_rate_denominator", "failure_fraction_planned", "planned_coverage_lower", "planned_coverage_upper", "mcse", "wilson_low", "wilson_high", "coverage_minus_gaussian", "bias", "estimate_sd", "mean_se", "rmse", "mean_ci_length"]
PAIR_FIELDS = ["scenario", "comparison", "method_left", "method_right", "planned", "common_attempted", "valid_pairs", "left_failed", "right_failed", "nonpaired_attempted", "missing_pairs_planned", "mean_difference", "mcse", "count_minus1", "count_zero", "count_plus1", "planned_difference_lower", "planned_difference_upper"]


def wilson(successes, count, level=0.95):
    """Conditional binomial Wilson interval; count is the valid-run denominator."""
    if count == 0:
        return None, None
    z = NormalDist().inv_cdf((1.0 + level) / 2.0)
    p = successes / count
    den = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / den
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return max(0.0, center - half), min(1.0, center + half)


def scenario_names(config, rows):
    configured = config.get("scenarios", [])
    if isinstance(configured, dict):
        names = list(configured)
    else:
        names = [s if isinstance(s, str) else s.get("name", s.get("id", s.get("scenario"))) for s in configured]
    if any(s is None for s in names):
        raise ValueError("Every configured scenario needs name, id, or scenario")
    return names or list(dict.fromkeys(r["scenario"] for r in rows))


def validate_records(rows, config):
    """Refuse duplicated keys, mixed configurations, and fabricated B/C estimates."""
    planned = int(config["replications"])
    if planned < 1:
        raise ValueError("replications must be positive")
    allowed_scenarios = set(scenario_names(config, rows))
    keys, hashes = set(), set()
    for row in rows:
        key = (row["scenario"], int(row["replication"]), row["method"])
        if key in keys:
            raise ValueError(f"Duplicate task/method key: {key}")
        keys.add(key)
        if key[0] not in allowed_scenarios or key[2] not in METHODS or not 0 <= key[1] < planned:
            raise ValueError(f"Unexpected scenario, method, or replication: {key}")
        if row.get("status") not in ("ok", "failed"):
            raise ValueError(f"Unknown status: {key}")
        if row["status"] == "ok" and not isinstance(row.get("covered"), bool):
            raise ValueError(f"Successful result must have a Boolean coverage indicator: {key}")
        if row["status"] == "failed" and row.get("covered") is not None:
            raise ValueError(f"A failed method must have covered=null: {key}")
        if row["method"] != METHODS[0] and any(row.get(field) is not None for field in ESTIMATE_FIELDS):
            raise ValueError(f"B/C point estimates and interval endpoints were not computed: {key}")
        if row.get("config_hash"):
            hashes.add(row["config_hash"])
    if len(hashes) > 1 or (hashes and config.get("config_hash") and hashes != {config["config_hash"]}):
        raise ValueError("Cannot merge results from different configurations")


def summarize_records(rows, config):
    validate_records(rows, config)
    planned = int(config["replications"])
    truth = float(config.get("theta0", 1.0))
    summary = []
    for scenario in scenario_names(config, rows):
        for method in METHODS:
            subset = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            valid = [r for r in subset if r["status"] == "ok"]
            attempted, nv = len(subset), len(valid)
            failed = attempted - nv
            successes = sum(r["covered"] for r in valid)
            coverage = successes / nv if nv else None
            lo, hi = wilson(successes, nv)
            item = dict.fromkeys(SUMMARY_FIELDS)
            item.update(scenario=scenario, method=method,
                        coverage_type="wald_coverage" if method == METHODS[0] else "inverted_test_coverage_at_truth",
                        planned=planned, attempted=attempted, pending=planned-attempted,
                        valid=nv, failed=failed, covered_count=successes, coverage_valid=coverage,
                        failure_rate=failed/attempted if attempted else None,
                        failure_rate_denominator="attempted", failure_fraction_planned=failed/planned,
                        planned_coverage_lower=successes/planned,
                        planned_coverage_upper=(successes+planned-nv)/planned,
                        mcse=math.sqrt(coverage*(1-coverage)/nv) if nv else None,
                        wilson_low=lo, wilson_high=hi)
            if method == METHODS[0] and valid:
                estimates = [float(r["theta_hat"]) for r in valid]
                if not all(math.isfinite(x) for x in estimates):
                    raise ValueError("Successful A point estimates must be finite")
                item.update(bias=mean(estimates)-truth,
                            estimate_sd=stdev(estimates) if nv > 1 else None,
                            mean_se=mean(float(r["theta_se"]) for r in valid),
                            rmse=math.sqrt(mean((x-truth)**2 for x in estimates)),
                            mean_ci_length=mean(float(r["ci_length"]) for r in valid))
            summary.append(item)
    gaussian = {r["method"]: r["coverage_valid"] for r in summary if r["scenario"] == "gaussian"}
    for row in summary:
        base = gaussian.get(row["method"])
        if row["coverage_valid"] is not None and base is not None:
            row["coverage_minus_gaussian"] = row["coverage_valid"] - base
    return summary


def paired_records(rows, config):
    validate_records(rows, config)
    planned = int(config["replications"])
    pairs = []
    for scenario in scenario_names(config, rows):
        lookup = {m: {int(r["replication"]): r for r in rows if r["scenario"] == scenario and r["method"] == m} for m in METHODS}
        for left, right, label in ((METHODS[2], METHODS[1], "C-B"), (METHODS[2], METHODS[0], "C-A"), (METHODS[1], METHODS[0], "B-A")):
            ls, rs = lookup[left], lookup[right]
            common = set(ls) & set(rs)
            good = [k for k in sorted(common) if ls[k]["status"] == rs[k]["status"] == "ok"]
            delta = [int(ls[k]["covered"])-int(rs[k]["covered"]) for k in good]
            n = len(delta)
            missing = planned-n
            pairs.append(dict(scenario=scenario, comparison=label, method_left=left, method_right=right,
                              planned=planned, common_attempted=len(common), valid_pairs=n,
                              left_failed=sum(r["status"] == "failed" for r in ls.values()),
                              right_failed=sum(r["status"] == "failed" for r in rs.values()),
                              nonpaired_attempted=len(set(ls)|set(rs))-n, missing_pairs_planned=missing,
                              mean_difference=mean(delta) if n else None,
                              mcse=stdev(delta)/math.sqrt(n) if n > 1 else None,
                              count_minus1=delta.count(-1), count_zero=delta.count(0), count_plus1=delta.count(1),
                              planned_difference_lower=(sum(delta)-missing)/planned,
                              planned_difference_upper=(sum(delta)+missing)/planned))
    return pairs


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, preferred=()):
    fields = list(preferred)
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    if not fields:
        fields = ["scenario", "replication", "method", "status", "failure_type", "failure_fold"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    temporary.replace(path)


def plot_comparison(path, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenarios = list(dict.fromkeys(r["scenario"] for r in summary))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.5*len(scenarios), 6), squeeze=False, sharey=True)
    colors = ("#486581", "#D99724", "#247C72")
    for ax, scenario in zip(axes[0], scenarios):
        subset = [r for r in summary if r["scenario"] == scenario]
        for j, row in enumerate(subset):
            c = row["coverage_valid"]
            if c is not None:
                ax.errorbar(j, c, yerr=[[max(0, c-row["wilson_low"])], [max(0, row["wilson_high"]-c)]], fmt="o", color=colors[j], markersize=9, capsize=5)
                ax.annotate(f"{c:.3f}", (j, row["wilson_high"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10)
            ax.text(j, -.18, f"valid {row['valid']}/{row['planned']}\nfailed {row['failed']}; pending {row['pending']}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, clip_on=False)
        ax.axhline(.95, linestyle="--", linewidth=1.1, color="#9B3547")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=12)
        ax.set_xticks(range(3), ["A\nWald", "B\nScore inversion", "C\nDR score inversion"])
        ax.set_ylim(0, 1.08)
        ax.set_xlim(-.5, 2.5)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Coverage among valid runs (Wilson 95% interval)")
    fig.suptitle("DGP2 | n = 1000, p = 5 | truth-only coverage screening", fontsize=14)
    fig.text(.5, .015, "Dashed line: 0.95. B/C evaluate inclusion of the truth; interval endpoints and widths are not computed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(path)


def pct(value):
    return "未计算" if value is None else f"{100*value:.1f}%"


def pp(value):
    return "未计算" if value is None else f"{100*value:+.1f} 个百分点"


def diagnostic_counts(folds):
    """Count explicitly marked bandwidth boundaries, with no guessing from h values."""
    counts = {"boundary": 0, "known": 0, "constant": 0}
    def walk(record):
        for key, value in record.items():
            lower = key.lower()
            if isinstance(value, dict):
                walk(value)
            elif "boundary" in lower and "side" not in lower:
                if isinstance(value, bool):
                    counts["known"] += 1
                    counts["boundary"] += int(value)
            elif "constant" in lower and isinstance(value, bool):
                counts["constant"] += int(value)
    for row in folds:
        walk(row)
    return counts


def decision_note(summary, pairs, folds, run_dir):
    complete = all(r["pending"] == 0 for r in summary)
    failures = sum(r["failed"] for r in summary)
    c_rows = [r for r in summary if r["method"] == METHODS[2]]
    diag = diagnostic_counts(folds)
    frequent_boundary = diag["known"] > 0 and diag["boundary"]/diag["known"] >= .2
    under = [SCENARIO_LABELS.get(r["scenario"], r["scenario"]) for r in c_rows if r["wilson_high"] is not None and r["wilson_high"] < .95]
    if not complete:
        lead = "筛查尚未完成，本文件是进度报告，不能据此决定是否更换选题。所有计划重复将按预先固定的配置完成，不能按中途 coverage 选停止点。"
    elif failures:
        lead = "本轮计划重复已完成尝试，但存在算法失败；先核查失败与实现，再决定选题。下述有效运行上的 coverage 与配对差不代表完整计划样本上的无条件性能。"
    elif frequent_boundary:
        lead = "本轮筛查已完成，但辅助带宽较多命中有限搜索边界，结果存在实现敏感性，暂不据此作更换选题的决定。最有价值的下一步是固定若干折，对照作者 R/np 实现核验带宽与完整 score。"
    elif under:
        lead = "本轮 Python 移植筛查中，C 在 " + "、".join(under) + " 的 Wilson 区间上界仍低于 95%，留下欠覆盖信号。须结合已完成的基准与方法验证讨论剩余问题，不能直接宣称发现新的研究问题，也不自动扩展到 Huber。"
    else:
        lead = "本轮特定三场景中，C 没有显示 Wilson 区间整体低于 95% 的明确欠覆盖信号。若相对 A/B 的改善与实现验证相符，可支持降低继续原选题的优先级；这不能证明等价于 95%，也不能证明全部重尾 DML 问题已经解决。"
    if complete and all(r["coverage_valid"] is not None for r in c_rows):
        absolute = "、".join(f"{SCENARIO_LABELS.get(r['scenario'], r['scenario'])} {pct(r['coverage_valid'])}" for r in c_rows)
        by_pair = {(r["scenario"], r["comparison"]): r for r in pairs}
        cb = [by_pair[(r["scenario"], "C-B")] for r in c_rows]
        ca = [by_pair[(r["scenario"], "C-A")] for r in c_rows]
        cb_text = "、".join(pp(r["mean_difference"]) for r in cb)
        cb_mcse = "、".join("未计算" if r["mcse"] is None else f"{100*r['mcse']:.2f} 个百分点" for r in cb)
        ca_text = "、".join(pp(r["mean_difference"]) for r in ca)
        evidence = f"已完成预先固定的 {sum(r['planned'] for r in c_rows)} 个数据集。C 的 coverage 依次为 {absolute}；相对 B 的配对改善依次为 {cb_text}（配对 MCSE：{cb_mcse}），相对 A 的配对差依次为 {ca_text}。"
        if len(under) == len(c_rows):
            evidence += "三个场景都仍明显远离 95%，这不支持“此实现已解决当前设计的欠覆盖”这一判断。"
        evidence += "\n\n"
        lead = evidence + lead
        if frequent_boundary:
            lead += "带宽边界频繁出现且尚无作者 R 软件对照，不能把本移植结果直接归结为原论文方法失败，也不能据此断言选题已经解决或必然值得继续。"
    lines = ["# 本轮选题筛查记录", "", lead, "", "## 绝对覆盖率与 Monte Carlo 不确定性", "",
             "A 是原 Mean-DML Wald 区间；B/C 是在真值 θ=1 处检验所对应反演置信集的包含率。下表 Wilson 95% 区间和 MCSE 均以有效运行数为分母。", "",
             "| 场景 | 方法 | 有效/已尝试/计划 | 失败/待运行 | coverage | MCSE | Wilson 95% | 全计划覆盖率缺失界限 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        label = "ABC"[METHODS.index(r["method"])]
        lines.append(f"| {SCENARIO_LABELS.get(r['scenario'], r['scenario'])} | {label} | {r['valid']}/{r['attempted']}/{r['planned']} | {r['failed']}/{r['pending']} | {pct(r['coverage_valid'])} | {pct(r['mcse'])} | [{pct(r['wilson_low'])}, {pct(r['wilson_high'])}] | [{pct(r['planned_coverage_lower'])}, {pct(r['planned_coverage_upper'])}] |")
    lines += ["", "尚未运行的重复属于 pending，不是算法失败。failure_rate 的分母是 attempted；所有计划运行完成时等于 (R−Rv)/R。全计划界限把失败及尚未运行的结果视为未知，不把它们记成成功覆盖。", "", "## 同一数据集上的配对比较", "",
              "| 场景 | 差值 | 有效配对 | 差值 | 配对 MCSE | d=−1/0/+1 次数 | 全计划保守界限 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {SCENARIO_LABELS.get(r['scenario'], r['scenario'])} | {r['comparison']} | {r['valid_pairs']} | {pp(r['mean_difference'])} | {pct(r['mcse'])} | {r['count_minus1']}/{r['count_zero']}/{r['count_plus1']} | [{pp(r['planned_difference_lower'])}, {pp(r['planned_difference_upper'])}] |")
    lines += ["", "A→C 包含学习目标变化与修正两个影响；B→C 才更直接反映修正步骤。差值只在共同有效的重复上逐对计算，MCSE 使用配对差的样本标准差。缺失配对的差值按 [−1,1] 给出保守界限；共同成功子集可能偏向较容易样本。少量不一致配对时，正态近似应谨慎。", "", "## 重尾相对 Gaussian 的变化", "",
              "以下是同方法 coverage_tail−coverage_Gaussian 的描述差，未提供跨场景差值标准误。不同误差家族不能仅凭 replication 编号被当作同一个数据集。", ""]
    for r in summary:
        if r["scenario"] != "gaussian":
            lines.append(f"- {SCENARIO_LABELS.get(r['scenario'],r['scenario'])}，{'ABC'[METHODS.index(r['method'])]}：{pp(r['coverage_minus_gaussian'])}。")
    lines += ["", "## 正确性与数值诊断", ""]
    validation = run_dir.parents[1] / "audit" / "validation.md"
    if validation.exists():
        lines += ["原实现一致性、相同样本与 folds、串并行复现和方法代数检查详见 [validation.md](../../audit/validation.md)；本汇总器不把单靠 coverage 接近旧汇总当作复现证据。"]
    else:
        lines += ["汇总时尚未找到 audit/validation.md，原基准一致性与方法验证状态不能由本报告确认；请先完成该项。"]
    lines += [f"本轮按方法统计的失败记录为 {failures} 条。带宽诊断中明确记录边界状态 {diag['known']} 项，命中边界 {diag['boundary']} 项；常数索引标记 {diag['constant']} 项。报告将边界比例 ≥20% 作为工程敏感性提醒（报告规则，不是新的显著性检验），不据此调带宽追逐 95%。", "",
              "本实现是明示搜索规则的 Python 核回归移植，与作者 R/np 带宽优化未做逐项软件复现；独立固定数组公式验证不能替代作者环境交叉核验。算法失败、非有限值与零 score 方差不会被记为接受真值。", "", "## 决策范围与后续价值", "",
              "本轮只评价真值是否被包含，尚未求置信集端点、宽度、信息量、功效或非连通性。高 coverage 也可能由过度保守产生，不能宣称推断问题全面解决。R=200 时 95% 附近的 MCSE 约 1.54 个百分点，0.93 与 0.95 不足以作确定区分；区间包含 0.95 不是等价证明。三场景属于探索性筛查，未进行多重比较的正式推断。", "",
              "如果边界或软件差异影响判断，优先核验作者实现；若只剩 coverage 精度不足，最小后续是补同一 n=1000 的 R。研究样本量趋势才考虑 n=2000，可推广性才考虑 DGP1。后续扩展需要用户选择投入方向，本轮不自动扩大实验或实现 Huber。", "",
              "Dukes 修正主要处理 nuisance estimation 带来的推断偏差，并非有界影响的重尾专用方法。其有效性仍依赖矩、收敛、经验过程与非退化等条件；有限场景模拟不能证明固定 HGBR 满足理论条件。t(2.1) 有有限方差而无四阶矩，lognormal 有全部正阶矩但有限样本可能非常不稳定。", ""]
    saved_path = run_dir.parents[1] / "audit" / "saved_score_verification.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8-sig"))
        if saved.get("status") == "passed" and Path(saved.get("run_dir", "")).resolve() == run_dir.resolve():
            counts = saved["counts"]
            maxima = saved["maximum_absolute_differences"]
            verification = ["", "## 保存数组的独立复核与带宽位置", "",
                            f"[saved_score_verification.json](../../audit/saved_score_verification.json) 对已保存的 {counts['tasks_read']} 个数据集进行了独立公式复核：{counts['successful_C_verified']} 个成功 C 结果、{counts['scores_checked']:,} 个逐点 score、{counts['folds_checked']:,} 折均通过。核验范围包括折内直接投影、完整 score 方差、T 及与卡方表达式一致的 p 值；没有重新拟合带宽，也没有运行 R。",
                            "",
                            f"score 最大绝对差 {maxima['score_absolute']:.3g}，T 最大绝对差 {maxima['T_absolute']:.3g}，p 值最大绝对差 {maxima['p_absolute']:.3g}；更新矩条件最大绝对残差 {maxima['update_moment_absolute']:.3g}。这支持保存结果与已声明公式一致，不能单独验证有限带宽搜索规则与作者软件等价。",
                            "", "| 场景 | 辅助方向 | 核回归数 | 搜索内部 | 下边界 | 上边界 | 常数索引 |", "|---|---|---:|---:|---:|---:|---:|"]
            for r in saved["bandwidth_counts"]:
                verification.append(f"| {SCENARIO_LABELS.get(r['scenario'], r['scenario'])} | {r['direction']} | {r['total']} | {r['interior']} | {r['lower']} | {r['upper']} | {r['constant_index']} |")
            verification += ["", "G 是结果残差对处理预测的核回归，M 是处理残差对变换结果预测的核回归。多数边界命中位于上边界，可能反映较平坦的辅助关系，也可能受有限优化范围影响；这里不擅自扩大搜索范围或重新挑选结果。最小后续应固定若干已保存折，让作者 R/np 选择带宽，再在相同带宽下比对逐点预测、更新和完整 score。", ""]
            insertion = lines.index("## 决策范围与后续价值")
            lines[insertion:insertion] = verification
    return "\n".join(lines)


def summarize_run(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    rows, folds = [], []
    for task_path in sorted((run_dir / "tasks").glob("*.json")):
        task = json.loads(task_path.read_text(encoding="utf-8-sig"))
        rows.extend(task["rows"])
        folds.extend(task.get("folds", []))
    summary = summarize_records(rows, config)
    pairs = paired_records(rows, config)
    rows.sort(key=lambda r: (r["scenario"], r["replication"], METHODS.index(r["method"])))
    write_csv(run_dir / "raw_results.csv", rows, ("scenario", "replication", "method", "status", "covered"))
    write_csv(run_dir / "fold_diagnostics.csv", folds)
    write_csv(run_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(run_dir / "paired_differences.csv", pairs, PAIR_FIELDS)
    write_csv(run_dir / "failures.csv", [r for r in rows if r["status"] == "failed"])
    plot_comparison(run_dir / "coverage_comparison.png", summary)
    atomic_text(run_dir / "decision_note.md", decision_note(summary, pairs, folds, run_dir))
    return summary, pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary, _ = summarize_run(args.run_dir)
    print(json.dumps({"report": str(args.run_dir.resolve()), "method_records_attempted": sum(r["attempted"] for r in summary), "method_records_pending": sum(r["pending"] for r in summary)}, ensure_ascii=False))

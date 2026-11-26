"""Generate stage-specific CSVs, Markdown, and exactly one vector PDF per scenario."""
import dgp
from estimators import method_names, variants
from population import benchmarks
from run import ROOT, db_path, signature
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","pdf.fonttype":42,
                     "font.size":10,"axes.spines.top":False,"axes.spines.right":False})
COLORS=["#3a3a3a","#804756","#87784a","#67885e","#707182","#1b6982","#669b9b","#aa572c","#42764b","#805c92"]

def records(root,mode):
    path=db_path(root.resolve(),mode)
    if not path.is_file():
        raise FileNotFoundError(f"No {mode} checkpoint at {path}")
    with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as db:
        assert db.execute("PRAGMA quick_check").fetchone()[0]=="ok"
        stored=db.execute("SELECT signature FROM meta WHERE id=1").fetchone()[0]
        if stored!=signature():
            raise ValueError("Checkpoint and scientific source versions differ")
        values=[json.loads(row[0]) for row in db.execute("SELECT json FROM replications ORDER BY scenario,n,rep")]
    expected={(sc,n,rep) for sc in dgp.SCENARIOS for n in dgp.SIZES for rep in range(dgp.COUNTS[mode])}
    actual={(row["scenario"],row["n"],row["rep"]) for row in values}
    if expected!=actual or len(values)!=len(expected):
        raise ValueError(f"Incomplete {mode} checkpoint: {len(actual&expected)}/{len(expected)}")
    keys=set()
    for row in values:
        sc,n,rep=row["scenario"],row["n"],row["rep"]
        assert row["mode"]==mode and row["theta0"]==dgp.target(sc)
        for stream,key in row["seeds"].items():
            assert key==dgp.seed_key(mode,sc,n,rep,stream) and tuple(key) not in keys
            keys.add(tuple(key))
        assert set(row["seeds"])==set(dgp.STREAMS)
        assert [m["name"] for m in row["methods"]]==list(method_names(sc))
        for m in row["methods"]:
            assert set(m["intervals"])==set(variants(m["name"],sc))
    return values

def flatten(values,targets):
    rows=[]
    for r in values:
        sc=r["scenario"]
        for m in r["methods"]:
            diag=m["diagnostics"]
            base=dict(mode=r["mode"],scenario=sc,n=r["n"],rep=r["rep"],
                      method=m["name"],theta0=r["theta0"],pseudo_true=targets[sc]["targets"].get(m["name"]),
                      point=m["point"],valid_point=m["valid_point"],seconds=m["seconds"],
                      nuisance_seconds=r["nuisance_seconds"],input_sha256=r.get("input_sha256"),
                      seed_key=json.dumps(r["seeds"],separators=(",",":")),
                      selected_L=diag.get("selected_L"),even_weight_share=diag.get("even_weight_share"),
                      even_one_step_increment=diag.get("even_one_step_increment"),
                      diagnostics_json=json.dumps(diag,separators=(",",":")))
            for version,ci in m["intervals"].items():
                rows.append(base|{"inference":version}|ci)
    frame=pd.DataFrame(rows)
    benchmark_coverage=(frame.lower<=frame.pseudo_true)&(frame.upper>=frame.pseudo_true)&frame.valid
    frame["covered_pseudo_true"]=benchmark_coverage.where(frame.pseudo_true.notna())
    return frame

def median_or_nan(series):
    return series.median() if series.notna().any() else np.nan

def summarize(frame):
    output=[]
    for (scenario,n,method,inference), g in frame.groupby(["scenario","n","method","inference"],sort=False):
        errors=g.point-g.theta0
        R=len(g)
        cover=g.covered.mean()
        sd=g.point.std()
        median_se=median_or_nan(g.se)
        output.append(dict(scenario=scenario,n=n,method=method,inference=inference,R=R,
            valid_points=int(g.valid_point.sum()),valid_intervals=int(g.valid.sum()),
            theta0=g.theta0.iloc[0],pseudo_true=g.pseudo_true.iloc[0],
            mean_estimate=g.point.mean(),mean_estimate_mcse=sd/np.sqrt(g.valid_point.sum()) if g.valid_point.any() else np.nan,
            bias=errors.mean(),bias_pseudo_true=(g.point-g.pseudo_true).mean(),
            rmse=np.sqrt((errors**2).mean()),median_absolute_error=median_or_nan(errors.abs()),
            empirical_sd=sd,coverage=cover,coverage_mcse=np.sqrt(cover*(1-cover)/R),
            coverage_pseudo_true=g.covered_pseudo_true.mean(),
            valid_only_coverage=g.loc[g.valid,"covered"].mean(),
            median_ci_length=median_or_nan(g.length),median_se=median_se,
            se_calibration_ratio=median_se/sd if sd>0 else np.nan,
            point_failure_rate=1-g.valid_point.mean(),failure_rate=1-g.valid.mean(),
            mean_runtime=g.seconds.mean()))
    return pd.DataFrame(output)

def save(fig,folder,name):
    fig.savefig(folder/f"{name}.pdf",format="pdf",bbox_inches="tight")
    fig.savefig(folder/f"{name}.png",dpi=155,bbox_inches="tight")
    plt.close(fig)

def size_axis(ax):
    ax.set(xscale="log",xlabel="Sample size")
    ax.set_xticks(dgp.SIZES,labels=dgp.SIZES)
    ax.set_xticks([],minor=True)
    ax.grid(axis="y",alpha=.20)

def plot_all(summary,flat,targets,folder,mode):
    subtitle=f"{mode.upper()}, $R={dgp.COUNTS[mode]}$ per cell"
    fig,axes=plt.subplots(1,3,figsize=(10.1,3.3),sharey=True,layout="constrained")
    for ax,name in zip(axes,("Rank-W","Rank-N","Rank-Ad")):
        s=summary[(summary.scenario=="D1")&(summary.method==name)]
        for var,c,mark in (("naive","#ab552e","o"),("sandwich","#196887","s")):
            t=s[s.inference==var].sort_values("n")
            ax.errorbar(t.n,t.coverage,yerr=1.96*t.coverage_mcse,color=c,marker=mark,
                        linewidth=1.5 if var=="naive" else 1.05,label=var.capitalize(),capsize=2.7,
                        mfc="white" if var=="naive" else c,ms=8 if var=="naive" else 3.4)
        ax.axhline(.95,color=".4",ls="--",lw=1)
        ax.set_title(name)
        size_axis(ax)
    axes[0].set_ylabel("95% Wald coverage")
    axes[-1].legend(frameon=False,loc="center right",fontsize=8)
    fig.suptitle(f"D1. X-dependent scale and rank inference ({subtitle})")
    save(fig,folder,"D1_Heteroskedasticity")

    fig,(ax,bench)=plt.subplots(1,2,figsize=(10,4.1),gridspec_kw={"width_ratios":[3.5,1.5]},
                               sharey=True,layout="constrained")
    main=summary[(summary.scenario=="D2")&summary.inference.isin(["wald","naive"])]
    for k,name in enumerate(method_names("D2")):
        s=main[main.method==name].sort_values("n")
        lw=2.4 if name in ("OLS","Rank-W") else 1.0
        ax.errorbar(s.n,s.mean_estimate,yerr=1.96*s.mean_estimate_mcse,color=COLORS[k],
                    marker="o",ms=3,lw=lw,label=name,capsize=2)
        ref=targets["D2"]["targets"].get(name)
        if ref is not None:
            bench.plot(k,ref,marker="D",ms=5,color=COLORS[k])
    ax.axhline(0.,color=".52",ls="--",lw=1)
    bench.axhline(0.,color=".52",ls="--",lw=1)
    size_axis(ax)
    ax.set_ylabel("Mean estimate")
    ax.legend(ncol=4,frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.22))
    bench.set_xticks(range(len(method_names("D2"))),method_names("D2"),rotation=65,ha="right",fontsize=8)
    bench.set_title("Population targets",fontsize=9)
    fig.suptitle(f"D2. Mirror skewness ({subtitle})")
    save(fig,folder,"D2_Mirror_Skewness")

    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.7,3.8),layout="constrained")
    for name,c in (("Odd-only",COLORS[8]),("Fixed mixed",COLORS[9]),("Rank-Ad",COLORS[7])):
        s=summary[(summary.scenario=="D3")&(summary.method==name)&(summary.inference=="naive")].sort_values("n")
        ax.errorbar(s.n,s.mean_estimate,yerr=1.96*s.mean_estimate_mcse,color=c,marker="o",
                    label=name,capsize=2.7)
    ax.axhline(0.,color=".4",ls="--",lw=1,label="Structural zero")
    ax.axhline(targets["D3"]["targets"]["Fixed mixed"],color=COLORS[9],ls=":",lw=1.2,label="Mixed one-step limit")
    ax.legend(frameon=False,fontsize=8)
    ax.set_ylabel("Mean estimate")
    ad=flat[(flat.scenario=="D3")&(flat.method=="Rank-Ad")&(flat.inference=="naive")]
    series=ad.groupby("n").even_weight_share
    ns=np.array(dgp.SIZES)
    med=series.median().reindex(ns).astype(float)
    lo=series.quantile(.25).reindex(ns).astype(float)
    hi=series.quantile(.75).reindex(ns).astype(float)
    bx.plot(ns,med,marker="o",color=COLORS[7],label="Median")
    bx.fill_between(ns,lo,hi,color=COLORS[7],alpha=.22,label="Central 50%")
    bx.set_ylabel("Selected squared even weight share")
    bx.set_ylim(bottom=0.)
    bx.legend(frameon=False,fontsize=8)
    size_axis(ax)
    size_axis(bx)
    fig.suptitle(f"D3. Odd and even scores ({subtitle})")
    save(fig,folder,"D3_Odd_Even_Scores")

def csv(frame,path):
    frame.to_csv(path,index=False,encoding="utf-8-sig",float_format="%.16g")

def build(root,mode):
    values=records(root,mode)
    targets=benchmarks()
    flat=flatten(values,targets)
    summary=summarize(flat)
    out=root/"results"/mode
    repdir=out/"replications"
    sumdir=out/"summary"
    figdir=out/"figures"
    for path in (repdir,sumdir,figdir,root/"reports"):
        path.mkdir(exist_ok=True,parents=True)
    csv(flat,repdir/"replications.csv")
    csv(flat[~flat.valid],repdir/"failures.csv")
    csv(summary,sumdir/"summary.csv")
    unique=flat.drop_duplicates(["scenario","n","rep","method"])
    ad=unique[unique.method.isin(["Rank-Ad","Odd-only"])]
    counts=ad.groupby(["scenario","n","method","selected_L"],dropna=False).size().rename("count").reset_index()
    csv(counts,sumdir/"adaptive_L_counts.csv")
    csv(ad[["scenario","n","rep","method","selected_L","even_weight_share","even_one_step_increment","diagnostics_json"]],
        repdir/"adaptive_diagnostics.csv")
    (sumdir/"population_benchmarks.json").write_text(json.dumps(targets,indent=2),encoding="utf-8")
    plot_all(summary,flat,targets,figdir,mode)
    assert len(list(figdir.glob("*.pdf")))==3
    rows=summary.groupby(["scenario","n"]).R.first()
    failures=int((~flat.valid).sum())
    text=[f"# Scenario D {mode.upper()} 结果",
          f"共 {len(values):,} 次重复，{len(rows)} 个条件，每条件 {dgp.COUNTS[mode]} 次。"
          f"点估计和区间分别记录。{failures} 条区间无效，均计入主要覆盖率分母。"
          "有效区间的条件覆盖率同时保存在汇总 CSV。",
          "## D1 误差尺度由 X 决定",
          "Rank-W、Rank-N 和 Rank-Ad 逐次同时保存 naive 和 sandwich 区间。图中的误差棒表示 ±1.96 个 Monte Carlo 标准误。"
          "处理扰动 v 与误差独立，两个方差口径有可能接近。应结合 SE 校准比和区间长度阅读。",
          "## D2 镜像偏度",
          f"论文尺度的零点分数为 {targets['D2']['paper_wilcoxon_score_at_zero']:.9f}。"
          f"OLS 总体目标为 0，直接 Rank-W 的重排秩目标为 {targets['D2']['targets']['Rank-W']:.9f}。"
          "基准文件也列出固定 F₀ 的分数根。Rank-N 和 Rank-Ad 的基准按其一步更新计算。",
          "## D3 处理相关的对称尺度",
          f"固定 J₁+J₂ 的一步目标为 {targets['D3']['targets']['Fixed mixed']:.9f}，"
          f"相应重排秩方程的根为 {targets['D3']['rank_score_roots']['Fixed mixed']['reranked']['root']:.9f}。"
          "原 Rank-Ad 仍按估计斜率加权。保存偶数阶所占平方权重、偶数项的一步增量及选阶分布。",
          "## 输出",
          f"- [汇总 CSV](../results/{mode}/summary/summary.csv)\n"
          f"- [逐次 CSV](../results/{mode}/replications/replications.csv)\n"
          f"- [失败 CSV](../results/{mode}/replications/failures.csv)\n"
          f"- [总体目标与积分核验](../results/{mode}/summary/population_benchmarks.json)\n"+
          "\n".join(f"- [{name} PDF](../results/{mode}/figures/{name}.pdf)" for name in
                    ("D1_Heteroskedasticity","D2_Mirror_Skewness","D3_Odd_Even_Scores"))]
    compact=summary[(summary.n==2000)&summary.inference.isin(["wald","naive","sandwich"])]
    def f(x):
        return f"{x:.4f}" if np.isfinite(x) else "—"
    headers="| 场景 | 方法 | 方差口径 | 均值 | RMSE | 覆盖率 | MCSE | 中位区间长 | 校准比 |"
    rows=["|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in compact.itertuples():
        rows.append(f"| {row.scenario} | {row.method} | {row.inference} | {f(row.mean_estimate)} | "
                    f"{f(row.rmse)} | {row.coverage:.2%} | {row.coverage_mcse:.2%} | "
                    f"{f(row.median_ci_length)} | {f(row.se_calibration_ratio)} |")
    text.insert(-2,"## n=2000 汇总\n\n"+headers+"\n"+"\n".join(rows))
    if mode!="main":
        text.insert(2,"本阶段用于检查实现。正式结果使用独立的 main 记录和报告。")
    else:
        text.insert(2,"[正式汇总 CSV](../results/main/summary/summary.csv) 包含真值与总体目标两套误差指标。"
                       "D2、D3 的独立误差公式区间作为适用边界诊断。")
    filename="Scenario_D_Results.md" if mode=="main" else f"Scenario_D_{mode.capitalize()}.md"
    (root/"reports"/filename).write_text("\n\n".join(text)+"\n",encoding="utf-8")
    print(f"Exported {len(flat)} interval rows and 3 vector PDFs to {out}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=dgp.MODES,required=True)
    p.add_argument("--output-root",type=Path,default=ROOT)
    a=p.parse_args()
    build(a.output_root,a.mode)


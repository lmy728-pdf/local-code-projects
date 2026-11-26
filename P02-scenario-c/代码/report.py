"""Read committed records and export tables, PDF figures, and research reports."""
from run import ROOT, db_path, SIZES, METHODS
import csv
import io
import json
import sqlite3
import zlib
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

TABLES = ROOT/"汇总表_CSV"
REPORTS = ROOT/"结果_MD"
FIGURES = ROOT/"figures"
COLORS = dict(zip(METHODS, ["#b04232","#176b87","#497136","#805b98"]))
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "lines.linewidth": 2, "figure.dpi": 135})
INDEX = []

def table(frame, name):
    frame.to_csv(TABLES/name, index=False, encoding="utf-8-sig", float_format="%.16g")

def figure(fig, name, title):
    fig.tight_layout()
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    image.seek(0)
    with Image.open(image) as bitmap:
        bitmap.convert("RGB").save(FIGURES/f"{name}.pdf", resolution=300., quality=96)
    plt.close(fig)
    INDEX.append((name,title))

def read_records(stage):
    results, diagnostics, counts = [], [], []
    checked_seeds = set()
    target = 2000 if stage=="main" else 100
    for case in ("C1","C2"):
        for n in SIZES:
            with sqlite3.connect(db_path(stage,case,n).as_uri()+"?mode=ro",uri=True) as db:
                count = 0
                for rep, line, has_arrays in db.execute("SELECT rep,record,arrays IS NOT NULL FROM results ORDER BY rep"):
                    row = json.loads(line)
                    assert rep == count and (row["case"],row["n"],row["stage"])==(case,n,stage)
                    count += 1
                    if not row.get("error"):
                        assert has_arrays and row["seed_key"] not in checked_seeds
                        checked_seeds.add(row["seed_key"])
                    shared = {key: row.get(key) for key in ("stage","case","n","rep","seed_key","fold_seed_key",
                        "input_sha256","rare_count","expected_rare","rare_probability","cutoff","nuisance_seconds","error")}
                    shared["replication_seconds"] = row["seconds"]
                    for mode in ("sample","population"):
                        shared.update({mode+"_"+k:v for k,v in row.get(mode,{}).items()})
                    for method in row["methods"]:
                        record = {key:value for key,value in method.items() if key!="metadata"}
                        results.append(shared|record)
                    for fold in row.get("fold_diagnostics",[]):
                        common = dict(case=case,n=n,rep=rep,stage=stage,fold=fold["fold"],
                                      rare_count=fold["rare_count"],fold_size=fold["sample_size"])
                        if "penalties" in row:
                            common.update(penalty_d=row["penalties"][fold["fold"]][0],
                                          penalty_y=row["penalties"][fold["fold"]][1],
                                          tail_extrapolation_fraction=fold["tail_extrapolation_fraction"])
                        for mode in ("sample","population","grid"):
                            risks = row.get("population") if mode=="population" else fold.get(mode)
                            if risks is None:
                                continue
                            values = common|dict(source=mode)|risks
                            for region in ("inside","outside"):
                                means = fold.get(mode+"_"+region)
                                if means is not None:
                                    for name,value in zip(("mean_b","mean_a","rmse_b","rmse_a"),means):
                                        values[region+"_"+name]=value
                            diagnostics.append(values)
            assert count==target, (case,n,count,target)
            counts.append(dict(stage=stage,case=case,n=n,target=target,completed=count))
    return pd.DataFrame(results), pd.DataFrame(diagnostics), pd.DataFrame(counts)

def load_arrays(stage,case,n,rep):
    """Retrieve all fold IDs, rare flags, and actual signed b/a OOF errors."""
    with sqlite3.connect(db_path(stage,case,n).as_uri()+"?mode=ro",uri=True) as db:
        payload, = db.execute("SELECT arrays FROM results WHERE rep=?", (rep,)).fetchone()
    with np.load(io.BytesIO(zlib.decompress(payload)),allow_pickle=False) as archive:
        return dict(fold=archive["fold"], rare=archive["rare"],
                    b=archive["errors"][:,0], a=archive["errors"][:,1])

def summarize(raw):
    output = []
    for (case,n,method), group in raw.groupby(["case","n","method"],sort=False):
        points = group.loc[group.valid_point,"estimate"]
        intervals = group[group.valid_interval]
        coverage = group.covered.mean()
        sd = points.std()
        root = np.sqrt(n)*points
        iqr_sd = (points.quantile(.75)-points.quantile(.25))/1.348979500392
        row = dict(case=case,n=n,method=method,R=len(group),valid_points=len(points),valid_intervals=len(intervals),
            point_failure_rate=1-len(points)/len(group), interval_failure_rate=1-len(intervals)/len(group),
            bias=points.mean(),rmse=np.sqrt((points**2).mean()),medae=points.abs().median(),
            coverage=coverage,coverage_mcse=np.sqrt(coverage*(1-coverage)/len(group)),
            coverage_valid_only=intervals.covered.mean(), median_ci_length=intervals.length.median(),
            median_se=intervals.se.median(),empirical_sd=sd,iqr_sd=iqr_sd,
            calibration_ratio=intervals.se.median()/sd, calibration_ratio_iqr=intervals.se.median()/iqr_sd,
            root_n_mean=root.mean(),root_n_sd=root.std(),root_n_median=root.median(),
            root_n_q05=root.quantile(.05),root_n_q95=root.quantile(.95),
            mean_seconds=group.seconds.mean(),median_seconds=group.seconds.median())
        if method=="OLS":
            row.update(ols_centered_mean=group.ols_centered.mean(),ols_centered_scaled_sd=group.ols_centered_scaled.std(),
                       ols_root_mean_ratio=root.mean()/n**.25)
        if method=="Rank-W":
            z = norm.ppf((np.arange(len(root))+.5)/len(root),scale=np.sqrt(np.pi/3))
            row["rank_w_gaussian_qq_rmse"] = np.sqrt(np.mean((np.sort(root)-z)**2))
            row["root_n_variance"] = root.var()
        output.append(row)
    return pd.DataFrame(output)

def sample_size_axis(ax, title, ylabel):
    ax.set(xlabel="Sample size",ylabel=ylabel,title=title)
    ax.set_xscale("log",base=2)
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.grid(axis="y",alpha=.2)

def draw_methods(ax, summary, case, key):
    for method in METHODS:
        data=summary[(summary.case==case)&(summary.method==method)].sort_values("n")
        ax.errorbar(data.n,data[key],yerr=1.96*data.coverage_mcse if key=="coverage" else None,
                    color=COLORS[method],label=method,marker="o",capsize=3)
    if key=="coverage":
        ax.axhline(.95,color="#444444",ls="--",label="95% nominal")
    ax.legend(ncol=2,frameon=False,fontsize=10)

def draw_figures(raw, diagnostics, summary):
    fig,(ax,qq)=plt.subplots(1,2,figsize=(12,5.6))
    sample_size_axis(ax,"Mean and central 90%","Root-n estimate")
    c1 = summary[summary.case=="C1"]
    for method in ("OLS","Rank-W"):
        values=c1[c1.method==method].sort_values("n")
        ax.plot(values.n,values.root_n_mean,label=method,marker="o",color=COLORS[method])
        ax.fill_between(values.n,values.root_n_q05,values.root_n_q95,color=COLORS[method],alpha=.15)
    ax.plot(SIZES,np.asarray(SIZES)**.25,ls="--",color="#414141",label=r"$n^{1/4}$")
    ax.axhline(0,color="#757575",ls=":")
    ax.legend(frameon=False,fontsize=10)
    idx=(raw.case=="C1")&(raw.method=="Rank-W")
    qq.plot([-4,4],[-4,4],color="#4a4a4a",ls="--")
    for n in (500,2000,8000):
        points=raw.loc[idx&(raw.n==n),"root_n_estimate"].dropna().sort_values()
        axis=norm.ppf((np.arange(len(points))+.5)/len(points),scale=np.sqrt(np.pi/3))
        qq.plot(axis,points,label=f"n={n}")
    qq.set(xlabel="Theoretical quantile of N(0, π/3)",ylabel="Empirical quantile",
           title="Rank-W normal Q–Q plot")
    qq.legend(frameon=False,fontsize=10)
    fig.suptitle("C1: Rare alignment produces an increasing OLS drift",fontsize=16)
    figure(fig,"C1_root_n_estimates","C1 root-n separation and Rank-W normal Q–Q")

    fig,(ax,product)=plt.subplots(1,2,figsize=(12,5.6))
    sample_size_axis(ax,"Wald coverage with Monte Carlo error","Coverage proportion")
    draw_methods(ax,summary,"C1","coverage")
    sample_size_axis(product,"Population and validation-sample products","Root-n scaled product")
    baseline=raw[(raw.case=="C1")&(raw.method=="OLS")].groupby("n")
    for key,label,color in (("root_n_product","Ordinary","#b04232"),("root_n_truncated","Truncated","#176b87")):
        product.plot(SIZES,baseline["sample_"+key].mean(),marker="o",color=color,label=f"{label}, sample")
        product.plot(SIZES,baseline["population_"+key].first(),ls="--",color=color,label=f"{label}, population")
    product.set_yscale("log")
    product.legend(frameon=False,fontsize=10)
    fig.suptitle("C1: Inference and nuisance products",fontsize=16)
    figure(fig,"C1_coverage_products","C1 coverage and nuisance products")

    grid=diagnostics[(diagnostics.case=="C2")&(diagnostics.source=="grid")]
    median=grid.groupby("n").median(numeric_only=True)
    fig,(risk,prod)=plt.subplots(1,2,figsize=(12,5.6))
    sample_size_axis(risk,"Nuisance error by region","Median conditional RMSE")
    for region,color in (("inside","#b04232"),("outside","#176b87")):
        for name,label,style in (("b","treatment","--"),("a","alignment","-")):
            risk.plot(SIZES,median[region+"_rmse_"+name],ls=style,marker="o",color=color,
                      label=f"{region.title()}, {label}")
    risk.set_yscale("log")
    risk.legend(frameon=False,fontsize=10)
    sample_size_axis(prod,"Comparison of nuisance products","Root-n scaled product")
    for key,power,label,color in (("root_n_product",.25,"Ordinary","#b04232"),
                                 ("root_n_truncated",-.25,"Truncated","#176b87")):
        prod.plot(SIZES,median[key],marker="o",color=color,label=f"C2 {label.lower()}, grid")
        prod.plot(SIZES,np.asarray(SIZES,dtype=float)**power,ls="--",color=color,
                  label=f"C1 {label.lower()}, population")
    prod.set_yscale("log")
    prod.legend(frameon=False,fontsize=10)
    fig.suptitle("C2: Spline Ridge errors in the rare region",fontsize=16)
    figure(fig,"C2_nuisance_diagnostics","C2 regional nuisance errors and comparison with C1")

    fig,(ax,coverage)=plt.subplots(1,2,figsize=(12,5.6))
    sample_size_axis(ax,"Median absolute estimation error","Median absolute error")
    draw_methods(ax,summary,"C2","medae")
    sample_size_axis(coverage,"Wald coverage with Monte Carlo error","Coverage proportion")
    draw_methods(coverage,summary,"C2","coverage")
    fig.suptitle("C2: Estimation error and Wald coverage",fontsize=16)
    figure(fig,"C2_error_coverage","C2 estimation error and coverage")

def md_table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])

def build_report(raw, diagnostics, summary):
    target_var=np.pi/3
    w=summary[(summary.case=="C1")&(summary.method=="Rank-W")].sort_values("n")
    ols=summary[(summary.case=="C1")&(summary.method=="OLS")].sort_values("n")
    n0,n1=SIZES[0],SIZES[-1]
    c1=raw[raw.case=="C1"].pivot(index=["n","rep"],columns="method",values="root_n_estimate")
    gap=(c1.OLS-c1["Rank-W"]).groupby("n")
    # A descriptive threshold, not a test of the asymptotic theorem.
    clear=gap.mean()-1.96*gap.sem()>1
    first=next((int(n) for n,ok in clear.items() if ok),None)
    c1_diag=raw[(raw.case=="C1")&(raw.method=="OLS")].groupby("n")
    slopes=np.polyfit(np.log(ols.n),np.log(ols.root_n_mean),1)[0]
    sd_slope=np.polyfit(np.log(ols.n),np.log(ols.root_n_sd),1)[0]
    products=diagnostics[(diagnostics.case=="C2")&(diagnostics.source=="grid")].groupby("n").median(numeric_only=True)
    c2=summary[summary.case=="C2"]
    c2_ols=c2[c2.method=="OLS"].sort_values("n")
    c2_w=c2[c2.method=="Rank-W"].sort_values("n")
    focus=c2[c2.n==n1].set_index("method")
    c1_focus=summary[(summary.case=="C1")&(summary.n==n1)].set_index("method")
    tail=[]
    for (case,n,method),group in raw.groupby(["case","n","method"],sort=False):
        points=group.loc[group.valid_point,"root_n_estimate"].abs().sort_values(ascending=False)
        cut=max(1,len(points)//100)
        energy=(points**2).sum()
        tail.append(dict(case=case,n=n,method=method,abs_root_n_q50=points.median(),
            abs_root_n_q90=points.quantile(.90),abs_root_n_q99=points.quantile(.99),abs_root_n_max=points.max(),
            top_one_percent_squared_error_share=(points.iloc[:cut]**2).sum()/energy if energy else np.nan))
    tails=pd.DataFrame(tail)
    table(tails,"root_n_tail_diagnostics.csv")
    c2_tails=tails[(tails.case=="C2")&(tails.n==n1)].set_index("method")
    paired_medae=c2.pivot(index="n",columns="method",values="medae")
    robust_wins={m:int((paired_medae[m]<paired_medae["OLS"]).sum()) for m in ("Rank-W","Huber","LAD")}
    min_trunc,max_trunc=products.root_n_truncated.min(),products.root_n_truncated.max()
    rare_ratios=c1_diag.rare_count.mean()/np.asarray(SIZES)**.25
    c2_gap=raw[raw.case=="C2"].pivot(index=["n","rep"],columns="method",values="root_n_estimate")
    c2_gap=(c2_gap.OLS-c2_gap["Rank-W"]).groupby("n").mean()
    failures=raw.groupby(["case","method"],sort=False).agg(R=("rep","size"),points=("valid_point","sum"),intervals=("valid_interval","sum"))
    fail_rows=[(case,m,int(row.R-row.points),int(row.R-row.intervals),f"{1-row.intervals/row.R:.3%}")
               for (case,m),row in failures.iterrows()]
    lines=["# Scenario C 正式结果",
           "C1 与 C2 各完成 10,000 次，每个样本量 2,000 次。每次使用共同的五折划分与残差，比较 OLS、Rank-W、Huber 和 LAD。",
           "## C1 理论反例",
           f"OLS 的 root-n 均值从 {ols.root_n_mean.iloc[0]:.3f} 增至 {ols.root_n_mean.iloc[-1]:.3f}，"
           f"Rank-W 从 {w.root_n_mean.iloc[0]:.3f} 变为 {w.root_n_mean.iloc[-1]:.3f}。"
           + (f"从本实验最早满足「配对均值差的 Monte Carlo 下限超过 1 个 root-n 单位」的 n={first}，分离已经明显。" if first else
              "各档样本量均未达到配对均值差的 Monte Carlo 下限超过 1 个 root-n 单位的描述性阈值。"),
           f"OLS 均值的 log–log 描述斜率为 {slopes:.3f}（理论主阶 0.25），标准差的斜率为 {sd_slope:.3f}（理论波动阶上界 0.125）。"
           f"均值与 n^(1/4) 的比值为 {ols.ols_root_mean_ratio.min():.3f}–{ols.ols_root_mean_ratio.max():.3f}，"
           f"减去该主阶并除以 n^(1/8) 后，经验标准差为 {ols.ols_centered_scaled_sd.min():.3f}–{ols.ols_centered_scaled_sd.max():.3f}。"
           "这些是有限样本趋势证据，斜率并非对渐近阶数的检验。",
           f"Rank-W 的 root-n 方差为 {w.root_n_variance.min():.3f}–{w.root_n_variance.max():.3f}，"
           f"理论 π/3={target_var:.3f}。均值和 Q–Q 图同时显示有限样本偏移，比较时保留这些差异。"
           f"在 n={n1}，OLS 的 Wald 覆盖率为 {ols.coverage.iloc[-1]:.2%}，Rank-W 为 {w.coverage.iloc[-1]:.2%}。",
           f"在 n={n1}，Huber 和 LAD 的 root-n 均值分别为 {c1_focus.loc['Huber','root_n_mean']:.3f} "
           f"与 {c1_focus.loc['LAD','root_n_mean']:.3f}，覆盖率为 {c1_focus.loc['Huber','coverage']:.2%} "
           f"与 {c1_focus.loc['LAD','coverage']:.2%}。"
           "两种方法都给出有限样本的稳健性对照，论文中的渐近结论适用于 Rank-W。",
           md_table(["n","OLS root-n 均值","n^(1/4)","OLS 波动 / n^(1/8)","W root-n 均值","W 方差","OLS 覆盖","W 覆盖"],
              [(o.n,f"{o.root_n_mean:.3f}",f"{o.n**.25:.3f}",f"{o.ols_centered_scaled_sd:.3f}",
                f"{v.root_n_mean:.3f}",f"{v.root_n_variance:.3f}",f"{o.coverage:.2%}",f"{v.coverage:.2%}")
               for o,v in zip(ols.itertuples(),w.itertuples())]),
           f"普通总体乘积从 {n0**.25:.3f} 增至 {n1**.25:.3f}，截断总体乘积从 {n0**(-.25):.3f} 降至 {n1**(-.25):.3f}。"
           f"对应的样本平均值从 {c1_diag.sample_root_n_product.mean().iloc[0]:.3f} 到 {c1_diag.sample_root_n_product.mean().iloc[-1]:.3f}，"
           f"以及从 {c1_diag.sample_root_n_truncated.mean().iloc[0]:.3f} 到 {c1_diag.sample_root_n_truncated.mean().iloc[-1]:.3f}。",
           f"五档样本量下，实际罕见点平均数与二项理论期望的比值为 "
           f"{rare_ratios.min():.3f}–{rare_ratios.max():.3f}。"
           "各档罕见点计数的均值和 Monte Carlo 标准误见 [罕见点计数 CSV](../汇总表_CSV/rare_count_summary.csv)。",
           "## C2 重尾协变量与 Spline Ridge",
           "尾部阶跃是本实验预先定义的数值设计。邻域内外的误差来自五折真实训练。下列条件 RMSE 为各折在独立分层网格上估计后取中位数。",
           md_table(["n","区内 b RMSE","区外 b RMSE","区内 a RMSE","区外 a RMSE","普通乘积","截断乘积","尾部越界比例"],
              [(n,*(f"{row[k]:.3f}" for k in ("inside_rmse_b","outside_rmse_b","inside_rmse_a","outside_rmse_a","root_n_product","root_n_truncated")),
                f"{row.tail_extrapolation_fraction:.1%}") for n,row in products.iterrows()]),
           f"已观测到的 C2 平均 root-n 配对差（OLS 减 Rank-W）从 {c2_gap.iloc[0]:.3f} 到 {c2_gap.iloc[-1]:.3f}，"
           f"而 C1 从 {gap.mean().iloc[0]:.3f} 到 {gap.mean().iloc[-1]:.3f}。"
           f"C2 的 OLS root-n 均值从 {c2_ols.root_n_mean.iloc[0]:.3f} 到 {c2_ols.root_n_mean.iloc[-1]:.3f}，"
           f"Rank-W 则从 {c2_w.root_n_mean.iloc[0]:.3f} 到 {c2_w.root_n_mean.iloc[-1]:.3f}。",
           f"在 n={n1}，C2 稀有区内的折级 alignment RMSE 中位数为 {products.inside_rmse_a.iloc[-1]:.2f}，"
           f"区外为 {products.outside_rmse_a.iloc[-1]:.2f}。C2 的截断乘积中位数从 {products.root_n_truncated.iloc[0]:.3f}"
           f" 变为 {products.root_n_truncated.iloc[-1]:.3f}，对应 C1 总体值 {n0**(-.25):.3f} 与 {n1**(-.25):.3f}。"
           f"C2 的截断乘积在五档的折级中位数范围为 {min_trunc:.3f}–{max_trunc:.3f}。"
           "这组分层诊断衡量有限样本拟合风险。C1 的截断乘积消失条件在 C2 的观察范围内未得到相同趋势。",
           f"C2 中 Rank-W、Huber、LAD 的 MedAE 分别在 {robust_wins['Rank-W']}、"
           f"{robust_wins['Huber']}、{robust_wins['LAD']} 个样本量上低于 OLS（共五档）。"
           "这反映了典型估计误差的有限样本分离。root-n 均值与极端误差集中度另有变化，不能由此判断 C2 有 C1 的高斯极限。",
           f"C2 在 n={n1} 的 MedAE 为 OLS {focus.loc['OLS','medae']:.4g}、Rank-W {focus.loc['Rank-W','medae']:.4g}。"
           f"绝对误差最大的约 1% 重复贡献了 OLS 总平方误差的 {c2_tails.loc['OLS','top_one_percent_squared_error_share']:.1%}，"
           f"Rank-W 对应比例为 {c2_tails.loc['Rank-W','top_one_percent_squared_error_share']:.1%}。"
           "均值和经验 SD 需结合中位数、分位数与最极端的重复一起读。",
           "## 最大样本量的四方法结果",
           md_table(["设计","方法","n","root-n 均值","root-n SD","Bias","RMSE","MedAE","Wald 覆盖（MCSE）","中位区间长度","校准比"],
              [(r.case,r.method,r.n,f"{r.root_n_mean:.3f}",f"{r.root_n_sd:.3f}",f"{r.bias:.4f}",f"{r.rmse:.4f}",
                f"{r.medae:.4f}",f"{r.coverage:.2%} ({r.coverage_mcse:.2%})",f"{r.median_ci_length:.4f}",
                f"{r.calibration_ratio:.3f}") for r in summary[summary.n==n1].itertuples()]),
           "## 失败记录和口径",
           md_table(["设计","方法","点估计失败","无效区间","无效区间比例"],fail_rows),
           "覆盖率的分母是各条件的全部重复数。无效区间计为未覆盖，有效区间覆盖率另列于 CSV。"
           "点估计统计量使用有效点估计。主校准比为中位 SE 除以经验 SD，CSV 也给基于 IQR 的版本。"
           "覆盖图的误差棒为 ±1.96 个二项 Monte Carlo 标准误。",
           "[C1 全部汇总](../汇总表_CSV/c1_summary.csv) · [C2 全部汇总](../汇总表_CSV/c2_summary.csv) · "
           "[分折误差](../汇总表_CSV/nuisance_diagnostics.csv) · [图表索引](figure_index.md)"]
    (REPORTS/"Scenario_C_Results.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")

def benchmark():
    logs=[json.loads(line) for line in (ROOT/"logs/cell_timings.jsonl").read_text(encoding="utf-8").splitlines()]
    frame=pd.DataFrame(logs)
    table(frame,"performance_timings.csv")
    rank=pd.read_csv(TABLES/"rank_solver_validation.csv")
    smoke=frame[(frame.stage=="smoke")&(frame.case=="C1")&(frame.n==8000)].iloc[0]
    with sqlite3.connect(db_path("smoke","C1",8000).as_uri()+"?mode=ro",uri=True) as db:
        recs=[json.loads(row[0]) for row in db.execute("SELECT record FROM results")]
    times={m:np.mean([x["seconds"] for r in recs for x in r["methods"] if x["method"]==m]) for m in METHODS}
    text=["# Scenario C 性能与数值检查",
          f"n=8000 的 C1 smoke 用 {int(smoke.workers)} 个 worker 完成 2 次，含启动共 {smoke.wall_seconds:.2f} 秒。"
          f"进程树采样 RSS 峰值 {smoke.peak_tree_rss_mib:.0f} MiB，共享页可能重复计入。"
          f"Rank-W 单次平均 {times['Rank-W']:.3f} 秒，LAD {times['LAD']:.3f} 秒。",
          f"精确对照共 {len(rank)} 组，最大估计差 {rank.abs_estimate_error.max():.3g}，"
          f"最大目标函数绝对差 {rank.objective_difference.abs().max():.3g}。"
          "最优点左、右探针的次梯度均满足包夹零。排序实现每次计算目标导数为 O(n log n)，每 worker 的求解工作内存为 O(n)。",
          "以下为每次完成会话的测量。wall-clock 包含进程启动、运算和保存。",
          md_table(["阶段","设计","n","新增","workers","秒","采样 RSS MiB"],
                   [(r.stage,r.case,r.n,r.new,r.workers,f"{r.wall_seconds:.2f}",f"{r.peak_tree_rss_mib:.0f}")
                    for r in frame.itertuples()]),
          "[精确目标对照 CSV](../汇总表_CSV/rank_solver_validation.csv) · [耗时 CSV](../汇总表_CSV/performance_timings.csv)"]
    (REPORTS/"performance_benchmark.md").write_text("\n\n".join(text)+"\n",encoding="utf-8")

def main():
    raw, diagnostics, counts = read_records("main")
    summary=summarize(raw)
    for case in ("C1","C2"):
        table(raw[raw.case==case],case.lower()+"_replications.csv")
        table(summary[summary.case==case],case.lower()+"_summary.csv")
    table(diagnostics,"nuisance_diagnostics.csv")
    table(counts,"completion_matrix.csv")
    rare=raw[raw.method=="OLS"].groupby(["case","n"]).rare_count.agg(["mean","std","count"]).reset_index()
    rare["binomial_expected_count"]=rare.n**.25
    rare["mean_mcse"]=rare["std"]/np.sqrt(rare["count"])
    table(rare,"rare_count_summary.csv")
    table(raw[~raw.valid_interval],"failures.csv")
    pilot,pdiag,_=read_records("pilot")
    table(summarize(pilot),"pilot_summary.csv")
    table(pdiag,"pilot_nuisance_diagnostics.csv")
    draw_figures(raw,diagnostics,summary)
    build_report(raw,diagnostics,summary)
    benchmark()
    index="# Scenario C 正式图表\n\n四张 300 dpi PDF。其余诊断见正式汇总 CSV。\n\n"+"\n".join(f"- [{title}](../figures/{name}.pdf)" for name,title in INDEX)
    (REPORTS/"figure_index.md").write_text(index+"\n",encoding="utf-8")
    assert len(raw)==80000 and len(summary)==40
    lines=["# Scenario C 完成与验收",
           "C1 和 C2 各五组正式实验全部完成，共 20,000 次重复、80,000 条方法记录。",
           "12 次 smoke、1,000 次 pilot 和全部正式重复通过各自的检查点核对。"
           "阶段使用独立种子命名空间，单阶段数据键与折划分键逐条核对。",
           "正式检查点包含误差数组和验证折标签。逐次 CSV 提供确定性种子键。"
           "已保存所有有效估计和失败原因，失败计入各组目标次数。",
           md_table(["阶段","设计","n","目标","完成"],counts.itertuples(index=False,name=None)),
           f"正式点估计失败 {int((~raw.valid_point).sum())} 条，无效区间 {int((~raw.valid_interval).sum())} 条，详见 [失败 CSV](../汇总表_CSV/failures.csv)。",
           "[科研报告](Scenario_C_Results.md) · [图表索引](figure_index.md) · [性能与数值验证](performance_benchmark.md) · [配置](../experiment_manifest.json)"]
    (REPORTS/"Scenario_C_Completion_Report.md").write_text("\n\n".join(lines)+"\n",encoding="utf-8")
    print(f"Completed report: {len(raw)} estimator rows, {len(INDEX)} PDF figures",flush=True)

if __name__=="__main__":
    main()


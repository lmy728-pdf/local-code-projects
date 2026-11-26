"""Reports always recomputed from replication records; smoke/pilot separated."""
from pathlib import Path
import sqlite3,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from methods import LAWS,METHODS,rng
ROOT=Path(__file__).resolve().parents[1]
def summarize(stage,condition=None,record_sources=None):
    if condition:
        from layout import paths
        layout=paths(*condition,stage);folder=layout["work"]
        output_base=layout["root"];reportdir=layout["reports"]
    else:
        folder=ROOT/(stage if stage!="main" else "replications")
        output_base=ROOT;reportdir=ROOT/"结果_MD"
    records=[]
    keep=("n","law","replication","source","rows","diagnostics","input_hash","seed_key","seconds","nuisance_seconds")
    mk=("failure","selected_L","candidates","uncorrected_information","raw_corrected_information","corrected_information","candidate_se")
    sources=record_sources if record_sources is not None else [folder/"records.sqlite"]
    for source in sources:
        with sqlite3.connect(Path(source).as_uri()+"?mode=ro",uri=True) as db:
            for item in db.execute("SELECT payload FROM records ORDER BY n,law,rep"):
                rec=json.loads(item[0]);slim={k:rec[k] for k in keep if k in rec}
                for result in slim["rows"]:
                    result["metadata"]={k:v for k,v in result["metadata"].items() if k in mk}
                records.append(slim)
    if not records:raise RuntimeError("No completed records to summarize")
    identities=[(r["n"],r["law"],r["replication"]) for r in records]
    if len(set(identities))!=len(identities):raise RuntimeError("Duplicate replication identities in report sources")
    rows=[];candidates=[];diagnostics=[]
    for rec in records:
        n,law,rep=rec["n"],rec["law"],rec["replication"]
        for result in rec["rows"]:
            meta=result["metadata"]
            rows.append(dict(n=n,law=law,rep=rep,source=rec["source"],method=result["method"],
                **{k:v for k,v in result.items() if k not in ("metadata","method")},
                failure=meta.get("failure",""),selected_L=meta.get("selected_L"),
                input_hash=rec.get("input_hash"),seed_key=rec.get("seed_key")))
            if result["method"]=="Rank-Ad":
                for L in range(1,7):
                    row=dict(n=n,law=law,rep=rep,L=L,selected=meta.get("selected_L")==L)
                    for key in ("candidates","uncorrected_information","raw_corrected_information","corrected_information","candidate_se"):
                        row[key]=meta.get(key,[None]*6)[L-1]
                    row["covered"]=abs(row["candidates"]-1)<=1.95996398454*row["candidate_se"] if row["candidates"] is not None and row["candidate_se"] is not None else None
                    candidates.append(row)
        for k,d in enumerate(rec.get("diagnostics",[])):
            a=d.get("alignment_variance",np.nan);a=float(np.asarray(a).ravel()[0])
            diagnostics.append(dict(n=n,law=law,rep=rep,fold=k,mse_m=d["mse_m"],
                mse_ell=float(np.asarray(d["mse_ell"]).ravel()[0]),
                alignment_product=d.get("alignment_product",np.sqrt(d["mse_m"]*a)),
                cov_b_a=float(np.asarray(d["cov_b_a"]).ravel()[0])))
    df=pd.DataFrame(rows);summary=[]
    for (n,law,method),g in df.groupby(["n","law","method"],sort=False):
        points=g.loc[g.valid_estimator,"estimate"].to_numpy(float)
        intervals=g[g.valid_interval];R=len(g);robust=law in ("t15","cauchy")
        cov=g.covered.sum()/R
        sd=np.subtract(*np.percentile(points,[75,25]))/1.348979500392 if len(points)>1 else np.nan
        medianse=intervals.se.median()
        summary.append(dict(n=n,law=law,method=method,R=R,bias=np.mean(points-1) if len(points) else np.nan,
            rmse=np.sqrt(np.mean((points-1)**2)) if len(points) else np.nan,
            median_absolute_error=np.median(abs(points-1)) if len(points) else np.nan,
            coverage=cov,coverage_mc_se=np.sqrt(cov*(1-cov)/R),coverage_valid_only=intervals.covered.mean(),
            median_interval_length=intervals.length.median(),median_se=medianse,iqr_empirical_sd=sd,
            calibration_ratio=medianse/sd if sd>0 else np.nan,valid_estimators=len(points),valid_intervals=len(intervals),
            failure_rate=1-len(intervals)/R,mean_seconds=g.seconds.mean(),median_seconds=g.seconds.median(),
            max_t_coverage=g.max_t_covered.eq(True).sum()/R if method=="Rank-Ad" else np.nan,
            max_t_median_length=g.max_t_length.median() if method=="Rank-Ad" else np.nan,
            infinite_variance_stress_test=robust))
    s=pd.DataFrame(summary)
    base=s[s.method=="OLS"].set_index(["n","law"]).rmse
    s["relative_rmse_ols"]=[row.rmse/base.loc[(row.n,row.law)] if base.loc[(row.n,row.law)]>0 else np.nan for row in s.itertuples()]
    paired=[]
    for (n,law),g in df.groupby(["n","law"]):
        wide=g.pivot(index="rep",columns="method",values="estimate")
        for method in METHODS:
            sub=wide[["OLS",method]].dropna() if method!="OLS" else wide[["OLS"]].dropna()
            if method=="OLS" or len(sub)<2:continue
            a=(sub[method].to_numpy()-1)**2;b=(sub.OLS.to_numpy()-1)**2
            r=rng(stage,int(n),law,0,"paired-"+method)
            ratios=[]
            for _ in range(1000):
                ix=r.integers(len(a),size=len(a));den=b[ix].mean()
                if den>0:ratios.append(np.sqrt(a[ix].mean()/den))
            if not ratios or b.mean()<=0:continue
            paired.append(dict(n=n,law=law,method=method,paired_count=len(sub),rmse_ratio=np.sqrt(a.mean()/b.mean()),
                bootstrap_lo=np.quantile(ratios,.025),bootstrap_hi=np.quantile(ratios,.975),
                ratio_mc_sd=np.std(ratios,ddof=1),infinite_variance_descriptive_only=law in ("t15","cauchy")))
    tables=output_base/"汇总表_CSV"/stage;tables.mkdir(parents=True,exist_ok=True)
    figdir=output_base/"图"/stage;figdir.mkdir(parents=True,exist_ok=True)
    outputs={"main_results":s,"replications":df,"paired_comparisons":pd.DataFrame(paired),
        "rank_ad_candidates":pd.DataFrame(candidates),"nuisance_diagnostics":pd.DataFrame(diagnostics),
        "failure_runtime":df[["n","law","rep","method","valid_estimator","valid_interval","failure","seconds"]]}
    selection=df[df.method=="Rank-Ad"].groupby(["n","law","selected_L"],dropna=False).size().reset_index(name="count")
    outputs["rank_ad_selection"]=selection
    theory=json.loads((ROOT/"config/theory.json").read_text())["laws"]
    theoretical=[]
    for law,inf in theory.items():
        if condition and law!=condition[1]:continue
        for method,var in [("Oracle",inf["efficient_variance"]),("Rank-W",inf["rank_w_variance"]),("Rank-N",inf["rank_n_variance"])]+[(f"Rank-Ad-L{L}",v) for L,v in enumerate(inf["rank_ad_variance"],1)]:
            theoretical.append(dict(law=law,method=method,variance=var,relative_to_efficient_bound=var/inf["efficient_variance"],fisher=inf["fisher"],quadrature_error=inf["quadrature_error"]))
    outputs["oracle_efficiency"]=pd.DataFrame(theoretical)
    for name,data in outputs.items():data.to_csv(tables/(name+".csv"),index=False,encoding="utf-8-sig")
    empirical=[]
    for row in s.itertuples():
        inf=theory[row.law]
        tv=inf["rank_w_variance"] if row.method=="Rank-W" else inf["rank_n_variance"] if row.method=="Rank-N" else inf["efficient_variance"] if row.method=="Oracle" else np.nan
        empirical.append(dict(n=row.n,law=row.law,method=row.method,R=row.R,relative_rmse_ols=row.relative_rmse_ols,
            empirical_n_mse_to_bound=row.n*row.rmse**2/inf["efficient_variance"],
            robust_n_variance_to_bound=row.n*row.iqr_empirical_sd**2/inf["efficient_variance"],
            fixed_score_theoretical_variance_to_bound=tv/inf["efficient_variance"],
            adaptive_fixed_L_theory_file="oracle_efficiency.csv",stress_test=row.infinite_variance_stress_test))
    pd.DataFrame(empirical).to_csv(tables/"empirical_theoretical_comparison.csv",index=False,encoding="utf-8-sig")
    plots=[]
    for law in s.law.unique():
        part=s[s.law==law]
        metrics=["median_absolute_error" if law in ("t15","cauchy") else "rmse","coverage","calibration_ratio","failure_rate","mean_seconds"]
        for metric in metrics:
            fig,ax=plt.subplots(figsize=(11,6))
            for method in METHODS:
                g=part[part.method==method].sort_values("n")
                ax.errorbar(g.n,g[metric],yerr=1.96*g.coverage_mc_se if metric=="coverage" else None,
                            marker="o",capsize=3,label=method)
            if metric=="coverage":ax.axhline(.95,color="black",ls="--");ax.set_ylim(-.05,1.05)
            if metric=="calibration_ratio":ax.axhline(1,color="black",ls="--")
            ax.set(xlabel="Sample size",ylabel=metric,title=f"{stage.upper()} | {law} | {metric}")
            ax.legend(ncol=3,fontsize=9);fig.tight_layout()
            name=f"{law}_{metric}.pdf";fig.savefig(figdir/name);plots.append(name);plt.close(fig)
        ad=part[part.method=="Rank-Ad"].sort_values("n")
        fig,ax=plt.subplots(figsize=(10,6))
        for name in ("coverage","max_t_coverage"):
            values=ad[name];err=np.sqrt(values*(1-values)/ad.R)
            ax.errorbar(ad.n,values,yerr=1.96*err,marker="o",capsize=4,label=name)
        ax.axhline(.95,color="black",ls="--");ax.set(xlabel="Sample size",ylabel="Coverage",ylim=(-.05,1.05),title=f"{stage.upper()} | {law} | Rank-Ad intervals");ax.legend();fig.tight_layout()
        name=f"{law}_rankad_intervals.pdf";fig.savefig(figdir/name);plots.append(name);plt.close(fig)
        sel=selection[selection.law==law]
        fig,ax=plt.subplots(figsize=(10,6))
        for n,g in sel.groupby("n"):ax.plot(g.selected_L,g["count"]/g["count"].sum(),marker="o",label=f"n={n}")
        ax.set(xlabel="Selected L",ylabel="Frequency (invalid selections retained in CSV)",xticks=range(1,7),title=f"{stage.upper()} | {law} | Rank-Ad selection");ax.legend();fig.tight_layout()
        name=f"{law}_selected_L.pdf";fig.savefig(figdir/name);plots.append(name);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6))
        for method in METHODS:
            g=part[part.method==method].sort_values("n")
            ax.plot(g.n,g.iqr_empirical_sd**2*g.n/theory[law]["efficient_variance"],marker="o",label=method)
        ax.axhline(1,color="black",ls="--");ax.set(xlabel="Sample size",ylabel="n * robust empirical SD squared / efficient bound",title=f"{stage.upper()} | {law} | Empirical efficiency diagnostic");ax.legend(ncol=3);fig.tight_layout()
        name=f"{law}_efficiency.pdf";fig.savefig(figdir/name);plots.append(name);plt.close(fig)
        diag=pd.DataFrame(diagnostics,columns=["n","law","rep","fold","mse_m","mse_ell","alignment_product","cov_b_a"]);dd=diag[diag.law==law]
        fig,ax=plt.subplots(figsize=(10,6))
        if not dd.empty:
            for n,g in dd.groupby("n"):ax.scatter([n], [np.median(g.alignment_product)*np.sqrt(n)],label=f"n={n}")
        ax.set(xlabel="Sample size",ylabel="Median sqrt(n) * alignment product",title=f"{stage.upper()} | {law} | Nuisance diagnostic");ax.legend();fig.tight_layout()
        name=f"{law}_alignment.pdf";fig.savefig(figdir/name);plots.append(name);plt.close(fig)
    for law in s.law.unique():
        g=s[(s.n==1000)&(s.law==law)].set_index("method").reindex(METHODS)
        if g["n"].isna().all():continue
        for metric in ("median_absolute_error" if law in ("t15","cauchy") else "relative_rmse_ols","coverage"):
            fig,ax=plt.subplots(figsize=(12,6))
            ax.bar(METHODS,g[metric],yerr=1.96*g.coverage_mc_se if metric=="coverage" else None,capsize=3)
            if metric=="coverage":ax.axhline(.95,color="black",ls="--");ax.set_ylim(0,1.05)
            ax.set(ylabel=metric,title=f"{stage.upper()} | n=1000 | {law} | {metric}")
            fig.tight_layout();name=f"n1000_{law}_{metric}.pdf";fig.savefig(figdir/name);plots.append(name);plt.close(fig)
    completion=[]
    for n,target in ((1000,2000),(500,1000),(2000,1000)):
        for law in LAWS:
            if condition and (n,law)!=condition:continue
            rr=[r for r in records if stage=="main" and r["n"]==n and r["law"]==law]
            completion.append(dict(n=n,law=law,target=target,completed=len(rr),reused=sum(r["source"]=="historical" for r in rr),
                new=sum(r["source"]=="new" for r in rr),status="COMPLETE" if len(rr)==target else "PENDING"))
    pd.DataFrame(completion).to_csv(tables/"experiment_completion_matrix.csv",index=False,encoding="utf-8-sig")
    reportdir.mkdir(parents=True,exist_ok=True)
    report=reportdir/(stage+"_results.md")
    invalid=sum(not row["valid_interval"] for row in rows)
    target=(2000 if condition[0]==1000 else 1000) if condition else 36000
    text=f"# Scenario A — {stage}\n\n{len(records)} replications; {len(rows)} method records; {invalid} invalid intervals. Design target: {target:,}.\n\nCoverage counts invalid intervals as not covered; conditional valid-only coverage is also reported. Point statistics use valid estimators with explicit denominators. Infinite-variance RMSE/bias are descriptive samples only, not finite population moment claims. Paired bootstrap uses matched valid pairs and reports pair counts. Method runtimes include their own computation plus the reused Wilcoxon preliminary cost where required. Common preliminary cost is attributed to each dependent method, so method times must not be summed as wall-clock time. Gaussian max-t simulation is included in Rank-Ad. Coverage error bars are plus/minus 1.96 binomial Monte Carlo SE.\n\n## Tables\n"
    text+="\n".join(f"- [{name}](../汇总表_CSV/{stage}/{name}.csv)" for name in outputs)
    text+="\n- [Empirical/theoretical comparison](../汇总表_CSV/"+stage+"/empirical_theoretical_comparison.csv)"
    text+="\n\n## Figure index\n"+"\n".join(f"- [{name}](../图/{stage}/{name})" for name in plots)
    report.write_text(text,encoding="utf-8")
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("smoke","pilot","main"));p.add_argument("--n",type=int);p.add_argument("--law")
    a=p.parse_args()
    if (a.n is None)!=(a.law is None):p.error("--n and --law must be supplied together")
    summarize(a.stage,condition=(a.n,a.law) if a.n is not None else None)


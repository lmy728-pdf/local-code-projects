"""Collect existing formal artifacts and make overview figures; never run experiments."""
from pathlib import Path
import json,csv,shutil,sys,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LogNorm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"代码"))
from methods import LAWS,METHODS
def main():
    manifest=json.loads((ROOT/"config/experiment_manifest.json").read_text(encoding="utf-8"))
    audit=json.loads((ROOT/"audit/formal_results_validation.json").read_text(encoding="utf-8"))
    summary=pd.read_csv(ROOT/"汇总表_CSV/main/main_results.csv")
    assert len(summary)==243 and audit["replications"]==36000
    assert summary.R.sum()==324000
    groups=[]
    for e in manifest["notebooks"]:
        group=(ROOT/e["notebook"]).parent.parent;name=group.name
        for key in ("汇总表_CSV","图"):
            source=group/key/"main";dest=ROOT/key/"分组结果"/name
            assert source.exists(),source
            dest.mkdir(parents=True,exist_ok=True)
            for p in source.iterdir():
                if p.is_file():shutil.copy2(p,dest/p.name)
        source=group/"结果_MD/main_results.md"
        text=source.read_text(encoding="utf-8")
        text=text.replace("../汇总表_CSV/main/",f"../../汇总表_CSV/分组结果/{name}/")
        text=text.replace("../图/main/",f"../../图/分组结果/{name}/")
        # Remove obsolete validation-stage boilerplate from previous group reports.
        text=text.replace(" This report does not certify statistical validity from a smoke test.","")
        text=text.replace("; with R=2 these are not meaningful accuracy estimates.",".")
        destination=ROOT/"结果_MD/分组结果"/(name+".md");destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(text,encoding="utf-8")
        # Also remove obsolete text from the original formal report, not its results.
        old=source.read_text(encoding="utf-8").replace(" This report does not certify statistical validity from a smoke test.","").replace("; with R=2 these are not meaningful accuracy estimates.",".")
        source.write_text(old,encoding="utf-8")
        groups.append(dict(group=e["id"],n=e["n"],law=e["law"],R=e["target"],
            report=str(destination.relative_to(ROOT)).replace("\\","/"),
            tables=f"汇总表_CSV/分组结果/{name}",figures=f"图/分组结果/{name}"))
    pd.DataFrame(groups).to_csv(ROOT/"汇总表_CSV/分组结果索引.csv",index=False,encoding="utf-8-sig")
    names=["Gaussian","Logistic","Student t(3)","Contaminated normal","Bimodal","Smoothed LN (0.75, 0.2)","Smoothed LN (1, 0.5)","Student t(1.5)*","Cauchy*"]
    overview=ROOT/"图/综合图";overview.mkdir(exist_ok=True)
    generated=[]
    for n in (500,1000,2000):
        part=summary[summary.n==n]
        R=int(part.R.iloc[0])
        for metric in ("coverage","relative_median_absolute_error"):
            if metric=="coverage":
                values=part.pivot(index="law",columns="method",values="coverage").reindex(index=LAWS,columns=METHODS)
                errors=part.pivot(index="law",columns="method",values="coverage_mc_se").reindex(index=LAWS,columns=METHODS)
                norm=TwoSlopeNorm(vmin=.5,vcenter=.95,vmax=1)
                title=f"Scenario A | Spline Ridge | n={n}, R={R}\n95% interval coverage: all nine laws and methods"
                caption="Cell: coverage % and +/- 1.96 Monte Carlo SE (percentage points). Target = 95% (color center).\nRank-Ad uses its Wald interval here; max-t is reported separately. Invalid intervals count as not covered."
                cmap="RdBu_r";label="Coverage (target 0.95)"
            else:
                values=part.pivot(index="law",columns="method",values="median_absolute_error").reindex(index=LAWS,columns=METHODS)
                values=values.div(values["OLS"],axis=0)
                norm=LogNorm(vmin=.02,vmax=3)
                title=f"Scenario A | Spline Ridge | n={n}, R={R}\nMedian absolute estimation error relative to OLS"
                caption="Cell: median(|estimate - 1|) / OLS median(|estimate - 1|). Smaller is better; OLS = 1.\n* Infinite-variance stress tests: median error is used for every law; no finite population MSE claim."
                cmap="viridis";label="Relative median absolute error (log color scale)"
            fig,ax=plt.subplots(figsize=(17,9.5))
            im=ax.imshow(values.to_numpy(),cmap=cmap,norm=norm,aspect="auto")
            ax.set_xticks(range(9),METHODS,fontsize=12)
            ax.set_yticks(range(9),names,fontsize=12)
            ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=10)
            for i in range(9):
                for j in range(9):
                    value=values.iloc[i,j]
                    display=f"{value*100:.1f}\n±{1.96*errors.iloc[i,j]*100:.1f}" if metric=="coverage" else f"{value:.3f}"
                    rgba=im.cmap(im.norm(value));luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                    ax.text(j,i,display,ha="center",va="center",fontsize=11,color="black" if luminance>.55 else "white")
            ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,9,1),minor=True)
            ax.grid(which="minor",color="white",linewidth=1.3);ax.tick_params(which="minor",length=0)
            fig.suptitle(title,fontsize=18,y=.985)
            fig.text(.5,.025,caption,ha="center",va="bottom",fontsize=11)
            fig.colorbar(im,ax=ax,fraction=.025,pad=.025,label=label)
            fig.subplots_adjust(left=.23,right=.94,top=.85,bottom=.13)
            file=f"A_Spline_n{n}_{metric}"
            fig.savefig(overview/(file+".pdf"),bbox_inches="tight")
            if n==1000:fig.savefig(overview/(file+".png"),dpi=120,bbox_inches="tight")
            generated.append(dict(n=n,metric=metric,file=f"图/综合图/{file}.pdf"))
            plt.close(fig)
    pd.DataFrame(generated).to_csv(ROOT/"汇总表_CSV/综合图索引.csv",index=False,encoding="utf-8-sig")
    # Table for the main sample size; retain all nine methods and laws.
    main=summary[summary.n==1000].copy()
    main["primary_metric"]=np.where(main.law.isin(["t15","cauchy"]),"median_absolute_error","rmse")
    main["primary_value"]=np.where(main.law.isin(["t15","cauchy"]),main.median_absolute_error,main.rmse)
    main.to_csv(ROOT/"汇总表_CSV/main/n1000_primary_results.csv",index=False,encoding="utf-8-sig")
    fail=pd.read_csv(ROOT/"汇总表_CSV/main/failure_details.csv")
    failgroups=fail.groupby(["n","law","method","failure"]).size().reset_index(name="count")
    lines=["# Scenario A · Spline Ridge 综合报告",
        "\n## 完成情况",
        "- 27/27组全部完成：n=500、1000、2000，各九分布。",
        "- n=1000每分布R=2000；n=500、2000每分布R=1000。",
        "- 共36,000个replications、324,000个方法记录；九方法都使用同一组对应数据和folds。",
        "- 第一阶段历史缓存复用10,000次、新生成26,000次；新版九方法二阶段全部计算。",
        "- 本次只整理已有正式结果，没有启动或补跑模拟。",
        "\n## n=1000 主结果",
        "有限方差七分布主看RMSE；t(1.5)、Cauchy主看中位绝对误差。下表误差一列为RMSE或MedAE，coverage为全部replications上的Wald覆盖率。",
        "\n| 分布 | 指标 | OLS误差 | Rank-W误差 | Rank-N误差 | Rank-Ad误差 | Oracle误差 | Ad Wald | Ad max-t |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for law in LAWS:
        part=main[main.law==law].set_index("method")
        metric="median_absolute_error" if law in ("t15","cauchy") else "rmse"
        vals=[part.loc[x,metric] for x in ("OLS","Rank-W","Rank-N","Rank-Ad","Oracle")]
        lines.append(f'| {law} | {"MedAE" if metric.startswith("median") else "RMSE"} | '+" | ".join(f"{v:.4f}" for v in vals)+f' | {part.loc["Rank-Ad","coverage"]:.2%} | {part.loc["Rank-Ad","max_t_coverage"]:.2%} |')
    lines+=["\n上表是Rank方法与基准的紧凑摘要；完整九方法（包括Huber、LAD、CQR、Bickel）都在综合图及CSV中。",
        "\n## 结果解读",
        "- 精度与推断需要分开评价。n=1000时，Rank-Ad在双峰分布的RMSE为0.0192，OLS为0.0393；但不能据此认定Rank-Ad在所有条件都最优，Bickel在部分偏态/污染条件也有更低RMSE。",
        "- Rank-Ad的max-t区间提高了覆盖率，但没有在所有条件达到95%。例如n=1000 Gaussian下，Wald为90.80%，max-t为92.45%；双峰下分别为91.40%、95.40%。这些结果全部保留，不调整算法来改善排名或覆盖率。",
        "- Cauchy下OLS的样本RMSE受极端值影响很大。主要采用MedAE和IQR稳健尺度比较；平方损失第一阶段仍按原设计执行，不将程序成功运行解释为满足无限方差下的理论学习率。",
        "- 配对RMSE比值及其bootstrap区间使用相同replication的有效配对；应结合不确定性判断接近的差异，不能只按点估计排优劣。",
        "\n## 失败与统计口径",
        f"- 无效区间共{len(fail)}个，均为Rank-Ad的nonpositive_corrected_I1；按原规则保留，未重抽样或更改阈值。",
        "- coverage分母包含无效区间（计为未覆盖）；另给有效区间条件下的coverage。点估计指标使用有效点估计，并明确计数。",
        "- 覆盖率误差棒为±1.96×binomial Monte Carlo SE；综合图单元格同时显示覆盖率和该不确定性。",
        "\n| n | 分布 | 方法 | 失败数 | 原因 |","|---:|---|---|---:|---|"]
    for x in failgroups.itertuples(index=False):lines.append(f"| {x.n} | {x.law} | {x.method} | {x.count} | {x.failure} |")
    lines+=["\n## 文件入口",
        "- [综合主表](../汇总表_CSV/main/main_results.csv)",
        "- [n=1000完整九方法主表](../汇总表_CSV/main/n1000_primary_results.csv)",
        "- [逐次方法结果](../汇总表_CSV/main/replications.csv)",
        "- [配对比较与不确定性](../汇总表_CSV/main/paired_comparisons.csv)",
        "- [各组完成与复用数量](../汇总表_CSV/main/experiment_completion_matrix.csv)",
        "- [逐次失败明细](../汇总表_CSV/main/failure_details.csv)",
        "- [全部综合趋势图索引](main_results.md)",
        "- [图表总索引](图表总索引.md)",
        "- [正式记录核对](final_validation_report.md)",
        "\n## 综合图"]
    for fig in generated:lines.append(f'- [n={fig["n"]} · {fig["metric"]}](../{fig["file"]})')
    lines+=["\n## 各组报告"]
    for group in groups:lines.append(f'- [{group["group"]} · n={group["n"]} · {group["law"]}](分组结果/{Path(group["report"]).name})')
    (ROOT/"结果_MD/A_Spline_综合报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    # Make a single, navigable vector-figure index.
    indexes=[]
    for p in sorted((ROOT/"图").rglob("*.pdf")):
        if "smoke" in p.parts:continue
        indexes.append(dict(file=str(p.relative_to(ROOT)).replace("\\","/"),category="综合图" if p.parent==overview else "分组图" if "分组结果" in p.parts else "跨样本量趋势图"))
    pd.DataFrame(indexes).to_csv(ROOT/"汇总表_CSV/全部图表索引.csv",index=False,encoding="utf-8-sig")
    text="# A — Spline Ridge 图表总索引\n\n正式图片均为PDF矢量图；PNG仅作预览。\n"
    for category in ("综合图","跨样本量趋势图","分组图"):
        text+="\n## "+category+"\n\n"
        text+="\n".join(f'- [{Path(x["file"]).stem}](../{x["file"]})' for x in indexes if x["category"]==category)+"\n"
    (ROOT/"结果_MD/图表总索引.md").write_text(text,encoding="utf-8")
    (ROOT/"结果_MD/final_validation_report.md").write_text(f"""# 正式结果核对报告

本报告核对现有正式结果，未启动新的模拟。

- 27组全部达到目标，共36,000次、324,000个方法记录。
- 各组(n,law,rep)与记录内标识一致；rep ID完整覆盖预定范围，无重复。
- 36,000个数据seed key唯一；每次恰好包含全部九方法。
- 所有正式数据库科学代码签名与当前methods、legacy_core和论文副本一致。
- 输入来源计数：历史10,000次、新生成26,000次。
- 无效区间{len(fail)}个，均按既定Rank-Ad修正信息正性规则记录；其他方法记录不删除。
- 首次实现检查通过的签名保留在config/code_validation.json，确保清理测试数据后正式读取/续跑入口仍正常。
- smoke数据、图、报告、日志按用户要求清除；现有结果目录只汇总正式数据。

[核对明细JSON](../audit/formal_results_validation.json) · [各组核对CSV](../汇总表_CSV/main/condition_validation.csv) · [综合报告](A_Spline_综合报告.md)
""",encoding="utf-8")
    (ROOT/"README.md").write_text("""# 毕设正式论文 numerical simulation

**Scenario A — Spline Ridge：27组全部完成，共36,000次正式模拟。**
RF与GBDT尚未纳入本次结果；结果均为scalar。

## 看结果

- [综合报告](结果_MD/A_Spline_综合报告.md)
- [正式主表CSV](汇总表_CSV/main/main_results.csv)
- [综合图索引](汇总表_CSV/综合图索引.csv)
- [全部PDF图表索引](结果_MD/图表总索引.md)
- [分组结果索引](汇总表_CSV/分组结果索引.csv)
- [完成情况](汇总表_CSV/main/experiment_completion_matrix.csv)
- [正式结果核对](结果_MD/final_validation_report.md)

## 文件位置

| 内容 | 统一目录 |
|---|---|
| 综合与逐次CSV | 汇总表_CSV/main/ |
| 各组CSV | 汇总表_CSV/分组结果/ |
| 全方法全分布综合图 | 图/综合图/ |
| 跨n趋势与诊断图 | 图/main/ |
| 各组PDF | 图/分组结果/ |
| 综合报告与图索引 | 结果_MD/ |
| 各组报告 | 结果_MD/分组结果/ |

分组实验/ 中保留各组notebook、原始SQLite记录和原有正式输出；根目录上述位置是统一阅览入口。
PDF是正式矢量图，PNG只是预览。测试输出已清理。

## 代码与复现

- [27个独立Notebook入口](开始运行_实验索引.md)
- [仅汇总已有结果的Notebook](代码/汇总已完成结果.ipynb)
- [数值实现约定](config/implementation_conventions.md)
- [实验配置](config/experiment_manifest.json)
- [当前环境版本](config/environment/versions.json)
- [锁定依赖](代码/requirements.txt)

默认12个单线程worker，每次只运行一组；后续模拟由用户手动启动。
完整复现需保留本文件夹中的公共代码、sources、config及cache/reused_inputs.sqlite。
原始结果位于分组实验/各组/代码目录/_work/replications/records.sqlite。
""",encoding="utf-8")
    p=ROOT/"开始运行_实验索引.md"
    text=p.read_text(encoding="utf-8").replace("此前完整九分布smoke验证","正式结果核对")
    p.write_text(text,encoding="utf-8")
    # Update implementation documentation's removed diagnostic links.
    p=ROOT/"config/implementation_conventions.md"
    s=p.read_text(encoding="utf-8").replace("smoke/density_validation.json","原实现精度检查记录（测试输出已清理）")
    p.write_text(s,encoding="utf-8")
    print(json.dumps(dict(groups=len(groups),replications=audit["replications"],method_records=audit["method_records"],invalid_intervals=len(fail),overview_pdfs=len(generated),indexed_pdfs=len(indexes)),ensure_ascii=False))
if __name__=="__main__":main()


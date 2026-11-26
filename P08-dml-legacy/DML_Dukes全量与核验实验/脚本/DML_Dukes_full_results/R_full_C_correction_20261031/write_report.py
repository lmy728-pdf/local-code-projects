"""Render a report from complete validated result tables; no computation of methods."""
import hashlib,json,pathlib
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
NAMES={'gaussian':'Gaussian','t2.1':'Student-t(2.1)','lognormal2':'Centered lognormal(2)'}
def percent(x):return f'{100*x:.1f}%'
def pp(x):return f'{100*x:+.2f}'
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def main():
    summary=pd.read_csv(ROOT/'summary.csv');pair=pd.read_csv(ROOT/'paired_comparison.csv')
    data=pd.read_csv(ROOT/'r_python_comparison.csv');verification=json.loads((ROOT/'verification.json').read_text())
    execution=json.loads((ROOT/'execution.json').read_text());runtime=json.loads((ROOT/'r_runtime.json').read_text())
    assert execution['result_files']==3000 and not any(execution['exit_codes'])
    assert verification['attempted']==3000
    indexed=summary.set_index(['scenario','method']);pindexed=pair.set_index('scenario')
    coverage_rows=[];stat_rows=[];change_rows=[];delta_rows=[]
    for scenario,name in NAMES.items():
        r=indexed.loc[(scenario,'C_R')];p=pindexed.loc[scenario]
        coverage_rows.append([name]+[percent(indexed.loc[(scenario,m),'coverage_at_truth']) for m in ['A','B','C_Python','C_R']]+
                             [f"{percent(r.coverage_wilson95_lower)}–{percent(r.coverage_wilson95_upper)}",int(r.valid)])
        for method in ['A','B','C_Python','C_R']:
            row=indexed.loc[(scenario,method)]
            stat_rows.append([name,method,f'{row.mean_T:.6f}',f'{row.sd_T:.6f}',percent(row.p_T_lt_minus1_96),
                              percent(row.p_T_gt_1_96),percent(row.p_abs_T_le_1_96),int(row.valid)])
        change_rows.append([name,pp(p.coverage_difference_R_minus_Python),f'{100*p.coverage_difference_mcse:.3f}',
                            f'[{pp(p.coverage_difference_normal95_lower)}, {pp(p.coverage_difference_normal95_upper)}]',
                            int(p.Python_reject_R_accept),int(p.Python_accept_R_reject),int(p.decision_disagreements)])
        delta_rows.append([name,f'{p.mean_T_difference:.6f}',f'{p.rmse_T_difference:.6f}',f'{p.max_abs_T_difference:.6f}',
                           f'{p.abs_T_difference_q50:.6f}',f'{p.abs_T_difference_q95:.6f}',int(p.abs_T_difference_gt_05)])
    cover=table(['误差场景','A 原 Wald','B 原 score','C Python','C 作者 R','R C 的95% MC区间','R有效数'],coverage_rows)
    stats=table(['场景','方法','Mean(T)','SD(T)','P(T<−1.96)','P(T>1.96)','P(abs(T)≤1.96)','有效数'],stat_rows)
    changes=table(['场景','Δcoverage（百分点）','配对 MCSE（百分点）','配对95%近似区间','Py拒绝→R接受','Py接受→R拒绝','总翻转'],change_rows)
    deltas=table(['场景','Mean(ΔT)','RMSE(ΔT)','最大abs(ΔT)','abs(ΔT)中位数','abs(ΔT)95%分位','abs(ΔT)>0.5次数'],delta_rows)
    cr=summary[summary.method=='C_R']
    remains_under=(cr.coverage_wilson95_upper<.95).all()
    conclusion=('作者 R 实现仍在三个场景下严重欠覆盖；带宽算法差异没有解释掉原实验的主要欠覆盖现象。'
                if remains_under else '作者 R 实现的覆盖率应按下表逐场景判断，不能将带宽差异概括为完全没有影响。')
    worst=data.loc[data.T_difference_R_minus_Python.abs().idxmax()]
    total_flips=int(pair.decision_disagreements.sum())
    text=f'''# 全部已保存数据的作者 R C correction 报告

日期：2026-10-31。数据范围：原 DGP2、n=1000、p=5，每种误差1000组，共3000组。θ₀=1。

**{conclusion}** 本次直接调用固定版本作者 `rdml_cross.test()`，没有重跑 DGP、HGBR、A/B，也没有扩大样本量或调整参数。

## 1. 覆盖率结果

{cover}

A/B/C Python 均取自原保存结果。R C 取自本次完整重算。A 使用原 Wald 区间；B/C 的“coverage”指在真值 θ₀=1 下的检验接受率，也就是反演检验置信集在真值处的覆盖率；本次没有计算 C 的区间端点或长度。R p≥0.05 记为接受。表中95% MC区间是1000次模拟接受比例的 Wilson 区间，不是 θ 的置信区间。

![四种实现的覆盖率](coverage_comparison.png)

## 2. Python C 与作者 R C 的配对差异

所有差异均按同一数据、同一初始预测和同一 fold 配对，Δ=R−Python。

{changes}

覆盖率差的 MCSE 使用逐 replication 的接受指示之差计算，保留配对关系；不是把两个覆盖率当作独立估计。总计 **{total_flips}/3000** 组改变5%水平检验结论。净覆盖率变化可能较小，但这不表示每组统计量都一致。

{deltas}

最大差异发生在 **{NAMES[worst.scenario]}，replication {int(worst.replication)}**：T_Python={worst.T_Python:.6f}，T_R={worst.T_R:.6f}，ΔT={worst.T_difference_R_minus_Python:.6f}。所有翻转记录见 [decision_flips.csv](decision_flips.csv)，最大差异的30组见 [largest_T_differences.csv](largest_T_differences.csv)。replication ID 延续原0-based编号。

## 3. 标准化统计量分布

{stats}

SD(T)使用样本标准差（ddof=1）；A的 T=(θ̂−1)/SE(θ̂)，B/C使用各自标准化 score。表中尾部按字面临界值±1.96计算，coverage主表按各方法原定义计算；两者临界值差异涉及的记录数为 **{int(summary.critical_value_difference_count.sum())}**。完整均值MCSE、coverage MCSE和异常数见 [summary.csv](summary.csv)。

![两种 C 的 T 分布及标准正态参考](T_distributions.png)

这组结果应同时看均值与标准差：T即使具有接近1的标准差，只要均值明显偏离0，双侧5%检验仍会严重过度拒绝。当前数据的偏移方向和左右尾比例已完整列出，不只报告总体覆盖率。

## 4. 执行与一致性核验

- 直接使用[固定作者代码](reference/dri_functions.r)的 `rdml_cross.test()`（第49–77行），commit `e7dfb038117aea3d95ca1371e2e85557ff9f25d8`；源码 SHA256 与前次30组验证完全相同。本次没有用自编诊断包装器替代作者函数。
- 输入重建：D=d_res+g_hat，Y−D=b+m_hat。B/C共享的初始预测原样导出；3000组残差回算均逐项相等。每组原观测顺序保持，fold 0–4映射为1–5，每折200条。没有R nuisance训练。
- R辅助随机种子规则为 `20261030+100000×场景序号+replication`，与前次30组完全一致；每次调用前单独设种子，因此结果不依赖工作进程分配或执行顺序。原DGP、HGBR及交叉拟合种子没有修改。
- 作者返回 `est=√N×mean(score)`、`se=sd(score)/√N`，本次使用 **T=est/(se×√N)**；p值保留作者的χ²₁双侧检验结果，并独立核对其与2Φ(−abs(T))一致。
- 计划3000组，完成{verification['attempted']}组，有效{verification['valid']}组，失败{verification['failed']}组，包含R warnings的记录{verification['warning_records']}组。未对失败进行填补或按结果换种子。
- 前次30组全部重现，作者原始返回值最大绝对差 **{verification['prior_30_max_author_difference']:.3g}**，T最大绝对差 **{verification['prior_30_max_T_difference']:.3g}**。见 [prior_30_reproduction.csv](prior_30_reproduction.csv)。
- 原notebook、原数据数组、原任务JSON及主要汇总共 **{verification['source_files_unchanged']}** 个源文件哈希均未改变。见 [input_manifest.json](input_manifest.json)、[verification.json](verification.json)。
- 使用{runtime['R_version']}，np **0.80-1**；保留作者调用下的默认核配置和`np.largeh=TRUE`，没有关闭近似、增加multistart或改变带宽搜索以改善结果。当前环境仍不等于已经复原作者2024年的历史依赖环境。
- 本机12个单线程R进程完成，批处理耗时 **{execution['elapsed_seconds']/60:.2f}分钟**，不含输入导出、汇总和绘图。只是correction阶段的耗时，不能外推成完整HGBR模拟的耗时。

每组作者原始返回值、T、p、score均值/标准差、种子、输入哈希和异常保存在 `results/*.json`。本次没有修改作者函数以截取内部数组，因此没有新增全部3000组的R逐观测score；逐replication标准化统计量已完整保存。前次30组逐观测与逐折诊断继续保留在原验证目录。

## 5. 对毕业论文选题的含义

这次回答的是：**在既定 DGP2、HGBR、五折交叉拟合、n=1000 和这三类误差下，换成作者 R correction 后推断表现怎样。** 结果已经排除了“只因 Python 自定义带宽实现才造成全部主要欠覆盖”的解释（判断依据是完整R结果，而非之前偏向困难样本的30组）。前次同带宽核对也支持score、更新和方差公式一致。

可以将当前配置下的欠覆盖写成可复核的实验发现；**不能由这一配置推导所有重尾条件下 Dukes 方法必然失效，或宣称否定了其渐近理论**。尤其需要同时报告Gaussian结果，避免将已有基准场景中的偏移全部归因于重尾。

本轮已完成所要求的全部保存数据C重算。无需为了本次R/Python实现核对再重复完整DGP/HGBR实验；后续是否研究其他稳健方法，应以这里明确限定范围的结果为依据，不通过事后调参追求更好看的coverage。

## 6. 可追溯交付

- [逐replication R/Python比较](r_python_comparison.csv)：3000行，含两版T/p、score摘要、检验翻转和种子。
- [四方法标准化统计量](standardized_statistics.csv)：12000行；A/B/原C只读复用。
- [场景×方法统计汇总](summary.csv)、[配对差异及MCSE](paired_comparison.csv)。
- [所有输入ID与哈希](replications.csv)、[失败清单](failures.csv)、[警告清单](warnings.csv)。
- [重跑说明](README.md)、[数据导出脚本](prepare_inputs.py)、[作者函数批处理入口](run_author_worker.R)、[并行启动器](launch_workers.py)、[统计汇总脚本](summarize_results.py)。

论文来源：[Dukes, Vansteelandt & Whitney, JMLR](https://www.jmlr.org/papers/v25/22-1233.html)；作者源文件：[固定GitHub版本](https://github.com/ordukes/DRInference/blob/e7dfb038117aea3d95ca1371e2e85557ff9f25d8/dri_functions.r)。
'''
    (ROOT/'full_C_report.md').write_text(text,encoding='utf-8')
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json' and '__pycache__' not in p.parts]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'delivery_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('Report and delivery hashes saved.')
if __name__=='__main__':main()

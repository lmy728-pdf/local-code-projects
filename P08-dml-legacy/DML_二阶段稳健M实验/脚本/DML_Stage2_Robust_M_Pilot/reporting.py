"""One cumulative summary table and one combined scientific figure."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

SCENES = ['gaussian', 't2.1', 'lognormal2']
LABELS = ['Gaussian', 'Student-t(2.1)', 'Centered lognormal(2)']
METHODS = ['A', 'B', 'C', 'D']
COLORS = ['#667085', '#007F86', '#DE7835', '#7253B5']
NAMES = ['A: Standard DML', 'B: Symmetric Huber', 'C: Asymmetric + intercept', 'D: Adaptive + intercept']


def summary_table(raw, planned=200):
    rows = []
    for scene in SCENES:
        allscene = raw[raw.scenario==scene]
        base = allscene[(allscene.method=='A') & (allscene.status=='ok')].set_index('rep_id')
        for method in METHODS:
            g = allscene[allscene.method==method]
            v = g[g.status=='ok']
            n = len(v)
            if n == 0:
                rows.append(dict(scenario=scene, method=method, planned=planned, completed=len(g), valid=0,
                                 failed=len(g), nonfinite_statistics=len(g)))
                continue
            coverage = v.covered.astype(float).mean()
            pair = v.set_index('rep_id').join(base[['covered']], rsuffix='_A', how='inner')
            diff = pair.covered.astype(float)-pair.covered_A.astype(float)
            row = dict(scenario=scene, method=method, planned=planned, completed=len(g), valid=n,
                       failed=int((g.status!='ok').sum()), nonfinite_statistics=int((~np.isfinite(g[['theta_hat','se','T']])).any(axis=1).sum()),
                       convergence_rate=float(g.converged.fillna(False).astype(bool).mean()),
                       optimizer_flag_false=int((~g.optimizer_success.fillna(False).astype(bool)).sum()),
                       warnings_count=int((g.warnings.fillna('[]')!='[]').sum()),
                       bias=float((v.theta_hat-1.).mean()), empirical_sd=v.theta_hat.std(ddof=1),
                       mean_se=v.se.mean(), rmse_theta=float(np.sqrt(np.mean((v.theta_hat-1.)**2))),
                       coverage=coverage, coverage_mcse=np.sqrt(coverage*(1-coverage)/n),
                       mean_T=v['T'].mean(), sd_T=v['T'].std(ddof=1),
                       reject_left=float((v['T'] < -1.96).mean()), reject_right=float((v['T'] > 1.96).mean()),
                       mean_ci_length=v.ci_length.mean(), mean_scale=v.scale.mean(),
                       mean_tau_minus=v.tau_minus.mean(), min_tau_minus=v.tau_minus.min(), max_tau_minus=v.tau_minus.max(),
                       mean_tau_plus=v.tau_plus.mean(), min_tau_plus=v.tau_plus.min(), max_tau_plus=v.tau_plus.max(),
                       mean_clipping_rate=v.clipping_rate.mean(), max_clipping_rate=v.clipping_rate.max(),
                       mean_clip_left=v.clip_left.mean(), mean_clip_right=v.clip_right.mean(),
                       mean_intercept=v.intercept.mean() if method in ['C','D'] else np.nan,
                       sd_intercept=v.intercept.std(ddof=1) if method in ['C','D'] else np.nan,
                       max_equation_error=v.score_norm.max(), max_bread_condition=v.bread_condition.max(),
                       near_knot_count=int(v.near_knots.fillna(0).sum()),
                       paired_valid=len(pair), paired_coverage_gain_vs_A=diff.mean(),
                       paired_gain_mcse=diff.std(ddof=1)/np.sqrt(len(pair)) if len(pair)>1 else np.nan,
                       se_to_empirical_sd=v.se.mean()/v.theta_hat.std(ddof=1))
            rows.append(row)
    table = pd.DataFrame(rows)
    for index, row in table.iterrows():
        if row.scenario=='gaussian':
            table.loc[index, 'delta_tail'] = 0.
            table.loc[index, 'delta_tail_mcse'] = 0.
            continue
        g = table[(table.scenario=='gaussian') & (table.method==row.method)].iloc[0]
        table.loc[index, 'delta_tail'] = g.get('coverage', np.nan)-row.get('coverage', np.nan)
        # Gaussian uses different seed block. t vs lognormal not treated independent.
        table.loc[index, 'delta_tail_mcse'] = np.sqrt(g.get('coverage_mcse', np.nan)**2+row.get('coverage_mcse', np.nan)**2)
    return table


def make_figure(raw, table, path):
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    width=.18
    x=np.arange(3)
    for j, method in enumerate(METHODS):
        g=table[table.method==method].set_index('scenario').loc[SCENES]
        pos=x+(j-1.5)*width
        axes[0,0].bar(pos, g.coverage, width, color=COLORS[j], label=NAMES[j])
        axes[0,0].errorbar(pos, g.coverage, yerr=1.96*g.coverage_mcse, fmt='none', ecolor='#333333', capsize=2, lw=.8)
        axes[0,1].bar(np.arange(2)+(j-1.5)*width, g.delta_tail.iloc[1:], width, color=COLORS[j])
        axes[0,1].errorbar(np.arange(2)+(j-1.5)*width,g.delta_tail.iloc[1:],yerr=1.96*g.delta_tail_mcse.iloc[1:],fmt='none',ecolor='#333333',capsize=2,lw=.8)
        axes[0,2].bar(pos,g.bias,width,color=COLORS[j])
        axes[1,0].plot(x,g.empirical_sd,'o-',color=COLORS[j],label=method+' empirical SD')
        axes[1,0].plot(x,g.mean_se,'x--',color=COLORS[j],label=method+' mean SE')
        axes[1,1].bar(pos,g.mean_ci_length,width,color=COLORS[j])
        axes[1,2].bar(pos,g.mean_clipping_rate,width,color=COLORS[j])
    axes[0,0].axhline(.95,color='black',ls='--',lw=1)
    axes[0,0].set_ylim(0,1.05)
    axes[0,0].set_title('95% coverage (bars: +/-1.96 MCSE)')
    axes[0,1].set_title('Coverage deterioration: Gaussian - tail')
    axes[0,1].set_xticks([0,1],['Student-t','Lognormal'])
    axes[0,1].axhline(0,color='black',lw=.7)
    axes[0,2].set_title('Bias of theta (target = 1)')
    axes[0,2].axhline(0,color='black',lw=.7)
    axes[1,0].set_title('Empirical SD and mean estimated SE')
    axes[1,0].legend(fontsize=7,ncol=2)
    axes[1,1].set_title('Mean confidence interval length')
    axes[1,2].set_title('Mean clipping rate')
    for ax in [axes[0,0],axes[0,2],*axes[1]]:
        ax.set_xticks(x,['Gaussian','Student-t','Lognormal'])
    for j,scene in enumerate(SCENES):
        ax=axes[2,j]
        v=raw[(raw.scenario==scene)&(raw.status=='ok')]
        low=min(-4.,float(v['T'].min())-.3); high=max(4.,float(v['T'].max())+.3)
        bins=np.linspace(low,high,27)
        for k,method in enumerate(METHODS):
            ax.hist(v.loc[v.method==method,'T'],bins=bins,density=True,histtype='step',lw=1.6,color=COLORS[k],label=method)
        grid=np.linspace(low,high,400)
        ax.plot(grid,norm.pdf(grid),'k--',lw=1,label='N(0,1)')
        ax.axvline(-1.96,color='gray',ls=':',lw=.8);ax.axvline(1.96,color='gray',ls=':',lw=.8)
        ax.set_title('Standardized T: '+LABELS[j]);ax.set_xlabel('T');ax.set_ylabel('Density')
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.946),ncol=4,frameon=False,fontsize=10)
    count=int(table.completed.min())
    fig.suptitle(f'Stage-2 robust M pilot | DGP2, n=1000, p=5 | R={count} per scenario\nSame saved HGBR predictions for all methods; fixed settings',fontsize=15,y=.994)
    fig.tight_layout(rect=(0,0,1,.91),h_pad=2.1,w_pad=1.5)
    fig.savefig(path,dpi=170,bbox_inches='tight')
    plt.close(fig)


def markdown_table(frame, cols):
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v,(float,np.floating)): return f'{v:.5f}'
        return str(v)
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    lines += ['| '+' | '.join(fmt(row[c]) for c in cols)+' |' for _,row in frame.iterrows()]
    return '\n'.join(lines)


def write_report(output, raw, table, checks, oracle, config, integrity):
    text=['# 第二阶段 Robust M / Adaptive Huber-DML：R=200 pilot 审计报告','',
          '本轮结论：B 的对称无截距 Huber 明显降低了这组 pilot 的重尾斜率偏差，并提高覆盖率，但所有方法仍远低于 95%。预设 C 未显示跨场景保护优势；D 的阈值很大、实际截断很少，重尾覆盖率反而下降。因此不能宣布通用 Adaptive Huber 已解决本 DGP 的推断恶化。以下判断仅针对本轮固定配置。',
          '本轮只运行预先固定的四种二阶段方法。每场景使用原实验 replication 0–199；不挑选结果、不调整阈值、不重新训练 HGBR。',
          '桌面 notebook 是唯一新 notebook；结果目录内只有一份汇总表和一张九面板图。逐次 CSV、Oracle CSV 和逐观测 NPZ 为审计底稿。','',
          '## 实验与实现','',
          'DGP2 nonlinear，n=1000，p=5，真实 θ₀=1，5 折。复用旧 g_hat（treatment）、ell_hat（outcome）、d_res、y_res。注意旧 m_hat 是变换结果的预测，不用于本轮。',
          '种子：20260826 + 100000×1 + 1000×error_id + rep；Gaussian error_id=7，t/lognormal=0；交叉拟合再加 500000。t 和 lognormal 同 ID 共享 X 与 treatment，不把这两类场景的差异视为独立样本。',
          '所有输入 NPZ 与旧 task 的 SHA256 核对；原 DGP 只用于重建并检查残差和 fold，完全未调用模型 fit。',
          'HGBR：squared_error，learning_rate=.05，max_iter=100，max_leaf_nodes=15，min_samples_leaf=20，l2_regularization=1；其余原配置保存在 run_config.json。','',
          '- A：原始无截距残差 OLS 与原函数 fit_mean_dml，IF 方差 ddof=1。',
          '- B：无截距，τ=1.345s，s=1.4826×初步无截距 OLS 残差 MAD。',
          '- C：联合截距和斜率，τ₋=2s、τ₊=1.5s，固定探索性不对称截断。',
          '- D：联合截距和斜率，τ=s√(n/log n)，自然对数；无分布名称或真实参数输入。',
          '- B/C/D：凸分段二次损失；BFGS 加保持损失不增的 Newton 精修。以最终所有估计方程的最大绝对误差≤1e-9判定收敛；单纯优化器 precision-loss 标志不等于失败，单独记录。',
          '- A=mean[ψ′zzᵀ]，B=mean[(zψ−mean(zψ))(…)ᵀ]，V=A⁻¹BA⁻ᵀ/n。B 是一维对应式；C/D 从完整 2×2 方差提取斜率 SE。A 最小特征值≤1e-12 或条件数>1e10 时失败，不用伪逆。','',
          '新 sandwich 使用分母 n，原 IF 使用 n−1。因此极大阈值无截距 Huber 的斜率等于 OLS，SE 等于原 SE×√((n−1)/n)，不是实现错误。阈值估计被视为固定，未计入其全部不确定性。','',
          '## 核验及异常','',
          f'有限差分梯度最大差 {checks["max_gradient_error"]:.3g}，Hessian 最大差 {checks["max_hessian_error"]:.3g}；独立 IF 构造核对 sandwich、损失连续性、固定数据回归测试、无效输入拒绝均通过。',
          f'旧 A θ 最大差 {raw.legacy_theta_difference.abs().max():.3g}，SE 最大差 {raw.legacy_se_difference.abs().max():.3g}。',
          f'输入文件审计：{integrity["checked_files"]} 个文件前后哈希一致={integrity["unchanged"]}；残差重建最大差 {integrity["max_residual_difference"]:.3g}。',
          f'总记录 {len(raw)}，有效 {(raw.status=="ok").sum()}，失败 {(raw.status!="ok").sum()}；优化器原始 success=False 有 {(~raw.optimizer_success.fillna(False).astype(bool)).sum()} 条，其最终方程及方差均另行核验。',
          f'最大估计方程误差 {raw.score_norm.max():.3g}，最大 bread 条件数 {raw.bread_condition.max():.4g}；接近截断结点的观测数 {int(raw.near_knots.fillna(0).sum())}。',
          '失败的覆盖率保存为空，不记为 0；汇总仅用有效结果，并同时披露计划、完成、有效和失败数量。','',
          '## 主要数值','',
          markdown_table(table,['scenario','method','valid','bias','empirical_sd','mean_se','coverage','coverage_mcse','mean_T','sd_T','reject_left','reject_right']),
          '',markdown_table(table,['scenario','method','rmse_theta','mean_ci_length','mean_tau_minus','mean_tau_plus','mean_clipping_rate','mean_clip_left','mean_clip_right','mean_intercept']),
          '',markdown_table(table,['scenario','method','delta_tail','delta_tail_mcse','paired_coverage_gain_vs_A','paired_gain_mcse']),
          '', '配对增益=mean(I_method−I_A)，其 MCSE=sd(I_method−I_A,ddof=1)/√R，有效配对数在 summary.csv。Δ_tail=Gaussian coverage−HeavyTail coverage。Gaussian 与各重尾场景使用不同种子块，Δ 的 MCSE 按两组方差相加；不据此推断 t 与 lognormal 之间独立。','',
          '## Oracle 与目标参数','',
          'Oracle 仅检查已有前 20 个 replication，不进入正式估计或阈值选择。以下 4×MCSE 范围检查是预先固定的数值识别诊断，不是统计有效性证明。',
          markdown_table(pd.DataFrame(oracle),['scenario','method','mean_theta','bias_mcse','mean_intercept','mean_se','empirical_sd','max_equation_error']),
          '', '在本 DGP 的 Oracle 情形，d̃=v，ỹ=θ₀v+ε，v 与 ε 独立且 E[v]=0。对于带截距估计，令 a 解 Eψ(ε−a)=0，则斜率方程 E[vψ(ε−a)]=0；有正的局部 bread 时 θ₀是唯一局部斜率目标。t 对称且 B 的 score 对称时 a=0。',
          '对于 lognormal，利用截断 lognormal 的精确一阶矩独立计算人口层面的 Huber 位置根。oracle_diagnostics.csv 保存每次拟合的该根、Eψ(ε) 与 Eψ(ε−a)。截距吸收的是 Huber 位置偏移，通常不等于误差的均值零。',
          '使用估计 nuisance 后，残差中会含与 X、treatment 相关的学习误差；上述 Oracle 论证不能直接推广。截距不能自动消除 nuisance 偏差，也未证明该稳健 score 对 nuisance Neyman orthogonal。','',
          '## 比较与解释','']
    for scene, method, label in [('t2.1','B','A vs B：对称重尾'),('lognormal2','C','A vs C：非对称重尾')]:
        r=table[(table.scenario==scene)&(table.method==method)].iloc[0]
        a=table[(table.scenario==scene)&(table.method=='A')].iloc[0]
        text.append(f'{label}：覆盖率 {a.coverage:.1%} → {r.coverage:.1%}，配对增益 {r.paired_coverage_gain_vs_A*100:.2f} 个百分点（MCSE {r.paired_gain_mcse*100:.2f}）；Bias {a.bias:.5f} → {r.bias:.5f}，Mean SE {a.mean_se:.5f} → {r.mean_se:.5f}，区间长度 {a.mean_ci_length:.5f} → {r.mean_ci_length:.5f}。')
    for scene in SCENES:
        r=table[(table.scenario==scene)&(table.method=='D')].iloc[0]
        a=table[(table.scenario==scene)&(table.method=='A')].iloc[0]
        text.append(f'A vs D（{scene}）：覆盖率 {a.coverage:.1%} → {r.coverage:.1%}，配对增益 {r.paired_coverage_gain_vs_A*100:.2f} pp（MCSE {r.paired_gain_mcse*100:.2f}）；Bias {a.bias:.5f} → {r.bias:.5f}，Mean SE {a.mean_se:.5f} → {r.mean_se:.5f}，区间长度 {a.mean_ci_length:.5f} → {r.mean_ci_length:.5f}；平均阈值 {r.mean_tau_plus:.4f}，clipping {r.mean_clipping_rate:.3%}。')
    text.extend(['','必须一起看绝对覆盖率和 Δ_tail。较小的恶化幅度可能来自 Gaussian 表现也下降，不等于重尾推断获得保护。Bias、经验 SD、Mean SE、CI 长度和 T 分布在同一张图内，不能只凭 coverage 排序。',
                 'B/C 的不同截距和不同阈值同时变化；A–C 不能单独识别“非对称截断”或“截距”的因果贡献。本文只报告这组固定方法的整体差异，不做事后归因实验。',
                 'R=200 的百分点差异必须结合配对 MCSE；这里的结果是限定 DGP 与已保存预测下的 pilot。B 对 t、C 对 lognormal 的解释是已知分布族条件下的探索性策略；仅 D 使用跨场景统一、不读真实分布标签的规则。',
                 '', '## 代码位置与重跑', '',
                 '`stage2_losses.py`：rho、psi、psi_prime、objective_gradient、bread。`stage2_estimators.py`：preliminary_scale、fit_huber、sandwich、fit_method。`data_access.py`：preflight、load_replication。`test_stage2.py`：numerical_tests、population_lognormal_score、run_tests。`run_pilot.py`：先测试与 smoke gate，再运行 200。',
                 '准确行号见 source_locations.json；完整代码哈希、原始输入路径与哈希、种子、版本、每折标识及预测哈希分别保存在 run_config.json、input_manifest.json、integrity_check.json。逐观测 residual/psi/estimating_score/bread/meat/covariance 位于 arrays/，失败和警告保留在 replication_results.csv。',
                 '', '## 文献边界与下一步', '',
                 'D 的增长阈值规则受 [Sun、Zhou、Fan：Adaptive Huber Regression](https://pmc.ncbi.nlm.nih.gov/articles/PMC7603940/) 启发；此处 MAD 尺度与 residualized DML 的组合是实验性实现，并非该文结论在本问题上的直接应用。',
                 '当前 sandwich 是固定阈值工作近似，未自动包含阈值估计、nuisance 学习和交叉拟合相关性的全部影响。正式研究还需识别、orthogonality 和自适应阈值影响的理论核查。',
                 '本次完成 R=200 后停止，不扩大 n、不自动跑 R=1000、不再训练 nuisance、不继续 Dukes、不自动按覆盖率优化参数。是否开展进一步方法或理论分析由用户决定。'])
    (output/'pilot_report.md').write_text('\n\n'.join(text),encoding='utf-8')

# -*- coding: utf-8 -*-
"""B 方法修偏诊断。用户本轮明确授权脚本及实际运行。

这是使用已知 theta0 的 Monte Carlo Oracle 诊断，不是可直接应用于真实数据的估计器。
1. 前 200 次：同批 T 减均值，验证人为居中后 coverage；不宣称有效推断。
2. 前 200 次估计修正量，后 800 次固定修正量独立评价，保留原 HGBR 预测。
3. 真正重解二阶段方程 mean[d*psi(y-theta*d)] = b_cal。
   b_cal 是前 200 次在 theta0 上的 score 平均，不依赖评价批结果。
4. 工作 sandwich 同时报告条件 SE 及加入独立校准量估计方差后的 SE。
以上均按误差场景分别校准，不是未知分布下的统一可行方法。
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import sys, json, hashlib, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DESKTOP=Path(__file__).resolve().parent
PILOT=DESKTOP/'DML_Stage2_Robust_M_Pilot'
LEGACY=DESKTOP/'DML_Dukes_full_results'/'n1000_R1000_f76477e6a5dc'
OUTPUT=DESKTOP/'B_DML_Bias_Correction_Results'
SCENES=['gaussian','t2.1','lognormal2']
N=1000
R_CAL=200
R_TOTAL=1000
TRUTH=1.0  # 仅 Oracle 校准和评价使用，绝不隐瞒其作用。
sys.path.insert(0,str(PILOT))
from stage2_estimators import fit_method, preliminary_scale


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok,message):
    if not ok:raise RuntimeError(message)


def load(scene,rep):
    stem=f'{scene}_{rep:04d}'
    taskpath=LEGACY/'tasks'/(stem+'.json')
    task=json.loads(taskpath.read_text(encoding='utf-8'))
    path=LEGACY/task['arrays_file']
    digest=sha(path)
    require(digest==task['arrays_sha256'],'原数组哈希错误：'+stem)
    with np.load(path,allow_pickle=False) as a:
        d=a['d_res'].copy();y=a['y_res'].copy()
        require(d.shape==y.shape==a['g_hat'].shape==a['ell_hat'].shape==(N,),'维度异常')
        require(np.isfinite(d).all() and np.isfinite(y).all(),'非有限残差')
        require(hashlib.sha256(a['fold_ids'].tobytes()).hexdigest()==task['rows'][0]['fold_hash'],'fold 哈希异常')
    return d,y,{str(path):digest,str(taskpath):sha(taskpath)}


def corrected_fit(d,y,tau,offset,var_offset):
    """固定旧 B 阈值，重解凸损失 + offset*theta 的梯度方程。"""
    def equation(theta):
        return float(np.mean(d*np.clip(y-theta*d,-tau,tau))-offset)
    limit=float(tau*np.mean(abs(d)))
    require(abs(offset)<limit,'修正量超出 score 可达范围，不能求根')
    ols=float(np.mean(d*y)/np.mean(d*d))
    width=1.
    for _ in range(40):
        lo,hi=ols-width,ols+width
        if equation(lo)>=0 and equation(hi)<=0:break
        width*=2
    else:raise ArithmeticError('修正方程无法括根')
    theta=brentq(equation,lo,hi,xtol=1e-12,rtol=1e-14)
    r=y-theta*d
    u=d*np.clip(r,-tau,tau)
    A=float(np.mean(d*d*(abs(r)<tau)))
    require(np.isfinite(A) and A>1e-12,'修正后 bread 退化')
    variance_u=float(np.mean((u-u.mean())**2))
    conditional=variance_u/N/A**2
    # offset 来自独立的 200 个 replication 均值，估计其均值方差；不是逐观测方差。
    calibration=var_offset/A**2
    require(conditional>0 and np.isfinite(conditional+calibration),'修正方差非法')
    require(abs(equation(theta))<1e-9,'修正后方程误差过大')
    return dict(theta=theta,se_conditional=np.sqrt(conditional),
                se_calibration=np.sqrt(conditional+calibration),
                score_error=abs(equation(theta)),bread=A,
                clipping=float(np.mean(abs(r)>tau)))


def result_row(scene,rep,method,theta,se,**extra):
    require(np.isfinite([theta,se]).all() and se>0,'结果非有限或 SE 非正')
    t=(theta-TRUTH)/se
    return dict(scenario=scene,rep_id=rep,method=method,status='ok',theta=theta,se=se,
                T=t,covered=bool(abs(t)<=1.96),ci_length=3.92*se,**extra)


def summarize(raw):
    rows=[]
    for (scene,method),g in raw.groupby(['scenario','method'],sort=False):
        v=g[g.status=='ok']
        row=dict(scenario=scene,method=method,attempted=len(g),valid=len(v),failed=len(g)-len(v))
        if len(v):
            coverage=v.covered.astype(float).mean()
            row.update(bias=float((v.theta-TRUTH).mean()),empirical_sd=v.theta.std(ddof=1),
                       mean_se=v.se.mean(),coverage=coverage,coverage_mcse=np.sqrt(coverage*(1-coverage)/len(v)),
                       mean_T=v['T'].mean(),sd_T=v['T'].std(ddof=1),reject_left=float((v['T']< -1.96).mean()),
                       reject_right=float((v['T']>1.96).mean()),mean_ci_length=v.ci_length.mean())
            ref='B_cal200' if method in ['B_cal200','T_center_same200_ORACLE'] else 'B_eval800'
            base=raw[(raw.scenario==scene)&(raw.method==ref)&(raw.status=='ok')].set_index('rep_id')
            pair=v.set_index('rep_id').join(base[['covered']],rsuffix='_B',how='inner')
            diff=pair.covered.astype(float)-pair.covered_B.astype(float)
            row.update(paired_valid=len(diff),paired_coverage_gain=diff.mean(),
                       paired_gain_mcse=diff.std(ddof=1)/np.sqrt(len(diff)))
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame,cols):
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,row in frame.iterrows():
        vals=[f'{row[c]:.5f}' if isinstance(row[c],(float,np.floating)) else str(row[c]) for c in cols]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def plot(raw,summary):
    colors=['#64748B','#00868B','#D87935','#7650A3']
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    titles=['Gaussian','Student-t(2.1)','Centered lognormal(2)']
    for j,scene in enumerate(SCENES):
        ax=axes[0,j]
        for method,label,color in [('B_cal200','Original B','#64748B'),
                                   ('T_center_same200_ORACLE','Same-sample oracle centering','#00868B')]:
            ts=raw.loc[(raw.scenario==scene)&(raw.method==method)&(raw.status=='ok'),'T']
            ax.hist(ts,bins=np.linspace(-5,6,29),density=True,histtype='step',lw=1.5,label=label,color=color)
        x=np.linspace(-5,6,400);ax.plot(x,norm.pdf(x),'k--',label='N(0,1)')
        ax.axvline(-1.96,color='gray',ls=':');ax.axvline(1.96,color='gray',ls=':')
        ax.set_title(titles[j]+' | calibration R=200');ax.set_xlabel('T');ax.set_ylabel('Density')
        if j==0:ax.legend(fontsize=7)
        ax=axes[1,j]
        methods=['B_eval800','T_shift_eval800_ORACLE','Score_corrected_conditional_ORACLE','Score_corrected_calSE_ORACLE']
        s=summary[(summary.scenario==scene)].set_index('method').loc[methods]
        ax.bar(np.arange(4),s.coverage,color=colors)
        ax.errorbar(np.arange(4),s.coverage,yerr=1.96*s.coverage_mcse,fmt='none',ecolor='black',capsize=3)
        for k,c in enumerate(s.coverage):ax.text(k,c+.025,f'{c:.1%}',ha='center',fontsize=9)
        ax.axhline(.95,color='black',ls='--',lw=1)
        ax.set_xticks(np.arange(4),['B','T shift','Score\nconditional SE','Score\n+ calibration SE'],fontsize=8)
        ax.set_ylim(0,1.08);ax.set_ylabel('Coverage');ax.set_title('Held-out evaluation: R=800')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('B bias diagnostics | Oracle information is used for ALL corrections\nSame-sample T centering is descriptive; score correction is evaluated on held-out replications',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUTPUT/'bias_correction_results.png',dpi=190,bbox_inches='tight');plt.close(fig)


def main():
    start=time.perf_counter()
    require(not OUTPUT.exists(),'输出目录已存在，拒绝覆盖旧结果')
    config=json.loads((LEGACY/'config.json').read_text(encoding='utf-8'))
    require(config['n']==N and config['p']==5 and config['base_seed']==20260826,'原配置变化')
    old=pd.read_csv(PILOT/'replication_results.csv')
    old=old[old.method=='B'].set_index(['scenario','rep_id'])
    protected={str(p):sha(p) for p in [PILOT/'stage2_estimators.py',PILOT/'stage2_losses.py',LEGACY/'config.json',PILOT/'replication_results.csv']}
    rows=[];calibrations=[];verification=[]
    with threadpool_limits(limits=1):
        for scene in SCENES:
            cache=[];score_values=[]
            for rep in tqdm(range(R_TOTAL),desc=scene,unit='rep'):
                d,y,hashes=load(scene,rep);protected.update(hashes)
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        fit,_=fit_method(d,y,'B')
                    tau=fit['tau_plus'];theta,se=fit['theta_hat'],fit['se']
                    extra=dict(tau=tau,clipping=fit['clipping_rate'],score_error=fit['score_norm'],
                               warnings=json.dumps([str(w.message) for w in caught],ensure_ascii=False),
                               optimizer_success=fit['optimizer_success'])
                    row=result_row(scene,rep,'B_cal200' if rep<R_CAL else 'B_eval800',theta,se,**extra)
                    rows.append(row);cache.append((rep,d,y,fit))
                    if rep<R_CAL:
                        prior=old.loc[(scene,rep)]
                        delta=max(abs(theta-prior.theta_hat),abs(se-prior.se))
                        require(delta<1e-10,'B 与旧结果不符')
                        verification.append(delta)
                        score_values.append(float(np.mean(d*np.clip(y-TRUTH*d,-tau,tau))))
                except Exception as exc:
                    # 原 B 失败则不替换种子；校准集失败直接终止，避免选择偏差。
                    if rep<R_CAL:raise
                    rows.append(dict(scenario=scene,rep_id=rep,method='B_eval800',status='failed',error=str(exc)))
            cal_rows=[r for r in rows if r['scenario']==scene and r['method']=='B_cal200' and r['status']=='ok']
            require(len(cal_rows)==R_CAL,'校准组不完整')
            mean_t=float(np.mean([r['T'] for r in cal_rows]))
            var_mean_t=float(np.var([r['T'] for r in cal_rows],ddof=1)/R_CAL)
            b=float(np.mean(score_values));var_b=float(np.var(score_values,ddof=1)/R_CAL)
            calibrations.append(dict(scenario=scene,R_cal=R_CAL,R_eval=R_TOTAL-R_CAL,
                                     T_shift=mean_t,T_shift_mcse=np.sqrt(var_mean_t),score_offset=b,
                                     score_offset_mcse=np.sqrt(var_b),truth_used=TRUTH))
            for rep,d,y,fit in cache:
                if rep<R_CAL:
                    rows.append(result_row(scene,rep,'T_center_same200_ORACLE',
                                           fit['theta_hat']-mean_t*fit['se'],fit['se'],
                                           note='同批次事后居中；不是真实可用估计方法'))
                    continue
                # 额外的 T-shift 对照也只使用校准均值，不强迫评价组 Mean(T)=0。
                rows.append(result_row(scene,rep,'T_shift_eval800_ORACLE',
                                       fit['theta_hat']-mean_t*fit['se'],fit['se'],
                                       note='校准均值固定；SE 条件于校准量，不含校准误差'))
                try:
                    corrected=corrected_fit(d,y,fit['tau_plus'],b,var_b)
                    for method,key in [('Score_corrected_conditional_ORACLE','se_conditional'),
                                       ('Score_corrected_calSE_ORACLE','se_calibration')]:
                        rows.append(result_row(scene,rep,method,corrected['theta'],corrected[key],
                                               score_error=corrected['score_error'],bread=corrected['bread'],
                                               clipping=corrected['clipping'],tau=fit['tau_plus']))
                except Exception as exc:
                    for method in ['Score_corrected_conditional_ORACLE','Score_corrected_calSE_ORACLE']:
                        rows.append(dict(scenario=scene,rep_id=rep,method=method,status='failed',error=str(exc)))
            # 有限差分验证修正目标梯度，使用同一数据，固定 tau/b，不生成新数据。
            _,d,y,fit=cache[0];tau=fit['tau_plus'];theta=fit['theta_hat'];h=1e-6
            def objective(t):
                r=y-t*d
                loss=np.where(abs(r)<=tau,.5*r*r,tau*abs(r)-.5*tau*tau)
                return float(np.mean(loss)+b*t)
            fd=(objective(theta+h)-objective(theta-h))/(2*h)
            grad=-np.mean(d*np.clip(y-theta*d,-tau,tau))+b
            require(abs(fd-grad)<1e-7,'修正目标梯度核验失败')
    raw=pd.DataFrame(rows);summary=summarize(raw);cal=pd.DataFrame(calibrations)
    require(all(sha(p)==h for p,h in protected.items()),'原文件运行期间发生变化')
    for scene in SCENES:
        r=summary[(summary.scenario==scene)&(summary.method=='T_center_same200_ORACLE')].iloc[0]
        require(abs(r.mean_T)<1e-10,'同批居中后均值未达到零')
    OUTPUT.mkdir()
    raw.to_csv(OUTPUT/'replication_diagnostics.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(OUTPUT/'summary.csv',index=False,encoding='utf-8-sig')
    cal.to_csv(OUTPUT/'calibration.csv',index=False,encoding='utf-8-sig')
    plot(raw,summary)
    report=['# B 方法修偏诊断结果','',
            '**本次所有修正均使用模拟真值 θ₀=1，属于 Oracle 诊断，不能宣称得到可应用于真实数据的修偏 DML。**',
            '同批次事后居中严格令 Mean(T)=0；均值为零不保证方差为一或 95% coverage。',
            '进一步用旧 rep 0–199 校准，在 rep 200–999（每场景 800 次）上固定修正量重解二阶段。未生成新 DGP，未训练 HGBR，原预测/树参数/种子不变。',
            't 和 lognormal 共享 X/treatment 种子，各场景结果不作相互独立处理。本次按误差标签分别校准，不是跨分布通用方法。',
            '', '## 修正方式','',
            '1. `T_center_same200_ORACLE`：μ=前200次 Mean(T)，θ_c=θ_B−μ SE_B，T_c=T_B−μ。SE 不变；同批 θ₀ 泄漏，coverage 仅作事后诊断。',
            '2. `T_shift_eval800_ORACLE`：在后800次用固定 μ；没有再用评价组均值强行归零；SE 条件于校准结果。',
            '3. `Score_corrected_*_ORACLE`：前200次估计 b=mean_rep[mean_i(d_i ψτ(y_i−θ₀d_i))]；后800次求 mean_i(d_i ψτ(y_i−θd_i))=b。等价于最小化 meanρτ(y−θd)+bθ，采用括根与 Brent 求根。仍然无截距，τ=1.345×原初步 OLS 残差 MAD，不重新调阈值。',
            '4. 修正后的 A=mean[d² I(|r|<τ)]，u=dψτ(r)。条件方差 var_n(u)/(n A²)；校准误差版本额外加 var_rep(score_mean)/(200 A²)。校准量按独立 Monte Carlo replication 的方差估计；阈值与 nuisance 的全部不确定性仍未包含。',
            '校准集和评价集不重叠，但它们都是同一已知 DGP 的既有模拟。评价结果仅用于本次诊断，不据此证明通用有效性；固定一次校准的覆盖率也不等于重复校准流程的无条件覆盖率。',
            '', '## 校准量','',md_table(cal,['scenario','T_shift','T_shift_mcse','score_offset','score_offset_mcse']),
            '', '## 数值结果','',md_table(summary,['scenario','method','valid','failed','bias','mean_se','coverage','coverage_mcse','mean_T','sd_T','reject_left','reject_right','mean_ci_length','paired_coverage_gain','paired_gain_mcse']),
            '', '## 结果解释','']
    for scene in SCENES:
        s=summary[summary.scenario==scene].set_index('method')
        a=s.loc['B_cal200'];c=s.loc['T_center_same200_ORACLE'];e=s.loc['B_eval800'];r=s.loc['Score_corrected_calSE_ORACLE']
        report.append(f'{scene}：原200次 B 的 Mean(T)={a.mean_T:.4f}，直接居中后 {c.mean_T:.3g}，coverage {a.coverage:.1%}→{c.coverage:.1%}，SD(T)仍为{c.sd_T:.4f}。独立800次上，B coverage {e.coverage:.1%}，重新求解修正 score 并计校准误差后 {r.coverage:.1%}（配对增益 {r.paired_coverage_gain:.1%}，MCSE {r.paired_gain_mcse:.2%}），Mean(T)={r.mean_T:.4f}、SD(T)={r.sd_T:.4f}，区间长度 {e.mean_ci_length:.4f}→{r.mean_ci_length:.4f}。')
    report.extend(['','## 核验与结论边界','',
                   f'旧前200次 B 最大 θ/SE 差 {max(verification):.3g}；原 {len(protected)} 个输入/代码文件前后哈希一致。修正目标的有限差分梯度通过 1e-7 精度检查；最大修正方程误差 {raw.score_error.max():.3g}。失败 {int((raw.status!="ok").sum())} 条，失败 coverage 为空，未当作 0；警告与逐次结果保存在 replication_diagnostics.csv。',
                   '本实验回答“若知道并移除偏移，能恢复多少 coverage”。它没有提供未知 θ₀ 下可计算的 b 或 μ，也没有解决稳健 score 的 nuisance orthogonality。不能把 Oracle 校准后的改善作为实际可行方法优于 B 的研究结论。',
                   '评价组均值允许非零：偏移校准误差、非线性和 SE 异质性均会影响它。若再用评价组真值居中以追求更好覆盖率，就会重新引入同批泄漏，因此不这样修改。',
                   '已停止。未扩样本量、未训练 nuisance、未覆盖旧实验。'])
    (OUTPUT/'bias_correction_report.md').write_text('\n\n'.join(report),encoding='utf-8')
    metadata=dict(completed=True,seconds=time.perf_counter()-start,n=N,calibration_ids='0..199',evaluation_ids='200..999',
                  oracle_truth_used=True,hashes=protected,script_sha256=sha(__file__),failed=int((raw.status!='ok').sum()),
                  timestamp=datetime.now().astimezone().isoformat())
    (OUTPUT/'audit.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[['scenario','method','valid','coverage','mean_T','sd_T']].to_string(index=False))
    print('Saved:',OUTPUT)


if __name__=='__main__':main()

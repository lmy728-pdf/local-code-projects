"""Read-only review of existing smoke arrays; never train or simulate."""
import hashlib,json,pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from legacy_core import fit_mean_dml
ROOT=pathlib.Path(__file__).resolve().parent
RUN=ROOT/'outputs'/'20261101_202557_smoke_9ef616ec'
OUT=ROOT/'reports'/'smoke_review_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    inputs=list(RUN.glob('*.csv'))+[RUN/'config.json']+list((RUN/'arrays').glob('*.npz'))+list((RUN/'tasks').glob('*.json'))
    hashes={str(p):sha(p) for p in inputs}
    config=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    df=pd.read_csv(RUN/'replications.csv',float_precision='round_trip')
    summary=pd.read_csv(RUN/'summary.csv',float_precision='round_trip')
    assert config['stage']=='smoke' and config['replications']==2 and len(df)==18
    assert (df.status=='ok').all() and not df.duplicated(['scenario','rep_id','method']).any()
    records=[]
    legacy=pathlib.Path(json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))['legacy_run'])
    for (scenario,rep),group in df.groupby(['scenario','rep_id']):
        assert set(group.method)=={'A','B','C'}
        assert group.fold_hash.nunique()==group.treatment_prediction_hash.nunique()==1
        b=json.loads(group[group.method=='B'].iloc[0].learner_params)
        c=json.loads(group[group.method=='C'].iloc[0].learner_params)
        assert {key for key in b if b[key]!=c[key]}=={'loss'}
        stem=f'{scenario}_{rep:04d}'
        with np.load(RUN/'arrays'/f'{stem}.npz') as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            folds=np.empty(1000,dtype=np.int64)
            for k,(_,test) in enumerate(KFold(5,shuffle=True,random_state=int(group.iloc[0].crossfit_seed)).split(z['X'])):folds[test]=k
            assert np.array_equal(folds,z['fold_ids'])
            task=json.loads((legacy/'tasks'/f'{stem}.json').read_text(encoding='utf-8'))
            with np.load(legacy/task['arrays_file']) as old:
                assert np.array_equal(z['m_hat'],old['g_hat']) and np.array_equal(z['ell_A'],old['ell_hat'])
            for row in group.itertuples():
                theta,se=fit_mean_dml(z['D']-z['m_hat'],z['Y']-z['ell_'+row.method])
                error=z['ell_'+row.method]-z['ell0']
                checks=dict(theta_error=theta-row.theta_hat,se_error=se-row.se,
                            T_error=(theta-1)/se-row.t_stat,
                            outcome_rmse_error=np.sqrt(np.mean(error**2))-row.outcome_rmse,
                            outcome_mae_error=np.mean(abs(error))-row.outcome_mae,
                            treatment_rmse_error=np.sqrt(np.mean((z['m_hat']-z['m0'])**2))-row.treatment_rmse)
                assert max(abs(v) for v in checks.values())<1e-12
                assert row.covered==bool(theta-1.96*se<=1<=theta+1.96*se)
                records.append(dict(scenario=scenario,rep_id=rep,method=row.method,**checks))
    pd.DataFrame(records).to_csv(OUT/'numerical_checks.csv',index=False)
    metrics=['theta_hat','se','t_stat','outcome_rmse','outcome_mae','outcome_error_mean']
    paired=[]
    for (scenario,rep),group in df.groupby(['scenario','rep_id']):
        b=group[group.method=='B'].iloc[0];c=group[group.method=='C'].iloc[0]
        paired.append(dict(scenario=scenario,rep_id=rep,**{key+'_C_minus_B':c[key]-b[key] for key in metrics}))
    pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
    names={'gaussian':'Gaussian','t2.1':'Student-t(2.1)','lognormal2':'Centered lognormal(2)'}
    rows=[];comparisons=[]
    for scenario,label in names.items():
        for method in ['A','B','C']:
            r=summary[(summary.scenario==scenario)&(summary.method==method)].iloc[0]
            rows.append([label,method,'0/2',f'{r.bias:.5f}',f'{r.mean_se:.5f}',f'{r.mean_T:.3f}',f'{r.sd_T:.3f}',f'{r.outcome_rmse:.5f}',f'{r.outcome_mae:.5f}'])
        b=summary[(summary.scenario==scenario)&(summary.method=='B')].iloc[0]
        c=summary[(summary.scenario==scenario)&(summary.method=='C')].iloc[0]
        comparisons.append([label,f'{c.bias-b.bias:+.5f}',f'{c.mean_T-b.mean_T:+.3f}',f'{c.outcome_rmse-b.outcome_rmse:+.5f}',f'{c.outcome_mae-b.outcome_mae:+.5f}',f'{c.mean_se-b.mean_se:+.5f}'])
    estimates=table(['场景','方法','覆盖次数','Bias','Mean SE','Mean(T)','SD(T)','Outcome RMSE','Outcome MAE'],rows)
    contrast=table(['场景','ΔBias','ΔMean(T)','ΔRMSE','ΔMAE','ΔMean SE'],comparisons)
    diag=df.drop_duplicates(['scenario','rep_id'])
    noise=table(['场景','rep ID','epsilon样本方差','epsilon最大绝对值'],[[names[r.scenario],r.rep_id,f'{r.epsilon_variance:.4f}',f'{r.epsilon_max_abs:.4f}'] for r in diag.itertuples()])
    max_error=max(abs(r[k]) for r in records for k in r if k.endswith('_error'))
    warnings=sum(bool(json.loads(s)) for s in df.warnings)
    report=f'''# Huber-DML 当前结果解读

**结论：代码通过了本次 smoke 检查，但正式pilot尚未运行；现在不足以判断Huber能否改善覆盖率。**

读取目录：`{RUN}`。配置为DGP2、n=1000、p=5、五折、theta0=1、Huber alpha=0.90；**每场景仅2次，rep ID为0和1**，合计6组数据、18条方法结果。不是R=100，更不是原Dukes的R=1000。本报告只读取已保存结果，没有追加模拟、拟合模型或改动结果。

## 1. 实现检查

- 18/18条成功，失败0，warnings记录{warnings}，保存数组均有限。
- 6组A的treatment和outcome OOF预测与旧Mean-DML保存数组逐项完全相同；表内legacy theta/SE差异也为0。
- 每组A/B/C的fold hash和treatment预测hash相同。由原KFold种子重建的fold与NPZ完全相同。
- B/C保存的学习器参数仅loss不同；共同alpha=.90在B中不生效，在C中为Huber分位数参数。
- 从保存的D、Y、m_hat、ell_A/B/C独立重算原第二阶段theta、IF-SE、T及oracle误差，最大绝对数值差为{max_error:.3g}；全部覆盖标记一致。证据见[numerical_checks.csv](numerical_checks.csv)。

这里A=HGBR平方损失，B=GBR平方损失，C=GBR Huber，与前面Dukes实验的B/C不是同一方法。

## 2. 实际观测到什么

{estimates}

所有方法在各场景均为0/2覆盖；18个T均大于1.96。**这仅表示这两次固定seed的置信区间都没有覆盖真值，不能写成“方法的真实coverage为0%”。** 每次覆盖事件便会改变该场景比例50个百分点，无法识别百分点级改善。

原summary中的coverage MCSE=0来自sqrt(p_hat*(1-p_hat)/2)在p_hat=0时的边界代入，**不表示估计没有不确定性**。同理，两个配对覆盖差都为0，配对MCSE=0也不能证明B/C等效。此处不依据这些零值作效果检验。

## 3. 重点看C−B

下表Δ=C−B。RMSE/MAE都对真实条件均值ell0计算。

{contrast}

- Gaussian：Huber的Mean(T)只下降约0.035，平均RMSE和MAE均略升；没有明确的预测收益。
- Student-t(2.1)：Mean(T)由3.167降到2.828，Bias由0.08295降到0.07731，MAE由0.49114降到0.48151；但RMSE由0.75630升到0.75986。两次的T都下降、MAE都改善，RMSE则一好一坏。
- Centered lognormal(2)：Mean(T)由2.834降到2.638，MAE由0.48865降到0.47600；但RMSE由0.75787升到0.76236，两次RMSE都上升。T一升一降，平均下降主要由rep 1贡献，不能称为逐次一致改善。

六组中C的Mean SE相对B按场景平均均略升，因此T下降不能全部解释为偏差减小，要同时看分子theta_hat−1和分母SE。两组重尾样本中MAE均改善，却没有平均RMSE改善；更不能把“MAE改善”等同于“均值nuisance估计改善”或“coverage改善”。逐次差异见[paired_differences.csv](paired_differences.csv)。

lognormal下平均预测误差从B的−0.00280变为C的−0.04956，出现更明显的负向偏移。这与需要检查Huber条件位置目标的提醒相符，但仅凭两组样本不能确定其来源或总体大小，也不能据此偷偷加入均值校准。

## 4. 重尾诊断和coverage deterioration

{noise}

误差采用总体方差标准化，不保证每次有限样本方差等于1；不要按这些样本方差重新标准化。如此少的replication尤其不能充分代表罕见尾部事件。

因各方法各场景目前都是0/2，计算出的loss_tail及gain_robust全部为0，**这些零没有方向筛选价值**：它们是本次极少样本覆盖指示相同的结果，不是证明重尾没有额外影响或Huber没有收益。t与lognormal沿用同一seed编号及部分共同随机数，跨场景差异也不应直接按独立样本处理。

## 5. 下一步

本轮支持的结论是“工程已跑通，原A可复现，三组配对成立”。Huber在这几组重尾样本的MAE和平均T上有值得继续观察的变化，但对RMSE和coverage尚无改善证据。

下一步是运行已经写好的**R=100正式pilot**：notebook参数单元将 `STAGE = "smoke"` 改为 `STAGE = "pilot"`，然后从参数单元按顺序运行。保持alpha=.90、原DGP、原seed和所有参数不变，不筛选有利样本。正式pilot会先核验前2次，再执行余下98次；新建输出目录，不覆盖本轮。

R=100仍只是趋势筛选，最坏coverage MCSE约5个百分点。应同时看C−B的配对coverage差/MCSE、绝对coverage、Bias/Mean(T)、RMSE和MAE，不凭单个指标定结论。**本次没有替你启动pilot。**
'''
    (OUT/'huber_smoke_review.md').write_text(report,encoding='utf-8')
    assert all(sha(pathlib.Path(p))==h for p,h in hashes.items())
    evidence=dict(stage=config['stage'],replications_per_scenario=2,datasets=6,method_rows=18,failed=0,warnings=warnings,
                  all_legacy_A_predictions_equal=True,folds_verified=True,shared_treatment_verified=True,
                  bc_only_loss_differs=True,max_recalculation_difference=max_error,source_files_unchanged=True,
                  new_simulations=0,input_sha256=hashes)
    (OUT/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Review complete; UTF-8 report and verification saved.')
if __name__=='__main__':main()

"""Summaries are generated only from a user-launched run; no synthetic results."""
import json,pathlib
import numpy as np
import pandas as pd

def summarize(output):
    output=pathlib.Path(output);cfg=json.loads((output/'config.json').read_text(encoding='utf-8'))
    df=pd.read_csv(output/'replications.csv')
    for c in ['theta_hat','se','covered','ci_length','t_stat','outcome_rmse','outcome_mae','outcome_error_mean','treatment_rmse']:
        if c not in df:df[c]=np.nan
    records=[];pairs=[];pair_summaries=[]
    for scenario in [s['name'] for s in cfg['scenarios']]:
        for method in ['A','B','C']:
            group=df[(df.scenario==scenario)&(df.method==method)]
            valid=group[(group.status=='ok')&np.isfinite(group.t_stat)]
            n=len(valid);theta=valid.theta_hat;T=valid.t_stat
            coverage=float(valid.covered.astype(float).mean()) if n else np.nan
            records.append(dict(scenario=scenario,method=method,planned=cfg['replications'],attempted=len(group),valid=n,
                                failed=int((group.status!='ok').sum()),not_attempted=cfg['replications']-len(group),
                                nonfinite_T=int((~np.isfinite(group.t_stat)).sum()),
                                bias=float(theta.mean()-cfg['theta0']),empirical_sd=float(theta.std(ddof=1)),
                                mean_se=float(valid.se.mean()),rmse=float(np.sqrt(np.mean((theta-cfg['theta0'])**2))),
                                coverage=coverage,coverage_mcse=np.sqrt(coverage*(1-coverage)/n) if n else np.nan,
                                mean_ci_length=float(valid.ci_length.mean()),mean_T=float(T.mean()),sd_T=float(T.std(ddof=1)),
                                p_left=float((T < -1.96).mean()) if n else np.nan,p_right=float((T>1.96).mean()) if n else np.nan,
                                outcome_rmse=float(valid.outcome_rmse.mean()),outcome_mae=float(valid.outcome_mae.mean()),
                                outcome_error_mean=float(valid.outcome_error_mean.mean()),treatment_rmse=float(valid.treatment_rmse.mean())))
        bb=df[(df.scenario==scenario)&(df.method=='B')].set_index('rep_id')
        cc=df[(df.scenario==scenario)&(df.method=='C')].set_index('rep_id')
        local=[]
        for rep in sorted(set(bb.index)&set(cc.index)):
            b,c=bb.loc[rep],cc.loc[rep];ok=b.status==c.status=='ok'
            row=dict(scenario=scenario,rep_id=rep,valid_pair=ok)
            if ok:
                assert b.fold_hash==c.fold_hash and b.treatment_prediction_hash==c.treatment_prediction_hash
                row.update(covered_C_minus_B=int(c.covered)-int(b.covered),
                           theta_C_minus_B=c.theta_hat-b.theta_hat,T_C_minus_B=c.t_stat-b.t_stat,
                           outcome_rmse_C_minus_B=c.outcome_rmse-b.outcome_rmse,
                           outcome_mae_C_minus_B=c.outcome_mae-b.outcome_mae,
                           outcome_error_mean_C_minus_B=c.outcome_error_mean-b.outcome_error_mean)
                local.append(row)
            pairs.append(row)
        for metric in ['covered_C_minus_B','theta_C_minus_B','T_C_minus_B','outcome_rmse_C_minus_B','outcome_mae_C_minus_B','outcome_error_mean_C_minus_B']:
            values=np.array([row[metric] for row in local],float)
            pair_summaries.append(dict(scenario=scenario,metric=metric,valid_pairs=len(values),
                                       mean_difference=float(values.mean()) if len(values) else np.nan,
                                       paired_mcse=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else np.nan))
    summary=pd.DataFrame(records);paired=pd.DataFrame(pair_summaries)
    summary.to_csv(output/'summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'paired_replications.csv',index=False)
    paired.to_csv(output/'paired_comparisons.csv',index=False)
    df[df.status!='ok'].to_csv(output/'failures.csv',index=False)
    df[df.warnings!='[]'].to_csv(output/'warnings.csv',index=False)
    indexed=summary.set_index(['scenario','method']);tails=[]
    for scenario in ['t2.1','lognormal2']:
        losses={m:indexed.loc[('gaussian',m),'coverage']-indexed.loc[(scenario,m),'coverage'] for m in ['A','B','C']}
        tails.append(dict(scenario=scenario,loss_tail_A=losses['A'],loss_tail_B=losses['B'],loss_tail_C=losses['C'],
                          gain_robust=losses['B']-losses['C'],absolute_coverage_C=indexed.loc[(scenario,'C'),'coverage'],
                          inference='Descriptive only: cross-scenario coupling not assumed independent or paired for MCSE'))
    pd.DataFrame(tails).to_csv(output/'tail_deterioration.csv',index=False)
    plot(summary,output,cfg)
    table='|场景|方法|有效/计划|Coverage|MCSE|Bias|Mean(T)|SD(T)|Outcome RMSE|\n|---|---|---|---|---|---|---|---|---|\n'
    for row in records:
        table+=f"|{row['scenario']}|{row['method']}|{row['valid']}/{row['planned']}|{row['coverage']:.3f}|{row['coverage_mcse']:.3f}|{row['bias']:.4f}|{row['mean_T']:.3f}|{row['sd_T']:.3f}|{row['outcome_rmse']:.3f}|\n"
    paired_text='\n'.join(f"- {r.scenario}: {r.metric} = {r.mean_difference:.5f}，配对MCSE={r.paired_mcse:.5f}，有效配对{r.valid_pairs}。" for r in paired.itertuples())
    tail_text='\n'.join(f"- {r['scenario']}: loss_tail(B)={r['loss_tail_B']:.3f}，loss_tail(C)={r['loss_tail_C']:.3f}，gain_robust={r['gain_robust']:.3f}，C绝对coverage={r['absolute_coverage_C']:.3f}。" for r in tails)
    report=f'''# Huber outcome DML {cfg['stage']} 报告

本次计划每场景{cfg['replications']}次，已尝试{len(df)//3}个数据集。失败方法记录{int((df.status!='ok').sum())}条；未尝试数和失败数明确保留，统计表仅对有效结果计算，不把失败当作未覆盖，也不更换seed。

{table}

重点比较C−B（相同GBR架构，仅loss不同）；C−A同时包含架构变化，不能全部归因于Huber。

{paired_text}

{tail_text}

coverage和gain单位均为比例；0.01为1个百分点。跨场景gain仅描述，不报告未验证耦合下的显著性。R=200最坏coverage MCSE约3.54个百分点，不能因小幅变化宣布有效；smoke结果不能作为效果证据。结果用于方向筛选，不自动扩展R、DGP或alpha。

Huber总体目标未必是条件均值，尤其偏态lognormal可能引入位置偏移。预测误差均以ell0=theta0*m0+g0评估，未作oracle校准；真值仅用于数据生成与评估，不进入模型训练或第二阶段估计。当前独立加性误差下，常数位置偏移在正确treatment nuisance等条件下未必影响总体识别，但有限样本偏差与SE仍需检查，不能声称保留全部DML理论保证。

仅改第一阶段并不保证第二阶段稳健。如果nuisance RMSE改善而coverage未改善，需要结合score/SE诊断，不能直接概括为Huber无效。本轮未加入第二阶段Huber、Dukes或均值校准。

![Coverage](coverage_comparison.png)

![Outcome nuisance RMSE](nuisance_rmse.png)

逐次指标见replications.csv，配对指示差见paired_replications.csv；fold、全部OOF预测和原始数据在arrays/；配置与源码hash在config.json。A逐次旧结果复现检查在legacy_*列。
'''
    (output/'pilot_report.md').write_text(report,encoding='utf-8')

def plot(summary,output,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scenarios=['gaussian','t2.1','lognormal2']
    for metric,name,title in [('coverage','coverage_comparison.png','95% Wald coverage'),('outcome_rmse','nuisance_rmse.png','Outcome OOF RMSE against true conditional mean')]:
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
        for j,method in enumerate(['A','B','C']):
            vals=summary[summary.method==method].set_index('scenario').loc[scenarios,metric].to_numpy()
            ax.bar(np.arange(3)+(j-1)*.24,vals,.24,label={'A':'A: HGBR squared','B':'B: GBR squared','C':'C: GBR Huber (adaptive delta, q=0.90)'}[method])
        if metric=='coverage':ax.axhline(.95,ls='--',color='black',label='Nominal 95%');ax.set_ylim(0,1.05)
        ax.set(xticks=np.arange(3),xticklabels=scenarios,ylabel='Proportion' if metric=='coverage' else 'Response units',
               title=f'{title}; n=1000, planned R={cfg["replications"]} per scenario')
        ax.legend(fontsize=8);fig.savefig(output/name,dpi=160);plt.close(fig)

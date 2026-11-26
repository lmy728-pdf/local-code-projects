"""Streaming progress and one cumulative summary/coverage figure. No fit on import."""
import json,pathlib,shutil
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm.auto import tqdm

def compact_summary(rows,config):
    df=pd.DataFrame(rows)
    for key in ['theta_hat','se','t_stat','covered','ci_length','outcome_rmse','outcome_mae','outcome_error_mean','treatment_rmse']:
        if key not in df:df[key]=np.nan
    result=[]
    for scenario in [s['name'] for s in config['scenarios']]:
        for method in ['A','B','C']:
            group=df[(df.scenario==scenario)&(df.method==method)] if len(df) else df
            valid=group[(group.status=='ok')&np.isfinite(group.t_stat)] if len(group) else group
            n=len(valid);cov=float(valid.covered.astype(float).mean()) if n else np.nan
            row=dict(run_id=config['run_id'],scenario=scenario,method=method,
                     planned=config['replications'],completed=len(group),valid=n,
                     failed=int((group.status!='ok').sum()) if len(group) else 0,
                     not_completed=config['replications']-len(group),
                     warning_records=sum(bool(json.loads(w)) for w in group.warnings) if len(group) else 0,
                     coverage=cov,coverage_mcse=np.sqrt(cov*(1-cov)/n) if n else np.nan,
                     bias=float(valid.theta_hat.mean()-config['theta0']),
                     empirical_sd=float(valid.theta_hat.std(ddof=1)),mean_se=float(valid.se.mean()),
                     theta_rmse=float(np.sqrt(np.mean((valid.theta_hat-config['theta0'])**2))),
                     mean_ci_length=float(valid.ci_length.mean()),mean_T=float(valid.t_stat.mean()),
                     sd_T=float(valid.t_stat.std(ddof=1)),
                     p_left=float((valid.t_stat < -1.96).mean()) if n else np.nan,
                     p_right=float((valid.t_stat > 1.96).mean()) if n else np.nan,
                     outcome_rmse=float(valid.outcome_rmse.mean()),outcome_mae=float(valid.outcome_mae.mean()),
                     outcome_error_mean=float(valid.outcome_error_mean.mean()),
                     treatment_rmse=float(valid.treatment_rmse.mean()))
            result.append(row)
    summary=pd.DataFrame(result)
    for scenario in summary.scenario.unique():
        block=df[(df.scenario==scenario)&(df.status=='ok')]
        b=block[block.method=='B'].set_index('rep_id');c=block[block.method=='C'].set_index('rep_id')
        ids=b.index.intersection(c.index);mask=(summary.scenario==scenario)&(summary.method=='C')
        summary.loc[mask,'BC_valid_pairs']=len(ids)
        for field in ['covered','theta_hat','t_stat','outcome_rmse','outcome_mae']:
            delta=c.loc[ids,field].astype(float)-b.loc[ids,field].astype(float)
            summary.loc[mask,'C_minus_B_'+field]=delta.mean()
            summary.loc[mask,'C_minus_B_'+field+'_mcse']=delta.std(ddof=1)/np.sqrt(len(delta)) if len(delta)>1 else np.nan
    for method in ['A','B','C']:
        normal=summary.loc[(summary.scenario=='gaussian')&(summary.method==method),'coverage'].iloc[0]
        summary.loc[summary.method==method,'loss_tail_vs_gaussian']=normal-summary.loc[summary.method==method,'coverage']
    for scenario in ['t2.1','lognormal2']:
        block=summary[summary.scenario==scenario].set_index('method')
        summary.loc[(summary.scenario==scenario)&(summary.method=='C'),'gain_robust_descriptive']=block.loc['B','loss_tail_vs_gaussian']-block.loc['C','loss_tail_vs_gaussian']
    return summary

def publish(rows,config,output,desktop,completed_label,show=True):
    output=pathlib.Path(output);desktop=pathlib.Path(desktop)
    summary=compact_summary(rows,config)
    # Preserve every run in its own directory; desktop is a two-file latest view.
    raw=pd.DataFrame(rows).sort_values(['scenario','rep_id','method'])
    raw.to_csv(output/'replications.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    scenarios=[s['name'] for s in config['scenarios']]
    for j,(method,label,color) in enumerate([('A','A: HGBR squared','#8a97a6'),('B','B: GBR squared','#4779ac'),('C','C: adaptive Huber q=0.90','#c45142')]):
        values=summary[summary.method==method].set_index('scenario').loc[scenarios]
        bars=ax.bar(np.arange(3)+(j-1)*.24,100*values.coverage.to_numpy(),.24,label=label,color=color)
        ax.bar_label(bars,labels=[f'{100*v:.1f}%' if np.isfinite(v) else 'N/A' for v in values.coverage],padding=3,fontsize=9)
    ax.axhline(95,color='black',ls='--',lw=1,label='Nominal 95%')
    counts=[int(summary[summary.scenario==s].completed.min()) for s in scenarios]
    ticks=[f'{s}\ncompleted {n}/{config["replications"]}' for s,n in zip(scenarios,counts)]
    ax.set(xticks=np.arange(3),xticklabels=ticks,ylim=(0,110),ylabel='Coverage at theta = 1 (%)',
           title=f'Huber-DML | n=1000 | {completed_label}\nRun: {config["run_id"]}')
    ax.legend(fontsize=8,ncol=2,loc='upper right')
    fig.savefig(output/'coverage_comparison.png',dpi=160);plt.close(fig)
    for src,dst in [('summary.csv','Huber_DML_summary.csv'),('coverage_comparison.png','Huber_DML_coverage.png')]:
        target=desktop/dst;tmp=target.with_name(target.stem+'.tmp'+target.suffix)
        try:
            shutil.copyfile(output/src,tmp);tmp.replace(target)
        except OSError as error:
            print(f'Desktop export unavailable ({target.name}): {error}; run copy is saved: {output}',flush=True)
    if show:
        columns=['scenario','method','completed','valid','failed','coverage','bias','mean_T','outcome_rmse']
        try:
            from IPython import get_ipython
            from IPython.display import display,Markdown
            if get_ipython() is not None:
                display(Markdown(f'**累计结果：{completed_label}**（截至当前全部replication，不是仅最近20次）'))
                display(summary[columns].round(4))
            else:print(completed_label+'\n'+summary[columns].round(4).to_string(index=False),flush=True)
        except ImportError:print(summary[columns].to_string(index=False),flush=True)
    return summary

def train_with_progress(config,output,legacy_run,worker,desktop,workers=4,show=True):
    rows=[];planned=int(config['replications']);total=3*planned
    progress=tqdm(total=total,desc='Training A/B/C',unit='dataset',dynamic_ncols=True)
    def batch(start,stop):
        jobs=[(s,r) for r in range(start,stop) for s in config['scenarios']]
        stream=Parallel(n_jobs=workers,backend='loky',return_as='generator_unordered')(
            delayed(worker)(config,s,r,output,legacy_run) for s,r in jobs)
        for group in stream:
            rows.extend(group);progress.update(1)
            progress.set_postfix(scenario=group[0]['scenario'],rep=group[0]['rep_id']+1,refresh=False)
    try:
        batch(0,min(2,planned))
        if any(r['status']!='ok' for r in rows):
            publish(rows,config,output,desktop,'smoke failed; stopped',show)
            raise RuntimeError('Smoke failed. Saved failures; no seed replacement or pilot continuation.')
        progress.write('Smoke passed: 2 replications per scenario; included in the final total.')
        if planned<=2:
            return publish(rows,config,output,desktop,'smoke complete (2 per scenario)',show)
        start=2
        for stop in range(20,planned+20,20):
            stop=min(stop,planned)
            batch(start,stop)
            publish(rows,config,output,desktop,f'{stop}/{planned} per scenario',show)
            start=stop
        return compact_summary(rows,config)
    except BaseException:
        if rows:pd.DataFrame(rows).sort_values(['scenario','rep_id','method']).to_csv(pathlib.Path(output)/'replications.csv',index=False,encoding='utf-8-sig')
        raise
    finally:progress.close()

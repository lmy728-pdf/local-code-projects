"""Summarize direct author R results and compare with the unchanged original A/B/C."""
import hashlib,json,math,pathlib,platform,sys
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parent
RUN=ROOT.parent/'n1000_R1000_f76477e6a5dc'
REFERENCE=ROOT.parent/'R_cross_validation'
SCENARIOS=['gaussian','t2.1','lognormal2']
LABELS={'gaussian':'Gaussian','t2.1':'Student-t(2.1)','lognormal2':'Centered lognormal(2)'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def wilson(k,n):
    z=norm.ppf(.975);p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(mid-rad),float(mid+rad)
def main():
    sample=pd.read_csv(ROOT/'replications.csv',float_precision='round_trip')
    raw=pd.read_csv(RUN/'raw_results.csv',float_precision='round_trip')
    assert len(sample)==3000 and len(sample[['scenario','replication']].drop_duplicates())==3000
    rrows=[]; prior_checks=[]
    for meta in sample.to_dict('records'):
        path=ROOT/'results'/f"{meta['stem']}.json"
        if not path.exists():raise RuntimeError('Incomplete R results: '+meta['stem'])
        r=json.loads(path.read_text(encoding='utf-8'))
        assert sha(ROOT/'inputs'/f"{meta['stem']}.csv")==meta['input_sha256']==r['input_sha256']
        assert r['r_auxiliary_seed']==meta['r_auxiliary_seed'] and r['scenario']==meta['scenario'] and r['replication']==meta['replication']
        row={**meta,'status_R':r['status'],'warnings_R':json.dumps(r['warnings'],ensure_ascii=False),
             'warning_count_R':len(r['warnings']),'error_R':r.get('error',''),'elapsed_seconds_R':r['elapsed_seconds']}
        if r['status']=='ok':
            T=r['T_R'];p=r['p_R'];original=r['author_original']
            assert np.isfinite([T,p,r['score_mean'],r['score_sd']]).all() and r['score_sd']>0
            assert math.isclose(p,2*norm.sf(abs(T)),rel_tol=1e-11,abs_tol=1e-14)
            assert math.isclose(T,original['est']/(original['se']*math.sqrt(1000)),rel_tol=1e-12,abs_tol=1e-12)
            row.update(T_R=T,p_R=p,accepted_R=r['accepted'],score_mean_R=r['score_mean'],score_sd_R=r['score_sd'],
                       author_est=original['est'],author_se=original['se'],T_difference_R_minus_Python=T-meta['T_Python'],
                       accepted_Python=meta['p_Python']>=.05,decision_disagrees=bool(r['accepted']!=(meta['p_Python']>=.05)))
        rrows.append(row)
        prior=REFERENCE/'r_results'/f"{meta['stem']}.json"
        if prior.exists():
            old=json.loads(prior.read_text(encoding='utf-8'))
            same=r['status']=='ok' and old['status']=='ok'
            error=max(abs(r['author_original'][k]-old['author_original'][k]) for k in ['est','se','p_val']) if same else np.nan
            assert same and error<1e-12
            prior_checks.append(dict(scenario=meta['scenario'],replication=meta['replication'],max_abs_author_output_difference=error,
                                     T_difference=r['T_R']-old['T_original'],same_auxiliary_seed=r['r_auxiliary_seed']==old['r_auxiliary_seed']))
    d=pd.DataFrame(rrows)
    d.to_csv(ROOT/'r_python_comparison.csv',index=False,float_format='%.17g')
    d[d.status_R!='ok'].to_csv(ROOT/'failures.csv',index=False)
    d[d.warning_count_R>0].to_csv(ROOT/'warnings.csv',index=False)
    pd.DataFrame(prior_checks).to_csv(ROOT/'prior_30_reproduction.csv',index=False,float_format='%.17g')
    assert len(prior_checks)==30
    statistics=[]
    for method,label in [('original_mean_dml','A'),('standard_score','B'),('dukes_dr_gcm','C_Python')]:
        for r in raw[raw.method==method].to_dict('records'):
            T=(r['theta_hat']-1)/r['theta_se'] if label=='A' else r['T']
            statistics.append(dict(scenario=r['scenario'],replication=r['replication'],method=label,status=r['status'],
                                   T=T,accepted=bool(r['covered']),p_value=2*norm.sf(abs(T)) if label=='A' else r['p_value'],
                                   warning_count=len(json.loads(r['warnings']))))
    for r in d.to_dict('records'):
        statistics.append(dict(scenario=r['scenario'],replication=r['replication'],method='C_R',status=r['status_R'],
                               T=r.get('T_R',np.nan),accepted=r.get('accepted_R',None),p_value=r.get('p_R',np.nan),
                               warning_count=r['warning_count_R']))
    s=pd.DataFrame(statistics)
    s.to_csv(ROOT/'standardized_statistics.csv',index=False,float_format='%.17g')
    summaries=[];paired=[]
    for scenario in SCENARIOS:
        for method in ['A','B','C_Python','C_R']:
            allrows=s[(s.scenario==scenario)&(s.method==method)]
            valid=allrows[(allrows.status=='ok')&np.isfinite(allrows['T'])];T=valid['T'];n=len(valid)
            if n==0:raise RuntimeError('No valid results: '+scenario+' '+method)
            k=int(valid.accepted.sum());lo,hi=wilson(k,n);coverage=k/n
            summaries.append(dict(scenario=scenario,method=method,planned=1000,valid=n,failed=int((allrows.status!='ok').sum()),
                                  nonfinite_T=int((~np.isfinite(allrows['T'])).sum()),warning_records=int((allrows.warning_count>0).sum()),
                                  mean_T=float(T.mean()),sd_T=float(T.std(ddof=1)),mean_T_mcse=float(T.std(ddof=1)/np.sqrt(n)),
                                  p_T_lt_minus1_96=float((T < -1.96).mean()),p_T_gt_1_96=float((T>1.96).mean()),
                                  p_abs_T_le_1_96=float((abs(T)<=1.96).mean()),coverage_at_truth=coverage,
                                  accepted_count=k,rejected_count=n-k,coverage_mcse=float(np.sqrt(coverage*(1-coverage)/n)),
                                  coverage_wilson95_lower=lo,coverage_wilson95_upper=hi,
                                  critical_value_difference_count=int(((abs(T)<=1.96)!=valid.accepted.astype(bool)).sum())))
        p=d[(d.scenario==scenario)&(d.status_R=='ok')].copy()
        diff=p.T_difference_R_minus_Python;difference=p.accepted_R.astype(int)-p.accepted_Python.astype(int)
        se=float(difference.std(ddof=1)/np.sqrt(len(p)));mean=float(difference.mean())
        paired.append(dict(scenario=scenario,paired_valid=len(p),mean_T_difference=float(diff.mean()),
                           sd_T_difference=float(diff.std(ddof=1)),mean_T_difference_mcse=float(diff.std(ddof=1)/np.sqrt(len(p))),
                           rmse_T_difference=rms(diff),max_abs_T_difference=float(abs(diff).max()),
                           abs_T_difference_q50=float(abs(diff).quantile(.5)),abs_T_difference_q95=float(abs(diff).quantile(.95)),
                           abs_T_difference_q99=float(abs(diff).quantile(.99)),abs_T_difference_gt_01=int((abs(diff)>.1).sum()),
                           abs_T_difference_gt_05=int((abs(diff)>.5).sum()),abs_T_difference_gt_1=int((abs(diff)>1).sum()),
                           T_correlation=float(p[['T_R','T_Python']].corr().iloc[0,1]),
                           decision_disagreements=int(p.decision_disagrees.sum()),
                           Python_reject_R_accept=int((p.accepted_R&~p.accepted_Python).sum()),
                           Python_accept_R_reject=int((~p.accepted_R&p.accepted_Python).sum()),
                           coverage_difference_R_minus_Python=mean,coverage_difference_mcse=se,
                           coverage_difference_normal95_lower=mean-norm.ppf(.975)*se,
                           coverage_difference_normal95_upper=mean+norm.ppf(.975)*se))
    summary=pd.DataFrame(summaries);pair=pd.DataFrame(paired)
    summary.to_csv(ROOT/'summary.csv',index=False,float_format='%.17g')
    pair.to_csv(ROOT/'paired_comparison.csv',index=False,float_format='%.17g')
    d[d.decision_disagrees==True].to_csv(ROOT/'decision_flips.csv',index=False,float_format='%.17g')
    d.assign(abs_delta=abs(d.T_difference_R_minus_Python)).sort_values('abs_delta',ascending=False).head(30).to_csv(ROOT/'largest_T_differences.csv',index=False,float_format='%.17g')
    manifest=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
    for path,digest in manifest['source_files_sha256'].items():
        assert sha(pathlib.Path(path))==digest,'Original file changed: '+path
    assert sha(ROOT/'reference'/'dri_functions.r')==manifest['author_sha256']
    evidence=dict(planned=3000,attempted=len(d),valid=int((d.status_R=='ok').sum()),failed=int((d.status_R!='ok').sum()),
                  warning_records=int((d.warning_count_R>0).sum()),source_files_unchanged=len(manifest['source_files_sha256']),
                  author_source_unchanged=True,prior_30_matched=True,
                  prior_30_max_author_difference=max(x['max_abs_author_output_difference'] for x in prior_checks),
                  prior_30_max_T_difference=max(abs(x['T_difference']) for x in prior_checks),
                  no_new_DGP_or_HGBR=True,score_saved_scope='per-replication mean, SD, T, p; direct author function does not return per-observation scores')
    (ROOT/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    make_plots(s,summary)
    print(summary[['scenario','method','valid','mean_T','sd_T','coverage_at_truth']].to_string(index=False))
    print(pair.to_string(index=False));print(json.dumps(evidence,indent=2))
def make_plots(stats,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'A':'#8a97a6','B':'#bd9b6a','C_Python':'#4779ac','C_R':'#c45142'}
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    x=np.arange(3);width=.18
    for j,method in enumerate(colors):
        values=summary[summary.method==method].set_index('scenario').loc[SCENARIOS]
        y=values.coverage_at_truth.to_numpy();low=values.coverage_wilson95_lower.to_numpy();high=values.coverage_wilson95_upper.to_numpy()
        ax.bar(x+(j-1.5)*width,y,width,color=colors[method],label=method,yerr=np.vstack([y-low,high-y]),capsize=2)
    ax.axhline(.95,color='#333333',ls='--',lw=1,label='Nominal 95%')
    ax.set(xticks=x,xticklabels=[LABELS[v] for v in SCENARIOS],ylim=(0,1.02),ylabel='Coverage at theta = 1',title='Saved datasets: n = 1000, 1000 replications per scenario')
    ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,1.02),fontsize=8)
    fig.savefig(ROOT/'coverage_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,scenario in zip(axes,SCENARIOS):
        z=np.linspace(-4,7,500);ax.plot(z,norm.pdf(z),color='black',ls=':',label='N(0,1)')
        for method in ['C_Python','C_R']:
            vals=stats[(stats.scenario==scenario)&(stats.method==method)&(stats.status=='ok')]['T']
            ax.hist(vals,bins=np.linspace(-4,7,56),density=True,histtype='step',lw=1.6,color=colors[method],label=method)
        ax.axvline(-1.96,color='gray',ls='--',lw=.8);ax.axvline(1.96,color='gray',ls='--',lw=.8)
        ax.set(title=LABELS[scenario],xlabel='Standardized score T',ylabel='Density')
    axes[0].legend(fontsize=8);fig.savefig(ROOT/'T_distributions.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()

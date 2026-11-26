"""Post-run independent numerical audit. No fitting, no Monte Carlo execution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from data_access import ROOT, LEGACY, sha, dump
from reporting import summary_table, make_figure, write_report

def verify():
    rawpath=ROOT/'replication_results.csv'
    original_raw_hash=sha(rawpath)
    raw=pd.read_csv(rawpath)
    summary=pd.read_csv(ROOT/'summary.csv')
    config=json.loads((ROOT/'run_config.json').read_text(encoding='utf-8'))
    inputs=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
    cfg=config['original_config']
    assert len(raw)==2400 and len(summary)==12 and raw.status.eq('ok').all()
    assert raw.groupby(['scenario','method']).size().eq(200).all()
    max_var_error=0.;max_score_error=0.;max_tau_error=0.
    for item in inputs:
        stem=f"{item['scenario']}_{item['rep_id']:04d}"
        with np.load(item['array_path']) as a:
            d=a['d_res'];y=a['y_res']
        ols=(d@y)/(d@d)
        olsr=y-ols*d
        scale=1.4826*np.median(abs(olsr-np.median(olsr)))
        rows=raw[(raw.scenario==item['scenario'])&(raw.rep_id==item['rep_id'])].set_index('method')
        with np.load(ROOT/'arrays'/(stem+'.npz')) as saved:
            for method in ['A','B','C','D']:
                row=rows.loc[method];n=len(d)
                intercept=method in ['C','D']
                z=np.column_stack([np.ones(n),d]) if intercept else d[:,None]
                beta=np.array([row.intercept,row.theta_hat]) if intercept else np.array([row.theta_hat])
                residual=y-z@beta
                np.testing.assert_allclose(residual,saved[method+'_residual'],atol=2e-13)
                if method=='A':
                    expected_se=np.std(d*(y-ols*d)/np.mean(d*d),ddof=1)/np.sqrt(n)
                    np.testing.assert_allclose([row.theta_hat,row.se],[ols,expected_se],atol=1e-13)
                else:
                    if method=='B': tm=tp=1.345*scale
                    elif method=='C': tm,tp=2*scale,1.5*scale
                    else: tm=tp=scale*np.sqrt(n/np.log(n))
                    max_tau_error=max(max_tau_error,abs(row.tau_minus-tm),abs(row.tau_plus-tp))
                    score=np.minimum(tp,np.maximum(-tm,residual))
                    u=z*score[:,None]
                    A=np.einsum('ni,nj,n->ij',z,z,((residual>-tm)&(residual<tp)).astype(float))/n
                    center=u-np.mean(u,axis=0)
                    transformed=np.linalg.solve(A,center.T).T
                    covariance=transformed.T@transformed/(n*n)
                    max_var_error=max(max_var_error,float(np.max(abs(covariance-saved[method+'_covariance']))))
                    max_score_error=max(max_score_error,float(np.max(abs(u.mean(axis=0)))))
                    np.testing.assert_allclose(covariance,saved[method+'_covariance'],atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(row.se,np.sqrt(covariance[-1,-1]),atol=1e-12)
                    np.testing.assert_allclose(row.clipping_rate,np.mean((residual<-tm)|(residual>tp)),atol=1e-12)
                np.testing.assert_allclose(row['T'],(row.theta_hat-1)/row.se,atol=1e-12)
                assert bool(row.covered)==bool(abs(row['T'])<=1.96)
    for _, row in summary.iterrows():
        g=raw[(raw.scenario==row.scenario)&(raw.method==row.method)]
        a=raw[(raw.scenario==row.scenario)&(raw.method=='A')].set_index('rep_id')
        paired=g.set_index('rep_id').covered.astype(int)-a.covered.astype(int)
        np.testing.assert_allclose(row.coverage,g.covered.mean(),atol=1e-14)
        np.testing.assert_allclose(row.paired_coverage_gain_vs_A,paired.mean(),atol=1e-14)
        np.testing.assert_allclose(row.paired_gain_mcse,paired.std(ddof=1)/np.sqrt(200),atol=1e-14)
        np.testing.assert_allclose([row.mean_T,row.sd_T],[g['T'].mean(),g['T'].std(ddof=1)],atol=1e-13)
    # Refresh presentation only; all estimator/results files remain unchanged.
    make_figure(raw,summary,ROOT/'coverage_comparison.png')
    validation=json.loads((ROOT/'validation'/'numerical_validation.json').read_text(encoding='utf-8'))
    integrity=json.loads((ROOT/'integrity_check.json').read_text(encoding='utf-8'))
    write_report(ROOT,raw,summary,validation['checks'],validation['oracle'],config,integrity)
    assert sha(rawpath)==original_raw_hash
    unchanged_estimators={}
    for name in ['stage2_losses.py','stage2_estimators.py','data_access.py','test_stage2.py','run_pilot.py','legacy_core.py']:
        unchanged_estimators[name]=sha(ROOT/name)==config['source_hashes'][str(ROOT/name)]
    assert all(unchanged_estimators.values())
    dump(ROOT/'delivery_validation.json',dict(passed=True,records=2400,datasets=600,
         independent_covariance_maxdiff=max_var_error,max_equation_error=max_score_error,
         max_tau_difference=max_tau_error,raw_results_unchanged=True,
         estimator_source_unchanged=unchanged_estimators,
         reporting_note='After pilot: fix figure title/legend overlap and add explicit report conclusion; no estimator/result changes.',
         reporting_original_sha=sha(ROOT/'audit'/'reporting_at_pilot.py.txt'),
         reporting_current_sha=sha(ROOT/'reporting.py'),verification_code_sha=sha(Path(__file__))))
    print('Independent audit passed for all 2400 estimates; only presentation refreshed.')

if __name__=='__main__':verify()

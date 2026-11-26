"""Authorized R=200 pilot; reuses saved OOF arrays, never calls learner.fit."""
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime
import argparse, ast, json, time, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits
from data_access import ROOT, ORIGINAL_NB, LEGACY, METHODS, sha, dump, preflight, load_replication
from stage2_estimators import fit_method
from test_stage2 import run_tests
from reporting import summary_table, make_figure, write_report

DEFAULT_SETTINGS=dict(b_kappa=1.345,c_minus=2.,c_plus=1.5,d_kappa=1.)


def one_replication(cfg, scene, rep, output, settings):
    # Integrity errors stop the run: do not silently replace data or seed.
    d,y,audit,unused_oracle=load_replication(cfg,scene,rep)
    rows=[];arrays={}
    for method in METHODS:
        row=dict(scenario=scene['name'],rep_id=rep,seed=audit['seed'],crossfit_seed=audit['crossfit_seed'],
                 fold_hash=audit['fold_hash'],array_sha256=audit['array_sha256'],method=method,
                 n=len(d),p=cfg['p'],status='failed',converged=False,optimizer_success=False,
                 theta_hat=np.nan,se=np.nan,T=np.nan,covered=np.nan,near_knots=0,
                 legacy_theta_difference=np.nan,legacy_se_difference=np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                fit,obs=fit_method(d,y,method,**settings)
                row.update(fit)
                # Evaluation ONLY: truth never enters estimator or thresholds.
                t=(fit['theta_hat']-cfg['theta0'])/fit['se']
                lo=fit['theta_hat']-1.96*fit['se'];hi=fit['theta_hat']+1.96*fit['se']
                if not np.isfinite([t,lo,hi]).all(): raise ArithmeticError('Nonfinite evaluation')
                row.update(status='ok',T=float(t),CI_low=lo,CI_high=hi,ci_length=hi-lo,
                           covered=bool(lo<=cfg['theta0']<=hi),error='')
                if method=='A':
                    row['legacy_theta_difference']=fit['theta_hat']-audit['original_theta']
                    row['legacy_se_difference']=fit['se']-audit['original_se']
                    if max(abs(row['legacy_theta_difference']),abs(row['legacy_se_difference']))>1e-12:
                        raise RuntimeError('Original A changed')
                arrays.update({method+'_'+k:v for k,v in obs.items()})
            except Exception as error:
                row.update(status='failed',covered=np.nan,T=np.nan,
                           error=type(error).__name__+': '+str(error))
        row['warnings']=json.dumps([str(w.message) for w in caught],ensure_ascii=False)
        rows.append(row)
    stem=f"{scene['name']}_{rep:04d}"
    np.savez_compressed(output/'arrays'/(stem+'.npz'),**arrays)
    dump(output/'tasks'/(stem+'.json'),dict(rows=rows,input=audit))
    return rows,audit


def smoke_gate(frame):
    if len(frame)!=24 or (frame.status!='ok').any():
        raise RuntimeError('Smoke incomplete/failed: inspect tasks before continuing')
    assert frame.groupby(['scenario','method']).size().eq(2).all()
    np.testing.assert_allclose(frame['T'],(frame.theta_hat-1.)/frame.se,atol=1e-13)
    assert np.array_equal(frame.covered.astype(bool).values,(abs(frame['T'])<=1.96).values)
    np.testing.assert_allclose(frame.ci_length,3.92*frame.se,atol=1e-13)
    assert frame.score_norm.max()<=1e-9
    assert np.isfinite(frame[['theta_hat','se','T','ci_length']]).all().all()
    return dict(passed=True,datasets=6,method_records=24,max_equation_error=float(frame.score_norm.max()),
                legacy_theta_maxdiff=float(frame.legacy_theta_difference.abs().max()),
                legacy_se_maxdiff=float(frame.legacy_se_difference.abs().max()))


def locations():
    result=[]
    for p in ROOT.glob('*.py'):
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            if isinstance(node,ast.FunctionDef):result.append(dict(file=str(p),function=node.name,line=node.lineno))
    return result


def run(stage='pilot', settings=None):
    if stage not in ['smoke','pilot']:raise ValueError('stage must be smoke or pilot')
    settings=dict(DEFAULT_SETTINGS if settings is None else settings)
    if settings!=DEFAULT_SETTINGS:
        raise ValueError('Pilot settings fixed. Sensitivity API is fit_method; separate explicit experiment required.')
    planned=200 if stage=='pilot' else 2
    started=time.perf_counter()
    cfg,info=preflight()
    output=ROOT if not (ROOT/'run_config.json').exists() else ROOT/'reruns'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=True)
    for sub in ['arrays','tasks','validation']:(output/sub).mkdir(exist_ok=True)
    sourcehash={str(p):sha(p) for p in ROOT.glob('*.py')}
    config=dict(stage=stage,planned_replications=planned,original_config=cfg,settings=settings,
                rep_ids=list(range(planned)),source=info,source_hashes=sourcehash,
                first_stage='Read existing A predictions/residuals; zero learner fits',
                adaptive_rule='kappa * MAD_OLS * sqrt(n / natural_log(n))',
                intervals='Wald +/- 1.96 SE; new methods centered sandwich divisor n',
                output=str(output),started_local=datetime.now().isoformat())
    dump(output/'run_config.json',config)
    dump(output/'source_locations.json',locations())
    print('Numerical validation + Oracle diagnostics (existing first 20 datasets per scene)...',flush=True)
    with threadpool_limits(limits=1):
        checks,oracle=run_tests(output/'validation')
    print('Validation passed. Smoke: first 2 replications per scene.',flush=True)
    rows=[];audits=[]
    bar=tqdm(total=3*planned,desc='Stage-2 datasets',unit='dataset')
    def add(rep,scene):
        with threadpool_limits(limits=1): rr,aa=one_replication(cfg,scene,rep,output,settings)
        rows.extend(rr);audits.append(aa)
        bar.set_postfix_str(f"{scene['name']}, rep={rep}",refresh=False);bar.update(1)
    try:
        for rep in range(2):
            for scene in cfg['scenarios']:add(rep,scene)
        smoke=pd.DataFrame(rows)
        smoke.to_csv(output/'replication_results.csv',index=False,encoding='utf-8-sig')
        sg=smoke_gate(smoke)
        dump(output/'validation'/'smoke_validation.json',sg)
        # Validate table, figure and file serialization before full pilot.
        st=summary_table(smoke,planned)
        st.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
        make_figure(smoke,st,output/'coverage_comparison.png')
        assert len(pd.read_csv(output/'summary.csv'))==12
        assert (output/'coverage_comparison.png').stat().st_size>10000
        print('Smoke passed: all four methods and output files verified.',flush=True)
        if stage=='pilot':
            for end in range(20,201,20):
                begin=2 if end==20 else end-20
                for rep in range(begin,end):
                    for scene in cfg['scenarios']:add(rep,scene)
                raw=pd.DataFrame(rows);summary=summary_table(raw,planned)
                raw.to_csv(output/'replication_results.csv',index=False,encoding='utf-8-sig')
                summary.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
                message=[]
                for scene in cfg['scenarios']:
                    s=summary[summary.scenario==scene['name']]
                    message.append(scene['name']+' '+', '.join(f'{r.method}={r.coverage:.1%}' for _,r in s.iterrows()))
                print(f'R={end}/200 per scene | cumulative coverage: '+ ' | '.join(message),flush=True)
    finally:
        bar.close()
        pd.DataFrame(rows).to_csv(output/'replication_results.csv',index=False,encoding='utf-8-sig')
        dump(output/'input_manifest.json',audits)
    raw=pd.DataFrame(rows)
    assert len(raw)==3*planned*4
    assert not raw.duplicated(['scenario','rep_id','method']).any()
    # Read-only sources checked again after ALL computation.
    changed=[]
    for item in audits:
        for key in ['array','task']:
            if sha(item[key+'_path'])!=item[key+'_sha256']:changed.append(item[key+'_path'])
    if sha(ORIGINAL_NB)!=info['notebook_sha256']:changed.append(str(ORIGINAL_NB))
    if sha(LEGACY/'config.json')!=info['legacy_config_sha256']:changed.append(str(LEGACY/'config.json'))
    for p,h in sourcehash.items():
        if sha(p)!=h:raise RuntimeError('Implementation changed during pilot: '+p)
    integrity=dict(unchanged=not changed,changed_files=changed,checked_files=2*len(audits)+2,
                   max_residual_difference=max(max(a['d_res_max_difference'],a['y_res_max_difference']) for a in audits),
                   learner_fit_calls=0,seconds=time.perf_counter()-started)
    dump(output/'integrity_check.json',integrity)
    if changed:raise RuntimeError('Original inputs changed during computation')
    summary=summary_table(raw,planned)
    summary.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
    make_figure(raw,summary,output/'coverage_comparison.png')
    write_report(output,raw,summary,checks,oracle,config,integrity)
    dump(output/'completion.json',dict(completed=True,stage=stage,datasets=3*planned,rows=len(raw),
                                      failed=int((raw.status!='ok').sum()),seconds=time.perf_counter()-started,
                                      completed_local=datetime.now().isoformat()))
    dump(ROOT/'latest_run.json',dict(output=str(output),stage=stage))
    print(f'Completed {stage}. Results: {output}',flush=True)
    return output


def load_or_run(recompute=False):
    marker=ROOT/'latest_run.json'
    if not recompute and marker.exists():
        info=json.loads(marker.read_text(encoding='utf-8'))
        out=Path(info['output'])
        completion=json.loads((out/'completion.json').read_text(encoding='utf-8'))
        if info['stage']=='pilot' and completion['completed']:
            print('Loading completed pilot; no computation repeated:',out)
            return out
    return run('pilot')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['smoke','pilot'],default='smoke')
    args=parser.parse_args()
    run(args.stage)

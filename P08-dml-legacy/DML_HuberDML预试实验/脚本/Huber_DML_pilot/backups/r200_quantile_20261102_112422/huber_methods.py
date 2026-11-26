"""Outcome-only Huber comparison; original DGP, initial A and IF-SE are imported."""
import hashlib,json,pathlib,time,warnings
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from threadpoolctl import threadpool_limits
from legacy_core import make_data,initial_a,fit_mean_dml,nuisance_functions,lognormal_nuisance_functions

GBR_COMMON=dict(learning_rate=.05,n_estimators=100,max_leaf_nodes=15,min_samples_leaf=20,
                max_depth=None,subsample=1.0)
METHODS=['A','B','C']

def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [json_safe(v) for v in value]
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def finite(x):
    if not np.isfinite(x).all():raise FloatingPointError('Nonfinite array or statistic')

def outcome_gbr(X,Y,folds,cfseed,loss):
    prediction=np.full(len(Y),np.nan);seen=np.zeros(len(Y),dtype=int)
    for k,(train,test) in enumerate(folds):
        if np.intersect1d(train,test).size:raise ValueError('Fold overlap')
        model=GradientBoostingRegressor(loss=loss,alpha=.90,random_state=cfseed+100*k+2,**GBR_COMMON)
        model.fit(X[train],Y[train])
        prediction[test]=model.predict(X[test]);seen[test]+=1
    if not np.all(seen==1):raise ValueError('OOF observations not covered exactly once')
    finite(prediction)
    return prediction

def run_replication(config,scenario,rep_id,output,legacy_run):
    output=pathlib.Path(output);started=time.perf_counter();rows=[];captured=[]
    base=dict(scenario=scenario['name'],dgp=config['dgp'],p=config['p'],n=config['n'],rep_id=rep_id)
    stem=f"{scenario['name']}_{rep_id:04d}";arrays={}
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            (X,D,Y,epsilon),folds,seed=make_data(config,scenario,rep_id)
            cfseed=seed+config['fold_seed_offset'];fold_ids=np.full(len(D),-1,dtype=np.int64)
            seen=np.zeros(len(D),dtype=int)
            for k,(train,test) in enumerate(folds):
                assert len(np.intersect1d(train,test))==0
                assert len(train)+len(test)==len(D)
                fold_ids[test]=k;seen[test]+=1
            assert np.all(seen==1)
            base.update(seed=seed,crossfit_seed=cfseed,fold_hash=hashlib.sha256(fold_ids.tobytes()).hexdigest(),
                        epsilon_variance=float(np.var(epsilon,ddof=1)),epsilon_max_abs=float(abs(epsilon).max()))
            finite(np.array([base['epsilon_variance'],base['epsilon_max_abs']]))
            m_hat,ell_A=initial_a(X,D,Y,folds,cfseed) # treatment fitted once, shared below
            finite(np.r_[m_hat,ell_A,D,Y])
            oracle=lognormal_nuisance_functions if scenario['error_family']=='lognormal' else nuisance_functions
            m0,g0=oracle(X,config['dgp']);ell0=config['theta0']*m0+g0
            base['treatment_rmse']=float(np.sqrt(np.mean((m_hat-m0)**2)))
            base['treatment_prediction_hash']=hashlib.sha256(m_hat.tobytes()).hexdigest()
            arrays.update(X=X,D=D,Y=Y,epsilon=epsilon,fold_ids=fold_ids,m_hat=m_hat,m0=m0,ell0=ell0)
        except Exception as error:
            rows=[dict(base,method=m,status='failed',error=f'{type(error).__name__}: {error}') for m in METHODS]
        if not rows:
            for method in METHODS:
                params=(config['hgbr_params_seed_zero'].copy() if method=='A' else
                        GradientBoostingRegressor(loss='squared_error' if method=='B' else 'huber',alpha=.90,**GBR_COMMON).get_params())
                params['random_state']='crossfit_seed + 100*fold + 2'
                row=dict(base,method=method,learner_params=json.dumps(params,sort_keys=True))
                try:
                    ell=ell_A if method=='A' else outcome_gbr(X,Y,folds,cfseed,'squared_error' if method=='B' else 'huber')
                    theta,se=fit_mean_dml(D-m_hat,Y-ell) # exact legacy second stage
                    finite(np.array([theta,se]));assert se>0,'Nonpositive SE'
                    low,high=theta-1.96*se,theta+1.96*se
                    error=ell-ell0;quantiles=np.quantile(error,[.01,.05,.25,.5,.75,.95,.99])
                    row.update(status='ok',theta_hat=theta,se=se,CI_low=low,CI_high=high,
                               covered=bool(low<=config['theta0']<=high),ci_length=high-low,
                               t_stat=(theta-config['theta0'])/se,
                               outcome_rmse=float(np.sqrt(np.mean(error**2))),outcome_mae=float(np.mean(abs(error))),
                               outcome_error_mean=float(error.mean()),outcome_error_median=float(np.median(error)))
                    row.update({f'outcome_error_q{q}':float(v) for q,v in zip(['01','05','25','50','75','95','99'],quantiles)})
                    finite(np.array([row['t_stat'],row['outcome_rmse'],row['outcome_mae'],low,high,*quantiles]))
                    arrays['ell_'+method]=ell
                    if method=='A':
                        task_path=pathlib.Path(legacy_run)/'tasks'/f'{stem}.json'
                        task=json.loads(task_path.read_text(encoding='utf-8'))
                        old=next(r for r in task['rows'] if r['method']=='original_mean_dml')
                        array_path=pathlib.Path(legacy_run)/task['arrays_file']
                        assert hashlib.sha256(array_path.read_bytes()).hexdigest()==task['arrays_sha256']
                        with np.load(array_path,allow_pickle=False) as saved:
                            md=float(np.max(abs(m_hat-saved['g_hat'])));ld=float(np.max(abs(ell-saved['ell_hat'])))
                            same_folds=np.array_equal(fold_ids,saved['fold_ids'])
                        row.update(legacy_m_maxdiff=md,legacy_ell_maxdiff=ld,
                                   legacy_theta_difference=theta-old['theta_hat'],legacy_se_difference=se-old['theta_se'])
                        row['legacy_match']=bool(same_folds and md<1e-12 and ld<1e-12 and
                                                 abs(theta-old['theta_hat'])<1e-12 and abs(se-old['theta_se'])<1e-12)
                        if not row['legacy_match']:raise ValueError('A differs from saved baseline: inspect legacy_* fields')
                except Exception as error:
                    row.update(status='failed',error=f'{type(error).__name__}: {error}',covered=None)
                rows.append(row)
    warning_text=[str(w.message) for w in captured]
    for row in rows:row.update(warnings=json.dumps(warning_text,ensure_ascii=False),elapsed_seconds=time.perf_counter()-started)
    np.savez_compressed(output/'arrays'/f'{stem}.npz',**arrays)
    rows=json_safe(rows)
    result=dict(rows=rows,seed=base.get('seed'),scenario=scenario['name'],rep_id=rep_id)
    path=output/'tasks'/f'{stem}.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return rows

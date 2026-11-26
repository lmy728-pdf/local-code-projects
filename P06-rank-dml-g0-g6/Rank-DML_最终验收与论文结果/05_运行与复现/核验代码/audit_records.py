"""Read-only audit of saved experiments. No generate(), crossfit(), or learner fit calls."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import json,hashlib,io,zipfile,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import beta,chi2
from scipy.special import ndtri,gammaln

OUT=Path(__file__).resolve().parents[2]
PROJECT=OUT.parent
ARCHIVE=Path(r'C:\Users\Lenovo\Documents\ChatGPT\毕设\rank_dml_study')
TMP=OUT/'05_运行与复现/核验中间数据'
METHODS=['LS','H','HT','W','N'];PIPES=['learned','oracle'];Z=ndtri(.975)
STAGES=[('G0','g0_smoke_scalar'),('G0','g0_smoke_vector'),('G0','g0_pilot'),('G1','g1_scalar_n1000'),('G2','g2_vector_n1000'),('G3','g3_n500'),('G4','g4_n2000'),('G5','g5_rf'),('G6A','g6a_tau050'),('G6B','g6b_c0.5'),('G6B','g6b_c2.0')]

def folder_for(stage,batch):
    return ARCHIVE/'runs'/batch if stage in ['G0','G1','G2','G3'] else PROJECT/(stage+'实验')/(stage+'实验_代码')/'_work'/batch

def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj

def save(path,obj):
    assert path.resolve().is_relative_to(OUT.resolve())
    path.write_text(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':')),encoding='utf-8')

def coverage(flags,valid):
    flags=np.asarray(flags,bool)&np.asarray(valid,bool);valid=np.asarray(valid,bool)
    b=len(flags);s=int(flags.sum());p=s/b
    return dict(attempted=b,valid_intervals=int(valid.sum()),coverage=p,coverage_valid_only=s/valid.sum() if valid.any() else np.nan,
                coverage_mcse=np.sqrt(p*(1-p)/b),coverage_lower=beta.ppf(.025,s,b-s+1) if s else 0.,coverage_upper=beta.ppf(.975,s+1,b-s) if s<b else 1.)

def describe(x,prefix):
    a=np.asarray(x,float);v=a[np.isfinite(a)];n=len(v)
    return {prefix+'_mean':v.mean() if n else np.nan,prefix+'_sd':v.std(ddof=1) if n>1 else np.nan,
            prefix+'_mcse':v.std(ddof=1)/np.sqrt(n) if n>1 else np.nan,prefix+'_min':v.min() if n else np.nan,
            prefix+'_median':np.median(v) if n else np.nan,prefix+'_max':v.max() if n else np.nan,prefix+'_valid':n}

def accuracy(est,truth,se,length,score,cov,valid):
    est=np.asarray(est,float);mask=np.isfinite(est);e=est[mask]-truth;sd=np.std(est[mask],ddof=1) if mask.sum()>1 else np.nan
    se=np.asarray(se,float);valid=np.asarray(valid,bool);lens=np.asarray(length,float);scores=np.asarray(score,float)
    return dict(truth=truth,mean_estimate=np.mean(est[mask]),bias=np.mean(e),SD=sd,RMSE=np.sqrt(np.mean(e**2)),MAE=np.mean(abs(e)),
        median_absolute_error=np.median(abs(e)),MeanSE=np.mean(se[valid]) if valid.any() else np.nan,
        SEratio=np.mean(se[valid])/sd if valid.any() else np.nan,mean_length=np.mean(lens[valid]) if valid.any() else np.nan,
        median_length=np.median(lens[valid]) if valid.any() else np.nan,mean_interval_score=np.mean(scores[valid]) if valid.any() else np.nan,
        valid_estimates=int(mask.sum()),metric_population='conditional-on-success',failure_rate=1-valid.mean(),**coverage(cov,valid))

def cell_id(c):return '_'.join(str(c[k]) for k in ('experiment','law','n','d','rho'))

def correlation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.std(a[ok])==0 or np.std(b[ok])==0:return np.nan
    return np.corrcoef(a[ok],b[ok])[0,1]

def audit_cell(task):
    stage,batch,folder,cell,R=task;folder=Path(folder);ident=cell_id(cell);d=cell['d'];n=cell['n']
    paths=sorted((folder/'records').glob(ident+'_*.json'))
    evidence=[];issues=[];datasets=[];cache_paths=[];reuse_checks=0;numeric_replays=0
    def check(condition,message):
        if not condition and message not in issues:issues.append(message)
    check(len(paths)==R,'record_count_mismatch')
    sys.path.insert(0,str(ARCHIVE))
    import core
    for path in paths:
        raw=path.read_bytes();data=json.loads(raw);datasets.append(data)
        evidence.append(dict(path=str(path),kind='record',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime_ns=path.stat().st_mtime_ns))
        check(data['cell']==cell,'cell_metadata_mismatch')
        check(data['stage']==batch,'stage_metadata_mismatch')
        rep=data['replication'];check(path.stem==ident+f'_{rep:04d}','filename_replication_mismatch')
        rows={(r['pipeline'],r['method']):r for r in data['records']}
        check(len(rows)==10 and len(data['records'])==10,'method_pipeline_set_mismatch')
        for pipe in PIPES:
            hashes={r['input_hash'] for (p,m),r in rows.items() if p==pipe and 'input_hash' in r}
            check(len(hashes)==1,'shared_input_hash_mismatch')
        if 'traceback' in data:continue
        cache=Path(data['source_cache']) if 'source_cache' in data else folder/'cache'/(path.stem+'.npz')
        cache_paths.append(str(cache));check(cache.is_file(),'missing_cache')
        if cache.is_file():
            content=cache.read_bytes()
            if stage!='G6B':evidence.append(dict(path=str(cache),kind='cache',bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),mtime_ns=cache.stat().st_mtime_ns))
            with np.load(io.BytesIO(content),allow_pickle=False) as z:
                arrays={k:z[k] for k in z.files}  # Reading every member checks ZIP CRC.
            f=arrays['folds'];check(set(f)==set(range(5)) and np.ptp(np.bincount(f))<=1,'unbalanced_outer_folds')
            for pipe in PIPES:
                suffix='_oracle' if pipe=='oracle' else '';w=arrays['w'+suffix];z=arrays['z'+suffix]
                digest=hashlib.sha256(w.tobytes()+z.tobytes()+f.tobytes()).hexdigest()
                check(all(rows[pipe,m].get('input_hash')==digest for m in METHODS),'cache_input_hash_mismatch')
            check(np.allclose(arrays['scales'],np.array(data['scales'],float),equal_nan=True),'scale_cache_mismatch')
            check(np.allclose(arrays['multipliers'],np.array(data['multipliers'],float),equal_nan=True),'multiplier_cache_mismatch')
            if stage=='G5':
                old=ARCHIVE/'runs'/('g1_scalar_n1000' if d==1 else 'g2_vector_n1000')/'cache'/cache.name
                with np.load(old,allow_pickle=False) as z:
                    check(all(np.array_equal(arrays[k],z[k]) for k in ['folds','w_oracle','z_oracle']),'RF_data_fold_reuse_mismatch')
                reuse_checks+=1
                fits=data.get('forest_fits',[]);check(len(fits)==20,'RF_fit_count_mismatch')
                for fit in fits:
                    expected=[int(core.rng('g5_rf',cell,rep,f'forest_fit_{fit["fit_index"]}_target_{j}').integers(0,2**32-1)) for j in range(d+1)]
                    check(fit['target_seeds']==expected,'RF_seed_mismatch')
            if stage=='G6B':
                source=json.loads(Path(data['source_record']).read_text(encoding='utf-8'))
                baseline={(r['pipeline'],r['method']):r for r in source['records']}
                for pipe in PIPES:
                    for m in ['LS','H','HT']:
                        for key in ['estimate','covariance','coverage','se']:
                            check(rows[pipe,m][key]==baseline[pipe,m][key],'G6B_baseline_reuse_mismatch')
                    check(rows[pipe,'W']['estimate']==baseline[pipe,'W']['estimate'],'G6B_W_point_changed')
                reuse_checks+=1
            # Only saved residuals are used. No generated samples or first-stage fit occurs.
            if rep==0:
                for pipe in PIPES:
                    suffix='_oracle' if pipe=='oracle' else '';w=arrays['w'+suffix];z=arrays['z'+suffix]
                    for m in METHODS:
                        threshold=arrays['scales'][f]*(1.345 if m=='H' else arrays['multipliers'][f]) if m in ['H','HT'] else None
                        point,V,meta,fail=core.estimate(w,z,f,m,threshold,c=data.get('rank_step',1.))
                        check(np.allclose(point,np.array(rows[pipe,m]['estimate'],float),atol=1e-10,rtol=1e-9,equal_nan=True),'cached_point_replay_mismatch')
                        check(np.allclose(V,np.array(rows[pipe,m]['covariance'],float),atol=1e-10,rtol=1e-9,equal_nan=True),'cached_covariance_replay_mismatch')
                        numeric_replays+=1
        for choice in data.get('training_choices',[]):
            check(len(choice['inner_folds'])==n-n//5 and set(choice['inner_folds'])=={0,1,2},'inner_fold_shape_mismatch')
            for j,coordinate in enumerate(choice['coordinates']):
                candidates=coordinate['candidates'];check([c['multiplier'] for c in candidates]==list(core.MULT),'HT_grid_mismatch')
                good=[c for c in candidates if c.get('criterion') is not None and c.get('converged')]
                if good:
                    best=min(good,key=lambda a:(a['criterion'],-a['multiplier']))
                    check(data['multipliers'][choice['fold']][j]==best['multiplier'],'HT_selection_mismatch')
    datasets.sort(key=lambda a:a['replication']);B=len(datasets)
    check([a['replication'] for a in datasets]==list(range(R)),'replication_missing_or_duplicate')
    records=[[{(r['pipeline'],r['method']):r for r in data['records']}[p,m] for p in PIPES for m in METHODS] for data in datasets]
    def arr(key,default=np.nan):return np.array([[r.get(key,default) for r in rr] for rr in records],float)
    est=arr('estimate');V=arr('covariance');truth=np.array(records[0][0]['truth'],float)
    check(np.allclose(truth,1/np.arange(1,d+1)),'truth_mismatch')
    check(all(r['coordinate_order']==list(range(1,d+1)) for rr in records for r in rr),'coordinate_order_mismatch')
    check(np.allclose(V,np.swapaxes(V,-1,-2),atol=1e-12,equal_nan=True),'covariance_asymmetry')
    V=(V+V.swapaxes(-1,-2))/2
    se=np.sqrt(np.diagonal(V,axis1=-2,axis2=-1)/n);valid=np.isfinite(est)&np.isfinite(se)&(se>0)
    cov=valid&(abs(est-truth)<=Z*se);length=2*Z*se;score=length+40*np.maximum(abs(est-truth)-Z*se,0)
    check(np.allclose(se,arr('se'),atol=1e-12,equal_nan=True),'SE_reconstruction_mismatch')
    check(np.array_equal(valid,arr('valid_interval').astype(bool)),'interval_validity_mismatch')
    check(np.array_equal(cov,arr('coverage').astype(bool)),'coverage_reconstruction_mismatch')
    check(np.allclose(length,arr('length'),atol=1e-12,equal_nan=True),'length_reconstruction_mismatch')
    check(np.allclose(score,arr('interval_score'),atol=1e-10,equal_nan=True),'interval_score_mismatch')
    eig=np.full((B,10,d),np.nan);condition=np.full((B,10),np.nan)
    finite=np.isfinite(V).all(axis=(-1,-2));eig[finite]=np.linalg.eigvalsh(V[finite]);condition[finite]=np.linalg.cond(V[finite])
    joint_valid=finite&(eig[:,:,0]>0)&np.isfinite(est).all(-1);joint_cov=np.zeros((B,10),bool);logvol=np.full((B,10),np.nan)
    if d>1 and joint_valid.any():
        L=np.linalg.cholesky(V[joint_valid]);error=(est-truth)[joint_valid];sol=np.linalg.solve(L,error[...,None])[...,0]
        joint_cov[joint_valid]=n*np.sum(sol*sol,axis=-1)<=chi2.ppf(.95,d)
        logvol[joint_valid]=d/2*np.log(np.pi)-gammaln(d/2+1)+d/2*np.log(chi2.ppf(.95,d))+np.log(np.diagonal(L,axis1=-2,axis2=-1)).sum(-1)-d/2*np.log(n)
        check(np.array_equal(joint_valid,arr('joint_valid').astype(bool)),'joint_validity_mismatch')
        check(np.array_equal(joint_cov,arr('joint_coverage').astype(bool)),'joint_coverage_mismatch')
        check(np.allclose(logvol,arr('log_volume'),atol=1e-10,equal_nan=True),'log_volume_mismatch')
        check(np.allclose(eig[:,:,0],arr('min_eigenvalue'),atol=1e-10,equal_nan=True),'eigenvalue_mismatch')
        check(np.allclose(condition,arr('condition'),rtol=1e-9,equal_nan=True),'condition_number_mismatch')
    gamma=np.full((B,10,d),np.nan);convergence=np.full((B,10,d),np.nan);iterations=np.full((B,10,d),np.nan);residual=np.full((B,10,d),np.nan)
    for b,rr in enumerate(records):
        for i,row in enumerate(rr):
            if len(row.get('metadata',[]))==d:
                for j,meta in enumerate(row['metadata']):
                    gamma[b,i,j]=meta.get('gamma',np.nan);convergence[b,i,j]=meta.get('converged',np.nan)
                    iterations[b,i,j]=meta.get('iterations',np.nan);residual[b,i,j]=meta.get('score_residual',np.nan)
    base=dict(stage=stage,batch=batch,**cell,tau=.5 if stage=='G6A' else .2,rank_c=float(batch.split('_c')[1]) if stage=='G6B' else 1.,learner='RF' if stage=='G5' else 'Spline Ridge',role='pilot' if stage=='G0' else ('main' if stage in ['G1','G2','G3','G4'] else 'sensitivity'))
    coords=[];vectors=[];contrasts=[];failure_rows=[];runtime=[];slope_rows=[]
    full=arr('full_pipeline_seconds');second=arr('seconds')
    costs={k:np.array([ds.get(k,np.nan) for ds in datasets],float) for k in ['nuisance_seconds','auxiliary_seconds','tuning_seconds','elapsed']}
    costs['diagnostic_seconds']=np.array([ds.get('diagnostics',{}).get('seconds',np.nan) for ds in datasets])
    con_est=np.full((B,10,2),np.nan);con_se=con_est.copy();con_cov=np.zeros((B,10,2),bool);con_valid=con_cov.copy();con_length=con_est.copy();con_score=con_est.copy()
    if d>1:
        for j,(name,a) in enumerate([('plus',np.ones(d)/np.sqrt(d)),('minus',np.r_[1,-1,np.zeros(d-2)]/np.sqrt(2))]):
            con_est[:,:,j]=est@a;con_se[:,:,j]=np.sqrt(np.einsum('i,abij,j->ab',a,V,a)/n)
            con_valid[:,:,j]=np.isfinite(con_est[:,:,j])&np.isfinite(con_se[:,:,j])&(con_se[:,:,j]>0)
            con_cov[:,:,j]=con_valid[:,:,j]&(abs(con_est[:,:,j]-a@truth)<=Z*con_se[:,:,j])
            con_length[:,:,j]=2*Z*con_se[:,:,j];con_score[:,:,j]=con_length[:,:,j]+40*np.maximum(abs(con_est[:,:,j]-a@truth)-Z*con_se[:,:,j],0)
            for attr,array in [('estimate',con_est),('se',con_se),('length',con_length),('interval_score',con_score)]:
                saved=np.array([[r['contrasts'][name][attr] for r in rr] for rr in records],float)
                check(np.allclose(saved,array[:,:,j],atol=1e-10,equal_nan=True),'contrast_'+attr+'_mismatch')
            check(np.array_equal(con_cov[:,:,j],np.array([[r['contrasts'][name]['coverage'] for r in rr] for rr in records],bool)),'contrast_coverage_mismatch')
    for i,(pipe,method) in enumerate((p,m) for p in PIPES for m in METHODS):
        bbase=dict(**base,pipeline=pipe,method=method)
        for j in range(d):
            coords.append(dict(**bbase,coordinate=j+1,**accuracy(est[:,i,j],truth[j],se[:,i,j],length[:,i,j],score[:,i,j],cov[:,i,j],valid[:,i,j])))
            slope_rows.append(dict(**bbase,coordinate=j+1,**describe(gamma[:,i,j],'gamma'),**describe(se[:,i,j],'SE'),**describe(eig[:,i,0],'cov_min_eigenvalue'),**describe(condition[:,i],'cov_condition')))
        ok=np.isfinite(est[:,i]).all(-1);e=est[ok,i]-truth
        if d>1:
            vectors.append(dict(**bbase,valid_vector_estimates=int(ok.sum()),vector_bias_norm=np.linalg.norm(e.mean(0)),vector_RMSE=np.sqrt(np.mean(np.sum(e*e,axis=-1))),metric_population='conditional-on-success',
               mean_volume=np.mean(np.exp(logvol[joint_valid[:,i],i])),median_log_volume=np.median(logvol[joint_valid[:,i],i]),
               all_marginal_coverage=cov[:,i].all(-1).mean(),joint_failure_count=int((~joint_valid[:,i]).sum()),
               **coverage(joint_cov[:,i],joint_valid[:,i]),**describe(eig[:,i,0],'cov_min_eigenvalue'),**describe(condition[:,i],'cov_condition')))
            for j,name in enumerate(['plus','minus']):
                a=np.ones(d)/np.sqrt(d) if name=='plus' else np.r_[1,-1,np.zeros(d-2)]/np.sqrt(2)
                contrasts.append(dict(**bbase,contrast=name,**accuracy(con_est[:,i,j],a@truth,con_se[:,i,j],con_length[:,i,j],con_score[:,i,j],con_cov[:,i,j],con_valid[:,i,j])))
        native={}
        for rr in records:
            for fail in rr[i].get('failures',[]):native[fail['type']]=native.get(fail['type'],0)+1
        failures=dict(nuisance_or_dataset=sum('traceback' in ds for ds in datasets),invalid_huber_scale=sum(np.any(~np.isfinite(np.array(ds.get('scales',[[np.nan]]),float))) or np.any(np.array(ds.get('scales',[[np.nan]]),float)<=0) for ds in datasets) if method in ['H','HT'] else 0,
            optimization=int(np.any(convergence[:,i]==0,axis=-1).sum()) if method in ['LS','H','HT'] else 0,
            nonpositive_or_nonfinite_rank_slope=int(np.any(~np.isfinite(gamma[:,i])|(gamma[:,i]<=0),axis=-1).sum()) if method in ['W','N'] else 0,
            nonfinite_SE=int(np.any(~np.isfinite(se[:,i]),axis=-1).sum()),nonpositive_definite_covariance=int((~(finite[:,i]&(eig[:,i,0]>0))).sum()) if d>1 else 0,
            invalid_estimate=int((~ok).sum()),invalid_any_interval=int((~valid[:,i].all(-1)).sum()),joint_interval=int((~joint_valid[:,i]).sum()) if d>1 else 0,
            any_recorded_failure=sum(bool(rr[i].get('failures')) for rr in records))
        for category,count in failures.items():
            applicable=not ((category in ['invalid_huber_scale'] and method not in ['H','HT']) or (category=='optimization' and method in ['W','N']) or (category=='nonpositive_or_nonfinite_rank_slope' and method not in ['W','N']) or (category in ['nonpositive_definite_covariance','joint_interval'] and d==1))
            failure_rows.append(dict(**bbase,category=category,applicable=applicable,count=count if applicable else None,denominator=B if applicable else None,rate=count/B if applicable else None,native_failure_types=json.dumps(native),**describe(iterations[:,i],'optimizer_iterations'),**describe(residual[:,i],'score_residual')))
        expected=second[:,i]+costs['nuisance_seconds']+(costs['auxiliary_seconds'] if method in ['H','HT'] else 0)+(costs['tuning_seconds'] if method=='HT' else 0)
        check(np.allclose(expected,full[:,i],atol=1e-8,equal_nan=True),'full_pipeline_timing_identity_mismatch')
        runtime.append(dict(**bbase,attempted=B,**{k+'_mean':np.nanmean(v) for k,v in costs.items()},second_stage_seconds_mean=np.nanmean(second[:,i]),reported_full_pipeline_seconds_mean=np.nanmean(full[:,i]),
            current_run_shared_nuisance_seconds_mean=0. if stage=='G6B' else np.nanmean(costs['nuisance_seconds']),
            current_run_second_stage_seconds_mean=np.nanmean(second[:,i]) if stage!='G6B' or method in ['W','N'] else 0.,
            timing_scope='cache_replay; baseline timing inherited; elapsed is original baseline elapsed' if stage=='G6B' else 'component timings per dataset; not exclusive-machine benchmark',
            oracle_cost_scope='training charges describe fixed learned choices, not a separately trained oracle' if pipe=='oracle' else 'feasible pipeline',
            counterfactual_full_pipeline=stage=='G6B'))
    diag=np.full((B,5,d,6),np.nan)
    for b,ds in enumerate(datasets):
        for k,p in enumerate(ds.get('diagnostics',{}).get('folds',[])):
            for j in range(d):diag[b,k,j]=[p['mse_m'],p['mse_ell'][j],p['alignment_variance'][j],p['var_b'],p['cov_b_a'][j],p['weight']]
    weighted=(diag[:,:,:,:5]*diag[:,:,:,5:6]).sum(axis=1);displacement=weighted[:,:,4]/(1+weighted[:,:,3])
    saved=np.array([ds.get('diagnostics',{}).get('LS_population_shift',[np.nan]*d) for ds in datasets],float)
    check(np.allclose(displacement,saved,atol=1e-12,equal_nan=True),'population_displacement_formula_mismatch')
    diagnostics=[];fold_diagnostics=[]
    names=['mse_m','mse_ell','alignment_variance','var_b','cov_b_A']
    for j in range(d):
        ds=dict(**base,coordinate=j+1,attempted=B,diagnostic_draws_per_replication=10000,
                diagnostic_source='inherited baseline diagnostic' if stage=='G6B' else 'saved fresh-draw fold aggregates',
                **describe(displacement[:,j],'LS_population_displacement'))
        for v,name in enumerate(names):ds.update(describe(weighted[:,j,v],name))
        diagnostics.append(ds)
        for k in range(5):
            fd=dict(**base,fold=k,coordinate=j+1,attempted=B,weight=float(np.nanmean(diag[:,k,j,5])))
            for v,name in enumerate(names):fd.update(describe(diag[:,k,j,v],name))
            fold_diagnostics.append(fd)
    relations=[]
    for i,(pipe,method) in enumerate((p,m) for p in PIPES for m in METHODS):
        for j in range(d):
            relations.append(dict(**base,pipeline=pipe,method=method,coordinate=j+1,attempted=B,
               corr_mse_m_squared_error=correlation(weighted[:,j,0],(est[:,i,j]-truth[j])**2),
               corr_mse_ell_squared_error=correlation(weighted[:,j,1],(est[:,i,j]-truth[j])**2),
               corr_alignment_variance_squared_error=correlation(weighted[:,j,2],(est[:,i,j]-truth[j])**2),
               corr_displacement_estimation_error=correlation(displacement[:,j],est[:,i,j]-truth[j]),
               mean_displacement=np.nanmean(displacement[:,j]),observed_bias=np.nanmean(est[:,i,j]-truth[j]),
               descriptive_only=True))
    stem=batch+'__'+ident;outpath=TMP/(stem+'.npz')
    np.savez_compressed(outpath,replications=np.array([a['replication'] for a in datasets]),estimate=est,covariance=V,se=se,valid=valid,coverage=cov,length=length,interval_score=score,
        truth=truth,joint_valid=joint_valid,joint_coverage=joint_cov,log_volume=logvol,condition=condition,min_eigenvalue=eig[:,:,0],gamma=gamma,
        contrast_estimate=con_est,contrast_se=con_se,contrast_valid=con_valid,contrast_coverage=con_cov,contrast_length=con_length,contrast_score=con_score,
        nuisance=weighted,displacement=displacement)
    result=dict(base=base,expected=R,completed=B,method_records=B*10,issues=issues,cache_checks=len(cache_paths),reused_data_checks=reuse_checks,
        cached_method_replays=numeric_replays,coordinates=coords,vectors=vectors,contrasts=contrasts,failures=failure_rows,runtime=runtime,slopes=slope_rows,
        diagnostics=diagnostics,fold_diagnostics=fold_diagnostics,diagnostic_relations=relations,source_files=evidence,array_file=str(outpath),source_folder=str(folder))
    save(TMP/(stem+'.json'),result)
    return dict(stage=stage,batch=batch,cell=ident,completed=B,issues=issues)

def main():
    tasks=[];inventory=[]
    for stage,batch in STAGES:
        folder=folder_for(stage,batch);config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        expected_stems={cell_id(c)+f'_{r:04d}' for c in config['cells'] for r in range(config['R'])}
        actual_stems={p.stem for p in (folder/'records').glob('*.json')}
        inventory.append(dict(stage=stage,batch=batch,source_folder=str(folder),expected=len(expected_stems),actual=len(actual_stems),missing=sorted(expected_stems-actual_stems),unexpected=sorted(actual_stems-expected_stems),config=config))
        tasks.extend((stage,batch,str(folder),cell,config['R']) for cell in config['cells'])
    save(TMP/'批次扫描.json',inventory)
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures=[pool.submit(audit_cell,task) for task in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result()
            print(f'{i}/{len(tasks)} {r["batch"]} {r["cell"]}: {r["completed"]}; issues={r["issues"]}',flush=True)

if __name__=='__main__':main()

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse, platform, sys, json, hashlib, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
import scipy
from core import *
ROOT=Path(__file__).resolve().parent

def clean(o):
    if isinstance(o,np.ndarray):return clean(o.tolist())
    if isinstance(o,(np.integer,np.bool_)):return o.item()
    if isinstance(o,(float,np.floating)):return float(o) if np.isfinite(o) else None
    if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [clean(v) for v in o]
    return o

def save(path,o):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as handle:
        handle.write(json.dumps(clean(o),ensure_ascii=False,indent=2));handle.flush();os.fsync(handle.fileno())
    for attempt in range(50):
        try:
            os.replace(temp,path);break
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)

def grid():
    return [dict(experiment=e,law=law,n=n,d=d,rho=rho) for e in ('scalar','vector') for n in (500,1000,2000) for d in ((1,) if e=='scalar' else (2,3)) for rho in ((0,) if e=='scalar' else (0,.5)) for law in LAWS]

def cell_id(c):return '_'.join(str(c[k]) for k in ('experiment','law','n','d','rho'))

def source_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(ROOT.glob('*.py')))).hexdigest()

def g1_preflight():
    frozen=json.loads((ROOT/'reports'/'freeze_manifest.json').read_text(encoding='utf-8'))
    assert frozen['status']=='G0_complete_definitions_frozen_before_main'
    for name in ('core.py','reporting.py'):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==frozen['sources'][name],name+' differs from G0'
    audit=json.loads((ROOT/'reports'/'final_audit.json').read_text(encoding='utf-8'))
    assert len(audit['checks_passed'])==9
    assert sum(a['completed_datasets'] for a in audit['stages'])==7200
    cells=[c for c in grid() if c['experiment']=='scalar' and c['n']==1000]
    assert len(cells)==4
    for cell in cells:
        for stage in ('g0_smoke_scalar','g0_pilot'):
            assert not np.array_equal(rng('g1_scalar_n1000',cell,0,'data').normal(size=8),rng(stage,cell,0,'data').normal(size=8))
    return cells

def worker(task):
    stage,cell,rep,folder=task;folder=Path(folder);ident=cell_id(cell)+f'_{rep:04d}';path=folder/'records'/f'{ident}.json'
    if path.exists():return ident,'cached'
    start=time.perf_counter();n=cell['n'];d=cell['d'];records=[];diagnostics={};choices=[]
    try:
        x,D,Y,wo,zo,true=generate(cell,rng(stage,cell,rep,'data'));f=folds(n,5,rng(stage,cell,rep,'outer'))
        ts=time.perf_counter();res,models=crossfit(x,np.column_stack([D,Y]),f);nuisance_time=time.perf_counter()-ts
        scales,chosen,choices,aux,tuning=training_choices(x,D,Y,f,stage,cell,rep)
        ts=time.perf_counter();fresh=covariates(10000,rng(stage,cell,rep,'diagnostic'));m,ell,_=truth(fresh,d,cell['experiment']);parts=[]
        for k,mod in enumerate(models):
            pred=mod.predict(fresh);b=pred[:,0]-m;q=pred[:,1:]-ell;a=q-b[:,None]*true
            parts.append(dict(weight=np.mean(f==k),mse_m=np.mean(b*b),mse_ell=np.mean(q*q,0),alignment_variance=np.var(a,axis=0),var_b=np.var(b),cov_b_a=np.mean((b-b.mean())[:,None]*(a-a.mean(0)),0)))
        diagnostics=dict(folds=parts,LS_population_shift=sum(p['weight']*p['cov_b_a'] for p in parts)/(1+sum(p['weight']*p['var_b'] for p in parts)),seconds=time.perf_counter()-ts)
        cache=folder/'cache'/f'{ident}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,w=res[:,0],z=res[:,1:],w_oracle=wo,z_oracle=zo,folds=f,scales=scales,multipliers=chosen,penalties=np.array([mod.alpha for mod in models]),truth=true)
        shared=hashlib.sha256(res.tobytes()+f.tobytes()).hexdigest()
        for pipeline,w,z in [('learned',res[:,0],res[:,1:]),('oracle',wo,zo)]:
            digest=hashlib.sha256(w.tobytes()+z.tobytes()+f.tobytes()).hexdigest()
            for method in METHODS:
                ts=time.perf_counter();threshold=None if method not in ('H','HT') else scales[f]*(1.345 if method=='H' else chosen[f])
                point,V,meta,fail=estimate(w,z,f,method,threshold);seconds=time.perf_counter()-ts
                inf=inference(point,V,true,n)
                if not np.all(inf['valid_interval']):fail.append(dict(type='invalid_interval'))
                if d>1 and not inf['joint_valid']:fail.append(dict(type='joint_covariance'))
                records.append(dict(method=method,pipeline=pipeline,truth=true,estimate=point,covariance=V,coordinate_order=list(range(1,d+1)),metadata=meta,failures=fail,seconds=seconds,full_pipeline_seconds=seconds+nuisance_time+(aux if method in ('H','HT') else 0)+(tuning if method=='HT' else 0),input_hash=digest,**inf))
        result=dict(cell=cell,replication=rep,stage=stage,records=records,diagnostics=diagnostics,training_choices=choices,scales=scales,multipliers=chosen,outer_penalties=[m.alpha for m in models],outer_gcv=[m.gcv for m in models],nuisance_seconds=nuisance_time,auxiliary_seconds=aux,tuning_seconds=tuning,shared_hash=shared,elapsed=time.perf_counter()-start)
    except Exception as ex:
        records=[]
        true=1/np.arange(1,d+1)
        for pipeline in ('learned','oracle'):
            for method in METHODS:
                point=np.full(d,np.nan);V=np.full((d,d),np.nan)
                records.append(dict(method=method,pipeline=pipeline,truth=true,estimate=point,covariance=V,coordinate_order=list(range(1,d+1)),failures=[dict(type='nuisance_or_dataset',detail=str(ex))],**inference(point,V,true,n)))
        result=dict(cell=cell,replication=rep,stage=stage,records=records,traceback=traceback.format_exc(),elapsed=time.perf_counter()-start)
    save(path,result);return ident,'failed' if 'traceback' in result else 'completed'

def batch(stage,cells,R,workers):
    folder=ROOT/'runs'/stage;folder.mkdir(parents=True,exist_ok=True)
    config=dict(stage=stage,cells=cells,R=R,workers=workers,threads_per_worker=1,source_hash=source_hash(),master_seed=2026111401,stream_policy='SHA256(master, stage, cell without learner, replication, stream); independent stages',numpy=np.__version__,scipy=scipy.__version__,python=sys.version,machine=platform.platform(),processor=platform.processor(),logical_cpu=os.cpu_count(),start=datetime.now(timezone.utc).isoformat(),learner='spline_ridge',knots='five incl endpoints; open clamped cubic; drop final column; linear boundary derivative',status='running')
    cp=folder/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text(encoding='utf-8'))
        for key in ('source_hash','cells','R'):
            if old[key]!=config[key]:raise RuntimeError('Resume configuration mismatch: '+key)
    save(cp,config)
    tasks=[(stage,c,r,str(folder)) for c in cells for r in range(R)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            ident,status=future.result()
            if i%10==0 or status=='failed':print(f'{stage} {i}/{len(tasks)} {ident} {status}',flush=True)
            save(folder/'progress.json',dict(done=i,total=len(tasks),last=ident,status=status))
    config.update(status='completed',end=datetime.now(timezone.utc).isoformat());save(cp,config)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','g0_smoke_scalar','g0_smoke_vector','g0_pilot','g0','g1','g1_scalar_n1000'],required=True);p.add_argument('--replications',type=int);p.add_argument('--n',type=int);p.add_argument('--law',choices=LAWS);p.add_argument('--d',type=int);p.add_argument('--rho',type=float);p.add_argument('--learner',choices=['spline_ridge'],default='spline_ridge');p.add_argument('--workers',type=int,choices=range(1,13),default=12);args=p.parse_args()
    if args.stage in ('g1','g1_scalar_n1000'):
        if args.replications not in (None,1000) or args.n not in (None,1000) or args.d not in (None,1) or args.rho not in (None,0) or args.law is not None:
            p.error('G1 requires the full four-law scalar n=1000 grid, R=1000')
        cells=g1_preflight()
        batch('g1_scalar_n1000',cells,1000,args.workers)
        from reporting import summarize
        summarize(ROOT/'runs'/'g1_scalar_n1000')
        return
    cells=[c for c in grid() if all(getattr(args,k) is None or c[k]==getattr(args,k) for k in ('n','law','d','rho'))]
    stages=['g0_smoke_scalar','g0_smoke_vector','g0_pilot'] if args.stage=='g0' else [args.stage]
    for stage in stages:
        selected=[c for c in cells if 'smoke' not in stage or c['experiment']==stage.split('_')[-1]]
        batch(stage,selected,args.replications or (100 if stage=='g0_pilot' else 20),args.workers)
    from reporting import summarize
    for stage in stages:summarize(ROOT/'runs'/stage)

if __name__=='__main__':main()

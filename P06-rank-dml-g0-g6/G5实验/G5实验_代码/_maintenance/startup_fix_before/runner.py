"""User-run G4/G5/G6 entry; never starts another stage automatically."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'): os.environ[k]='1'
import argparse, copy, hashlib, json, time, zipfile, sys, platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE=Path(__file__).resolve().parent
SETTINGS=json.loads((HERE/'settings.json').read_text(encoding='utf-8'))
ARCHIVE=Path(SETTINGS['archive'])
sys.path.insert(0,str(HERE/'dependencies'))
import numpy as np
import core, run
BASE_RNG=core.rng
BASE_ERRORS=core.errors
BASE_RIDGE=core.Ridge
CONTEXT={}; FORESTS=[]

def baseline(cell): return 'g1_scalar_n1000' if cell['d']==1 else 'g2_vector_n1000'

def mapped_rng(stage,cell,rep,stream):
    return BASE_RNG(baseline(cell) if stage=='g5_rf' else stage,cell,rep,stream)

def smooth_errors(n,d,rho,law,r):
    if law!='sln100_tau050': return BASE_ERRORS(n,d,rho,law,r)
    L=np.linalg.cholesky(rho**np.abs(np.subtract.outer(np.arange(d),np.arange(d))))
    g=r.normal(size=(n,d))@L.T; h=r.normal(size=(n,d))@L.T
    return (np.exp(g)+.5*h-np.exp(.5))/np.sqrt(np.exp(1)*np.expm1(1)+.25)

class Forest:
    def fit(self,x,y):
        from sklearn.ensemble import RandomForestRegressor
        call=len(FORESTS); self.models=[]; seeds=[]
        for j in range(y.shape[1]):
            seed=int(BASE_RNG('g5_rf',CONTEXT['cell'],CONTEXT['rep'],f'forest_fit_{call}_target_{j}').integers(0,2**32-1))
            model=RandomForestRegressor(n_estimators=500,criterion='squared_error',max_features=1.0,max_depth=None,min_samples_leaf=10,bootstrap=True,max_samples=None,n_jobs=1,random_state=seed)
            model.fit(x,y[:,j]);self.models.append(model);seeds.append(seed)
        FORESTS.append(dict(fit_index=call,training_rows=len(x),target_seeds=seeds))
        self.alpha=np.full(y.shape[1],np.nan);self.gcv=[]
        return self
    def predict(self,x):return np.column_stack([m.predict(x) for m in self.models])

def configure(stage):
    core.Ridge=Forest if stage=='g5_rf' else BASE_RIDGE
    core.rng=mapped_rng;run.rng=mapped_rng;core.errors=smooth_errors

def task_worker(task):
    stage,cell,rep,folder=task
    CONTEXT.update(cell=cell,rep=rep);FORESTS.clear()
    if stage.startswith('g6b_'):return step_worker(task)
    result=run.worker(task)
    if stage=='g5_rf' and result[1]!='cached':
        p=Path(folder)/'records'/(result[0]+'.json');data=json.loads(p.read_text(encoding='utf-8'))
        data.update(learner='random_forest',forest_fits=FORESTS,penalty_fields='not applicable to RF; stored as null')
        # Check against the exact previous residual/fold cache, not a new RF-only seed.
        old=ARCHIVE/'runs'/baseline(cell)/'cache'/(result[0]+'.npz')
        if 'traceback' not in data:
            with np.load(old) as a, np.load(Path(folder)/'cache'/(result[0]+'.npz')) as b:
                for key in ('folds','w_oracle','z_oracle'):assert np.array_equal(a[key],b[key]),'G5 reuse mismatch: '+key
        run.save(p,data)
    return result

def step_worker(task):
    stage,cell,rep,folder=task; c=float(stage.split('_c')[1]);ident=run.cell_id(cell)+f'_{rep:04d}'
    output=Path(folder)/'records'/(ident+'.json')
    if output.exists():return ident,'cached'
    base=ARCHIVE/'runs'/baseline(cell);source=base/'records'/(ident+'.json');cache=base/'cache'/(ident+'.npz')
    data=copy.deepcopy(json.loads(source.read_text(encoding='utf-8')));data.update(stage=stage,rank_step=c,source_record=str(source),source_cache=str(cache))
    assert 'traceback' not in data,'Cannot replay missing baseline residuals'
    with np.load(cache) as a:
        for row in data['records']:
            row['reused_baseline']=row['method'] not in ('W','N')
            if row['method'] not in ('W','N'):continue
            oracle=row['pipeline']=='oracle';w=a['w_oracle' if oracle else 'w'];z=a['z_oracle' if oracle else 'z'];start=time.perf_counter()
            p,V,meta,fail=core.estimate(w,z,a['folds'],row['method'],c=c);inf=core.inference(p,V,a['truth'],cell['n'])
            if not np.all(inf['valid_interval']):fail.append(dict(type='invalid_interval'))
            if cell['d']>1 and not inf['joint_valid']:fail.append(dict(type='joint_covariance'))
            seconds=time.perf_counter()-start
            row.update(estimate=p,covariance=V,metadata=meta,failures=fail,seconds=seconds,full_pipeline_seconds=seconds+data['nuisance_seconds'],**inf)
    run.save(output,data);return ident,'completed'

def cells_for(stage):
    if stage=='G4':return 'g4_n2000',[c for c in run.grid() if c['n']==2000]
    cells=[c for c in run.grid() if c['n']==1000 and (c['d']==1 or (c['d']==3 and c['rho']==.5))]
    if stage=='G5':return 'g5_rf',cells
    if stage=='G6A':return 'g6a_tau050',[dict(c,law='sln100_tau050') for c in cells if c['law']=='sln100']
    return None,[c for c in cells if c['law'] in ('sln075','sln100')]

def code_hash():return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()

def validate_cache(folder,cells):
    valid={run.cell_id(c):c for c in cells}
    for p in (folder/'records').glob('*.json'):
        a=json.loads(p.read_text(encoding='utf-8'));c=a['cell'];r=a['replication']
        assert run.cell_id(c) in valid and c==valid[run.cell_id(c)] and 0<=r<1000
        assert p.stem==run.cell_id(c)+f'_{r:04d}' and len(a['records'])==10
        if 'traceback' not in a:
            cache=Path(a['source_cache']) if 'source_cache' in a else folder/'cache'/(p.stem+'.npz')
            with zipfile.ZipFile(cache) as z:assert z.testzip() is None

def execute(stage,cells,workers,export_only=False):
    folder=HERE/'_work'/stage;folder.mkdir(parents=True,exist_ok=True)
    config=dict(stage=stage,cells=cells,R=1000,workers=workers,threads_per_worker=1,source_hash=code_hash(),learner='random_forest' if stage=='g5_rf' else 'spline_ridge',python=sys.version,numpy=np.__version__,platform=platform.platform(),seed_policy='G5 maps all data/fold/diagnostic streams to G1/G2; forest seeds independent by replication, fit and target; G6B replays baseline caches',status='running')
    import scipy
    config['scipy']=scipy.__version__
    if stage=='g5_rf':
        import sklearn
        config.update(sklearn=sklearn.__version__,forest_settings=dict(trees=500,min_leaf=10,max_features=1.0,bootstrap=True,max_samples=None,max_depth=None,n_jobs=1))
    cp=folder/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text(encoding='utf-8'))
        for k in ('source_hash','cells','R'):assert old[k]==config[k],'Resume mismatch: '+k
    elif export_only:raise FileNotFoundError('Run this stage first')
    validate_cache(folder,cells)
    if not export_only:
        run.save(cp,config)
        todo=[(stage,c,r,str(folder)) for c in cells for r in range(1000) if not (folder/'records'/(run.cell_id(c)+f'_{r:04d}.json')).exists()]
        with ProcessPoolExecutor(max_workers=workers,initializer=configure,initargs=(stage,)) as pool:
            jobs=[pool.submit(task_worker,t) for t in todo]
            for i,f in enumerate(as_completed(jobs),1):
                ident,status=f.result()
                if i%100==0 or i==len(jobs) or status=='failed':print(stage,i,'/',len(jobs),status,flush=True)
        config['status']='completed';run.save(cp,config)
    validate_cache(folder,cells)
    from reporting import summarize
    summarize(folder)
    from deliver import export
    export(folder,HERE.parent,stage,ARCHIVE)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['G4','G5','G6A','G6B'],default=SETTINGS['stage']);p.add_argument('--workers',type=int,choices=range(1,13),default=12);p.add_argument('--export-only',action='store_true');a=p.parse_args()
    assert a.stage==SETTINGS['stage'],'Use the matching experiment folder'
    for name,digest in SETTINGS['frozen'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,'Frozen source changed: '+name
    accepted=json.loads((ARCHIVE/'reports'/'G3_final_audit.json').read_text(encoding='utf-8'))
    assert accepted['completed_datasets']==20000 and accepted['all_cells_expected_R']
    stage,cells=cells_for(a.stage)
    if a.stage=='G5':
        try:import sklearn
        except ImportError:raise SystemExit('First run install_G5_dependencies.ps1 in this code folder.')
    if a.stage=='G6B':
        for step in (.5,2.):execute('g6b_c'+str(step),cells,a.workers,a.export_only)
    else:execute(stage,cells,a.workers,a.export_only)

if __name__=='__main__':main()

"""Export saved nuisance inputs only. No simulation or model training."""
import hashlib,json,pathlib,shutil,sys
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
RUN=ROOT.parent/'n1000_R1000_f76477e6a5dc'
REFERENCE=ROOT.parent/'R_cross_validation'
SCENARIOS=['gaussian','t2.1','lognormal2']
AUTHOR_SHA='fa32ffae449144fb0bf85c894f2cd9271bc5f492657f289db324da8db4b9bd71'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for folder in ('inputs','results','logs','reference'): (ROOT/folder).mkdir(exist_ok=True)
    author=REFERENCE/'downloads'/'dri_functions.r';assert sha(author)==AUTHOR_SHA
    shutil.copy2(author,ROOT/'reference'/'dri_functions.r')
    for name in ('r_package_versions.csv','np_runtime_options.json','package_lock.json'):
        shutil.copy2(REFERENCE/name,ROOT/'reference'/name)
    raw=pd.read_csv(RUN/'raw_results.csv',float_precision='round_trip')
    c=raw[raw.method=='dukes_dr_gcm'].set_index(['scenario','replication'])
    assert len(c)==3000 and (c.status=='ok').all()
    sources={}
    for p in [RUN/'raw_results.csv',RUN/'config.json',RUN/'fold_diagnostics.csv',RUN/'summary.csv',
              pathlib.Path('C:/Users/Lenovo/Desktop/DML_Dukes_n1000_R1000.ipynb')]:sources[str(p)]=sha(p)
    rows=[]
    # Interleave scenarios so fixed worker shards distribute them evenly.
    for rep in range(1000):
        for scenario_index,scenario in enumerate(SCENARIOS):
            stem=f'{scenario}_{rep:04d}';meta=c.loc[(scenario,rep)]
            task_path=RUN/'tasks'/f'{stem}.json';task=json.loads(task_path.read_text(encoding='utf-8'))
            array_path=RUN/task['arrays_file'];array_sha=sha(array_path)
            assert array_sha==task['arrays_sha256']
            sources[str(task_path)]=sha(task_path);sources[str(array_path)]=array_sha
            with np.load(array_path,allow_pickle=False) as z:
                D=z['d_res']+z['g_hat'];y=z['b']+z['m_hat'];fold=z['fold_ids']
                assert len(D)==1000 and np.array_equal(np.bincount(fold),np.repeat(200,5))
                assert np.array_equal(D-z['g_hat'],z['a']) and np.array_equal(y-z['m_hat'],z['b'])
                table=pd.DataFrame({'observation_id':np.arange(1,1001),'D':D,'y_transformed':y,
                                    'g_initial':z['g_hat'],'m_initial':z['m_hat'],'fold':fold+1})
                assert np.isfinite(table.to_numpy()).all()
            input_path=ROOT/'inputs'/f'{stem}.csv'
            table.to_csv(input_path,index=False,float_format='%.17g')
            rows.append(dict(scenario=scenario,replication=rep,stem=stem,dataset_seed=int(meta.dataset_seed),
                             crossfit_seed=int(meta.crossfit_seed),r_auxiliary_seed=20261030+100000*scenario_index+rep,
                             input_sha256=sha(input_path),array_sha256=array_sha,fold_hash=meta.fold_hash,
                             T_Python=float(meta['T']),p_Python=float(meta.p_value)))
        if (rep+1)%100==0:print('Exported',len(rows),'/ 3000',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'replications.csv',index=False,float_format='%.17g')
    config=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    manifest=dict(run=str(RUN),reference_validation=str(REFERENCE),author_commit='e7dfb038117aea3d95ca1371e2e85557ff9f25d8',
                  author_sha256=AUTHOR_SHA,config_hash=config['config_hash'],source_files_sha256=sources,
                  data_reconstruction='D=d_res+g_hat; y=b+m_hat; exact residual roundtrip for all 3000 inputs',
                  seed_rule='20261030+100000*scenario_index+replication; identical to prior 30 validation samples',
                  scenario_order=SCENARIOS,theta0=1,N=1000,nsplits=5,planned=3000,
                  python_executable=sys.executable,source_files_checked=len(sources))
    (ROOT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('All 3000 inputs ready; original files read only.',flush=True)
if __name__=='__main__':main()

"""Run a fixed set of single-threaded R processes with deterministic per-data seeds."""
import datetime,hashlib,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
RSCRIPT=ROOT.parent/'R_cross_validation'/'runtime'/'R-4.6.1'/'bin'/'Rscript.exe'
WORKERS=12
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'reference'/'dri_functions.r')==manifest['author_sha256']
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
    env['LANG']='C';env['LC_ALL']='C'
    started=time.time(); processes=[];handles=[]
    metadata={'workers':WORKERS,'threads_per_worker':1,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'Rscript':str(RSCRIPT),'worker_script_sha256':sha(ROOT/'run_author_worker.R')}
    try:
        for worker in range(WORKERS):
            log=(ROOT/'logs'/f'worker_{worker:02d}.log').open('w',encoding='utf-8');handles.append(log)
            kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            processes.append(subprocess.Popen([str(RSCRIPT),'--vanilla','run_author_worker.R',str(worker),str(WORKERS)],
                                              cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,**kwargs))
        metadata['pids']=[p.pid for p in processes]
        (ROOT/'execution.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        while any(p.poll() is None for p in processes):
            complete=len(list((ROOT/'results').glob('*.json')))
            print(f'Completed {complete}/3000; elapsed {time.time()-started:.1f}s; active {sum(p.poll() is None for p in processes)}',flush=True)
            time.sleep(20)
        metadata.update(elapsed_seconds=time.time()-started,exit_codes=[p.returncode for p in processes],
                        end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        result_files=len(list((ROOT/'results').glob('*.json'))))
        (ROOT/'execution.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(json.dumps(metadata,indent=2),flush=True)
        if any(metadata['exit_codes']):raise RuntimeError('R worker failure; inspect logs')
    finally:
        for log in handles:log.close()
if __name__=='__main__':main()

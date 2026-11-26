"""User-run Scenario A driver. Defaults: 12 workers, one numerical thread each."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):os.environ[key]="1"
import argparse,csv,hashlib,io,json,multiprocessing as mp,platform,sqlite3,sys,time,traceback,zlib
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy
import psutil
from tqdm import tqdm
from threadpoolctl import threadpool_limits
import methods as m
ROOT=Path(__file__).resolve().parents[1]

def clean(obj):
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,(np.integer,np.bool_)):return obj.item()
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [clean(x) for x in obj]
    return obj
def atomic(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2),encoding="utf-8")
    # Windows readers can briefly hold a file without delete-sharing.
    for attempt in range(20):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.02*(attempt+1),.2))
def process_rss_mib():
    total=0
    try:processes=[psutil.Process()]+psutil.Process().children()
    except psutil.Error:return None
    for process in processes:
        try:total+=process.memory_info().rss
        except psutil.Error:pass
    return total/2**20

def hardware():
    count=os.cpu_count() or 1
    try:affinity=len(psutil.Process().cpu_affinity())
    except (AttributeError,psutil.Error):affinity=count
    quotas=[]
    for path in (Path("/sys/fs/cgroup/cpu.max"),):
        if path.exists():
            quota,period=path.read_text().split()
            if quota!="max":quotas.append(int(quota)/int(period))
    q=Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");p=q.with_name("cpu.cfs_period_us")
    if q.exists() and p.exists() and int(q.read_text())>0:quotas.append(int(q.read_text())/int(p.read_text()))
    available=min([affinity]+quotas)
    return dict(platform=platform.platform(),cpu=platform.processor(),logical_cpus=count,affinity=affinity,
        cpu_quota=min(quotas) if quotas else None,available_cpus=available,memory_gib=psutil.virtual_memory().total/2**30,
        free_disk_gib=psutil.disk_usage(str(ROOT)).free/2**30,python=sys.version,numpy=np.__version__,scipy=scipy.__version__)
def signature():
    return hashlib.sha256(b"".join((ROOT/"代码"/name).read_bytes() for name in ("methods.py","legacy_core.py"))+
                           (ROOT/"sources/Rank DML.tex").read_bytes()).hexdigest()
def information():
    path=ROOT/"config/theory.json"
    if path.exists():
        data=json.loads(path.read_text(encoding="utf-8"))
        if data["signature"]==signature():return data["laws"]
    data={}
    for law in m.LAWS:
        print("Density/information integration:",law,flush=True)
        a=m.theory(law,192)
        for order in (384,768,1536,3072):
            b=m.theory(law,order)
            difference=abs(a["fisher"]-b["fisher"])/b["fisher"]
            if difference<=2e-4 and abs(b["mass"]-1)<=2e-5 and order>=({"sln075":768,"sln100_tau050":3072}.get(law,384)):
                break
            a=b
        else:
            raise RuntimeError(f"Oracle quadrature validation failed: {law}, relative difference={difference}, mass={b['mass']}")
        b["relative_fisher_difference"]=difference;data[law]=b
    atomic(path,dict(signature=signature(),laws=data))
    return data
def load_reuse():
    path=ROOT/"audit/existing_data_audit.csv"
    if not path.exists():raise RuntimeError("Run audit.py before main; refusing to rerun potentially reusable historical data.")
    with path.open(encoding="utf-8-sig",newline="") as f:
        return {(int(r["n"]),r["law"],int(r["replication"])):r for r in csv.DictReader(f) if r["status"]=="REUSE"}
def worker(task):
    stage,n,law,rep,source,info=task;start=time.perf_counter();cache=None
    meta=dict(stage=stage,n=n,law=law,replication=rep,theta0=1.,source="historical" if source else "new",pid=os.getpid())
    try:
        with threadpool_limits(1):
            if source:
                packed=ROOT/"cache/reused_inputs.sqlite"
                if packed.exists():
                    with sqlite3.connect(packed.as_uri()+"?mode=ro",uri=True) as archive:
                        entry=archive.execute("SELECT cache,record FROM inputs WHERE n=? AND law=? AND rep=?",(n,law,rep)).fetchone()
                    if entry is None:raise RuntimeError("Missing portable historical input")
                    raw,record_bytes=entry[0],zlib.decompress(entry[1])
                else:
                    raw=Path(source["cache"]).read_bytes();record_bytes=Path(source["record"]).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=source["cache_sha256"]:raise RuntimeError("Historical cache changed")
                with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                    cache={k:z[k] for k in ("w","z","folds","w_oracle","z_oracle","scales")}
                old=json.loads(record_bytes.decode("utf-8-sig"))
                if hashlib.sha256(record_bytes).hexdigest()!=source["record_sha256"]:raise RuntimeError("Historical record changed")
                meta.update(seed_key=source["seed_key"],source_cache=source["cache"],source_hash=source["cache_sha256"],
                    nuisance_seconds=old["nuisance_seconds"],diagnostics=old["diagnostics"]["folds"],
                    outer_penalties=old["outer_penalties"],outer_gcv=old["outer_gcv"])
            else:
                cache,detail=m.nuisance(stage,n,law,rep);meta.update(detail)
                meta["seed_key"]=json.dumps([m.VERSION,20261118,stage,n,law,rep,"data"])
            cache["z"]=cache["z"].reshape(-1);cache["z_oracle"]=cache["z_oracle"].reshape(-1);cache["scales"]=cache["scales"].reshape(-1)
            rows=m.estimate_all(cache,law,m.rng(stage,n,law,rep,"max-t"),info)
            meta["input_hash"]=hashlib.sha256(b"".join(cache[k].tobytes() for k in ("w","z","folds"))).hexdigest()
    except Exception as exc:
        meta["dataset_failure"]=traceback.format_exc()
        rows=[dict(method=name,estimate=None,se=None,valid_estimator=False,valid_interval=False,covered=False,
                   length=None,max_t_covered=None,max_t_length=None,seconds=0,metadata=dict(failure=str(exc))) for name in m.METHODS]
    meta.update(seconds=time.perf_counter()-start,rows=rows)
    return meta,cache

def run(args):
    stage="smoke" if args.smoke else "pilot" if args.pilot else "main"
    if stage!="smoke" and (args.n is None or args.law is None):
        raise ValueError("Choose exactly one condition: --n and --law are required")
    if stage!="smoke":
        checked=ROOT/"config/code_validation.json"
        if not checked.exists() or json.loads(checked.read_text()) .get("signature")!=signature():
            raise RuntimeError("Run validate.py after a successful smoke before pilot/main")
    env=hardware()
    if env["free_disk_gib"]<2:raise RuntimeError("Less than 2 GiB free disk; stopped")
    workers=min(args.workers,max(1,int(env["available_cpus"])))
    if not 1<=args.workers<=12:raise ValueError("Formal experiments limited to 12 workers")
    atomic(ROOT/"config/dependency_versions.json",env)
    condition=(args.n,args.law) if args.n in (500,1000,2000) and args.law else None
    if condition:
        from layout import paths
        folder=paths(*condition,stage)["work"]
    else:folder=ROOT/(stage if stage!="main" else "replications")
    folder.mkdir(parents=True,exist_ok=True);lock=ROOT/"logs/running.lock"
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f"Lock exists: {lock}; only remove after confirming old process has stopped")
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        db=sqlite3.connect(folder/"records.sqlite")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS config (signature TEXT)")
        stored=db.execute("SELECT signature FROM config").fetchone()
        sig=signature()
        if stored and stored[0]!=sig:
            exists=db.execute("SELECT name FROM sqlite_master WHERE name='records'").fetchone()
            if exists and db.execute("SELECT COUNT(*) FROM records").fetchone()[0]:
                raise RuntimeError("Scientific code changed; use a separate delivery/version directory")
            db.execute("UPDATE config SET signature=?",(sig,))
        if not stored:db.execute("INSERT INTO config VALUES (?)",(sig,))
        db.execute("CREATE TABLE IF NOT EXISTS records(n INTEGER,law TEXT,rep INTEGER,payload TEXT,PRIMARY KEY(n,law,rep))")
        db.execute("CREATE TABLE IF NOT EXISTS cache_batches(id TEXT PRIMARY KEY,payload BLOB)")
        db.commit()
        done=set(db.execute("SELECT n,law,rep FROM records").fetchall())
        if done and not args.resume:raise RuntimeError("Existing output: add --resume")
        info=information();reuse=load_reuse() if stage=="main" else {}
        ns=[args.n] if args.n else ([100] if stage=="smoke" else [1000,500,2000])
        laws=[args.law] if args.law else m.LAWS
        tasks=[];requested=0
        for n in ns:
            reps=2 if stage=="smoke" else 100 if stage=="pilot" else (2000 if n==1000 else 1000)
            stop=args.stop if args.stop is not None else reps
            if stop>reps or args.start<0 or stop<=args.start:raise ValueError("Invalid replication range")
            for law in laws:
                for rep in range(args.start,stop):
                    requested+=1
                    if (n,law,rep) not in done:tasks.append((stage,n,law,rep,reuse.get((n,law,rep)),info[law]))
        print(f"Stage={stage}; workers={workers}; requested={requested}; remaining={len(tasks)}; output={folder}",flush=True)
        atomic(folder/"session_config.json",dict(stage=stage,arguments=vars(args),signature=sig,environment=env,workers=workers))
        start=time.perf_counter();completed=failures=0;buffer=[]
        def flush():
            if not buffer:return
            arrays={}
            for meta,cache in buffer:
                if cache is not None:
                    prefix=f"n{meta['n']}_{meta['law']}_r{meta['replication']}"
                    for key,value in cache.items():arrays[prefix+"__"+key]=value
            binary=io.BytesIO();np.savez_compressed(binary,**arrays)
            batch=hashlib.sha256(binary.getvalue()).hexdigest()
            with db:
                db.execute("INSERT OR IGNORE INTO cache_batches VALUES (?,?)",(batch,binary.getvalue()))
                for meta,cache in buffer:
                    meta["cache_batch"]=batch if cache is not None else None
                    db.execute("INSERT INTO records VALUES (?,?,?,?)",(meta["n"],meta["law"],meta["replication"],json.dumps(clean(meta),ensure_ascii=False)))
            buffer.clear()
        def progress(active):
            try:
                atomic(folder/"progress.json",dict(stage=stage,requested=requested,previously_done=requested-len(tasks),
                    completed_this_session=completed,replications_with_invalid_intervals=failures,active_submitted_tasks=active,
                    workers=workers,elapsed_seconds=time.perf_counter()-start,output=str(folder),parent_pid=os.getpid(),
                    process_tree_rss_mib=process_rss_mib()))
            except OSError as exc:
                # A display-only snapshot must not interrupt committed scientific work.
                print(f"Progress snapshot temporarily unavailable: {exc}",flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:
            queue=iter(tasks);futures={}
            def submit():
                for _ in range(workers-len(futures)):
                    task=next(queue,None)
                    if task is None:break
                    futures[pool.submit(worker,task)]=task
            submit()
            with tqdm(total=requested,initial=requested-len(tasks),desc=stage,unit="rep",ascii=True,mininterval=1) as bar:
                while futures:
                    ready,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                    if not ready:
                        progress(len(futures));print(f"Heartbeat: {len(futures)} submitted/running; {completed}/{len(tasks)} done; elapsed {time.perf_counter()-start:.0f}s",flush=True)
                    for future in ready:
                        task=futures.pop(future)
                        try:result=future.result()
                        except Exception as exc:
                            raise RuntimeError(f"Worker process failure; committed records preserved: {task[:4]}") from exc
                        buffer.append(result);completed+=1
                        failures+=any(not row["valid_interval"] for row in result[0]["rows"])
                        if len(buffer)>=10:flush()
                        bar.update();bar.set_postfix(invalid_reps=failures,n=task[1],law=task[2],refresh=False)
                    progress(len(futures));submit()
        flush();progress(0);db.close()
        from reporting import summarize
        summarize(stage,condition=condition)
        print(f"FINISHED {stage}: {completed} new/replayed tasks, {failures} with >=1 invalid interval. Not a statistical acceptance test.",flush=True)
    finally:lock.unlink(missing_ok=True)
def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for name in ("smoke","pilot","main"):g.add_argument("--"+name,action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--workers",type=int,default=12)
    p.add_argument("--law",choices=m.LAWS);p.add_argument("--n",type=int)
    p.add_argument("--start",type=int,default=0);p.add_argument("--stop",type=int)
    a=p.parse_args()
    if a.n is not None and a.n not in (100,200,500,1000,2000):p.error("Unsupported n")
    if not a.smoke and a.n in (100,200):p.error("Small n reserved for smoke")
    if not a.smoke and (a.n is None or a.law is None):p.error("Choose ONE experiment: both --n and --law are required")
    run(a)
if __name__=="__main__":
    mp.freeze_support();main()


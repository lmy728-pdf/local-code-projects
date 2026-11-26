"""Run one Scenario D stage. Default concurrency is 12 processes."""
import dgp
from estimators import one_rep, method_names
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sqlite3
import sys
import warnings

import numpy as np
import scipy
from scipy.optimize import OptimizeWarning
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
THREAD_LIMIT = None

def initialize_worker():
    global THREAD_LIMIT
    THREAD_LIMIT=threadpool_limits(1)
    warnings.filterwarnings("ignore", message=r"Unrecognized options detected: .*threads.*",
                            category=OptimizeWarning)

def signature():
    sha=hashlib.sha256()
    for name in ("dgp.py","estimators.py","methods.py","legacy_core.py","experiment_manifest.json"):
        sha.update((ROOT/name).read_bytes())
    sha.update(f"{platform.python_version()}|{np.__version__}|{scipy.__version__}".encode())
    return sha.hexdigest()

def canonical(value):
    if isinstance(value, dict):
        return {key:canonical(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [canonical(item) for item in value]
    if isinstance(value,np.generic):
        return canonical(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value

def db_path(output_root,mode):
    return output_root/"checkpoints"/f"{mode}.sqlite"

def execute(mode,scenario,n,workers,resume,output_root,stop_after=None):
    keys=[(mode,sc,size,rep) for sc in dgp.SCENARIOS if scenario is None or sc==scenario
          for size in dgp.SIZES if n is None or size==n for rep in range(dgp.COUNTS[mode])]
    folder=output_root/"checkpoints"
    folder.mkdir(parents=True,exist_ok=True)
    lock=folder/"running.lock"
    with lock.open("x",encoding="utf-8") as file:
        file.write(str(os.getpid()))
    try:
        with sqlite3.connect(db_path(output_root,mode)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY CHECK(id=1), signature TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS replications (
                       scenario TEXT, n INTEGER, rep INTEGER, json TEXT,
                       PRIMARY KEY(scenario,n,rep))""")
            version=signature()
            meta=db.execute("SELECT signature FROM meta WHERE id=1").fetchone()
            if meta is None:
                db.execute("INSERT INTO meta VALUES (1,?)",(version,))
                db.commit()
            elif version!=meta[0]:
                raise ValueError("Checkpoint science or numerical-library version differs")
            prior={tuple(row) for row in db.execute("SELECT scenario,n,rep FROM replications")}
            done=sum(key[1:] in prior for key in keys)
            if done and not resume:
                raise ValueError("Existing records in requested cells. Rerun with --resume.")
            remaining=[key for key in keys if key[1:] not in prior]
            if stop_after is not None:
                remaining=remaining[:stop_after]
            if not remaining:
                print(f"{mode} complete for selection: {done}/{len(keys)}",flush=True)
                return
            pending={}
            tasks=iter(remaining)
            parallelism=min(workers,os.cpu_count() or 1,len(remaining))
            print(f"{mode.upper()} | {len(keys)} target | {done} saved | {parallelism} workers",flush=True)
            with ProcessPoolExecutor(max_workers=parallelism,mp_context=mp.get_context("spawn"),
                                     initializer=initialize_worker) as pool:
                def submit():
                    for _ in range(parallelism-len(pending)):
                        job=next(tasks,None)
                        if job is not None:
                            pending[pool.submit(one_rep,job)]=job
                submit()
                with tqdm(total=len(keys),initial=done,desc=mode,unit="rep",ascii=True) as bar:
                    while pending:
                        ready,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                        for future in ready:
                            job=pending.pop(future)
                            record=canonical(future.result())
                            assert (record["mode"],record["scenario"],record["n"],record["rep"])==job
                            assert tuple(m["name"] for m in record["methods"])==method_names(job[1])
                            with db:
                                db.execute("INSERT INTO replications VALUES(?,?,?,?)",
                                           (*job[1:],json.dumps(record,allow_nan=False)))
                            invalid=sum(not c["valid"] for m in record["methods"] for c in m["intervals"].values())
                            bar.set_postfix(cell=f"{job[1]}-{job[2]}",invalid_ci_last_rep=invalid,refresh=False)
                            bar.update(1)
                        submit()
            print(f"Saved {len(remaining)} new results. {done+len(remaining)}/{len(keys)} requested.",flush=True)
    finally:
        lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=dgp.MODES,required=True)
    parser.add_argument("--workers",type=int,default=12,choices=range(1,13))
    parser.add_argument("--scenario",choices=dgp.SCENARIOS)
    parser.add_argument("--n",type=int,choices=dgp.SIZES)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--rep",type=int)
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--output-root",type=Path,default=ROOT)
    args=parser.parse_args()
    if args.rep is not None:
        if args.scenario is None or args.n is None or not 0<=args.rep<dgp.COUNTS[args.mode]:
            parser.error("--rep needs --scenario and --n and a valid zero-based rep ID")
        initialize_worker()
        print(json.dumps(canonical(one_rep((args.mode,args.scenario,args.n,args.rep))),indent=2,
                         allow_nan=False))
        return
    if args.stop_after is not None and args.stop_after<=0:
        parser.error("--stop-after must be positive")
    execute(args.mode,args.scenario,args.n,args.workers,args.resume,args.output_root,args.stop_after)

if __name__=="__main__":
    mp.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Committed replications are safe. Use --resume.",file=sys.stderr)
        raise SystemExit(130)


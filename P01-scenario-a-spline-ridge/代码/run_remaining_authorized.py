"""Authorized sequential completion. Does not modify scientific code or rerun completed rows."""
from pathlib import Path
import os,sys,json,sqlite3,subprocess,time,hashlib,traceback
import psutil
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"logs/authorized_run_status.json"
LOCK=ROOT/"logs/authorized_run.lock"
CURRENT=None
def write_state(**values):
    values.update(coordinator_pid=os.getpid(),updated_local=time.strftime("%Y-%m-%d %H:%M:%S"))
    temp=STATE.with_suffix(".tmp");temp.write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding="utf-8")
    for attempt in range(20):
        try:os.replace(temp,STATE);return
        except PermissionError:
            if attempt==19:return
            time.sleep(min(.02*(attempt+1),.2))
def wait_existing():
    tasklock=ROOT/"logs/running.lock";announced=False
    while tasklock.exists():
        raw=tasklock.read_text().strip()
        if not raw:time.sleep(1);continue
        pid=int(raw)
        if not psutil.pid_exists(pid):
            # Remove only a dead process's exact lock, and recheck content.
            if tasklock.read_text().strip()==raw:tasklock.unlink()
            continue
        if not announced:
            print(f"WAIT existing experiment PID={pid}",flush=True);announced=True
        write_state(status="waiting_existing_experiment",existing_pid=pid)
        time.sleep(5)
def count(entry):
    code=(ROOT/entry["notebook"]).parent;dbpath=code/"_work/replications/records.sqlite"
    if not dbpath.exists():return 0
    with sqlite3.connect(dbpath.as_uri()+"?mode=ro",uri=True) as db:
        got=db.execute("SELECT signature FROM config").fetchone()
        if not got or got[0]!=signature:raise RuntimeError("Scientific signature mismatch: "+entry["id"])
        bad=db.execute("SELECT COUNT(*) FROM records WHERE n!=? OR law!=? OR rep<0 OR rep>=?",(entry["n"],entry["law"],entry["target"])).fetchone()[0]
        if bad:raise RuntimeError("Unexpected record identities: "+entry["id"])
        return db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
def main():
    global CURRENT,signature
    import runner
    signature=runner.signature()
    checked=json.loads((ROOT/"config/code_validation.json").read_text(encoding="utf-8"))
    if checked["status"]!="PASS" or checked["signature"]!=signature:raise RuntimeError("Validation mismatch")
    for name,digest in json.loads((ROOT/"config/code_checksums.json").read_text(encoding="utf-8")).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError("Code changed: "+name)
    if psutil.disk_usage(str(ROOT)).free<5*2**30:raise RuntimeError("Insufficient free disk")
    manifest=json.loads((ROOT/"config/experiment_manifest.json").read_text(encoding="utf-8"))
    entries=manifest["notebooks"];done=[]
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    env=os.environ.copy();env.update(PYTHONUTF8="1",PYTHONIOENCODING="utf-8",PYTHONUNBUFFERED="1")
    try:
        wait_existing()
        for entry in entries:
            wait_existing();finished=count(entry)
            if finished>entry["target"]:raise RuntimeError("More records than planned")
            if finished==entry["target"]:
                print("SKIP complete",entry["id"],finished,flush=True);done.append(entry["id"]);continue
            if psutil.disk_usage(str(ROOT)).free<5*2**30:raise RuntimeError("Insufficient free disk")
            code=(ROOT/entry["notebook"]).parent
            log=code/"logs"/("authorized_"+time.strftime("%Y%m%d_%H%M%S")+".log")
            cmd=[sys.executable,"-u",str(code/"run_this_experiment.py"),"--main","--workers","12","--resume"]
            print(f"START {entry['id']} n={entry['n']} law={entry['law']} existing={finished}/{entry['target']} log={log}",flush=True)
            with log.open("w",encoding="utf-8") as stream:
                CURRENT=subprocess.Popen(cmd,cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT)
                while CURRENT.poll() is None:
                    progress=code/"_work/replications/progress.json";p={}
                    if progress.exists():
                        try:p=json.loads(progress.read_text(encoding="utf-8"))
                        except (OSError,json.JSONDecodeError):pass
                    write_state(status="running",group=entry["id"],n=entry["n"],law=entry["law"],target=entry["target"],
                                experiment_pid=CURRENT.pid,completed_groups=done,progress=p,log=str(log))
                    time.sleep(5)
                rc=CURRENT.returncode;CURRENT=None
            if rc:raise RuntimeError(f"{entry['id']} exit={rc}; see {log}")
            after=count(entry)
            if after!=entry["target"]:raise RuntimeError(f"Incomplete {entry['id']}: {after}")
            done.append(entry["id"]);print("FINISHED",entry["id"],after,flush=True)
        write_state(status="collecting",completed_groups=done)
        print("COLLECT all completed groups",flush=True)
        with (ROOT/"logs/authorized_collection.log").open("w",encoding="utf-8") as stream:
            result=subprocess.run([sys.executable,"-u",str(ROOT/"代码/collect_results.py"),"--stage","main"],
                                  env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError("Collection failed; see logs/authorized_collection.log")
        totals={e["id"]:count(e) for e in entries}
        if sum(totals.values())!=36000:raise RuntimeError("Total count mismatch")
        write_state(status="complete",completed_groups=done,total_records=sum(totals.values()),counts=totals)
        print("COMPLETE 27 conditions; 36000 replications; reports generated.",flush=True)
    except BaseException:
        write_state(status="stopped_with_error",error=traceback.format_exc(),child_pid=CURRENT.pid if CURRENT else None)
        raise
    finally:LOCK.unlink(missing_ok=True)
if __name__=="__main__":main()


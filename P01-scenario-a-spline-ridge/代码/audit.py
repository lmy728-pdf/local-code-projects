"""Read-only audit of historical residual caches; no historical writes."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
from pathlib import Path
import json,hashlib,csv,time,sys
import numpy as np
from tqdm import tqdm
import legacy_core as old
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=Path(r"C:\Users\Lenovo\Documents\ChatGPT\毕设\rank_dml_study")
DESKTOP=Path(r"C:\Users\Lenovo\Desktop\Rank-DML_G0-G6实验汇总")
SOURCES=[("G1",ARCHIVE/"runs/g1_scalar_n1000",1000,("gaussian","t3","sln075")),
         ("G3",ARCHIVE/"runs/g3_n500",500,("gaussian","t3","sln075")),
         ("G4",DESKTOP/"G4实验/G4实验_代码/_work/g4_n2000",2000,("gaussian","t3","sln075")),
         ("G6A",DESKTOP/"G6A实验/G6A实验_代码/_work/g6a_tau050",1000,("sln100_tau050",))]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit():
    results=[];seen=set();start=time.perf_counter()
    for label,folder,n,laws in SOURCES:
        config_path=folder/"config.json"
        config=json.loads(config_path.read_text(encoding="utf-8"))
        stage=config["stage"]
        source_core=ARCHIVE/"core.py" if label in ("G1","G3") else folder.parents[1]/"core.py"
        assert digest(source_core)==digest(ROOT/"代码/legacy_core.py"),"Historical core differs: "+str(source_core)
        for law in laws:
            cell=dict(experiment="scalar",law=law,n=n,d=1,rho=0)
            for rep in tqdm(range(1000),desc=f"Audit {label}/{law}",ascii=True):
                ident=f"scalar_{law}_{n}_1_0_{rep:04d}"
                cache=folder/"cache"/(ident+".npz");record=folder/"records"/(ident+".json")
                row=dict(source=label,stage=stage,n=n,law=law,replication=rep,cache=str(cache),record=str(record),status="INCOMPATIBLE",reason="",cache_sha256="",record_sha256="",seed_key="")
                try:
                    data=json.loads(record.read_text(encoding="utf-8"))
                    assert data["cell"]==cell and data["replication"]==rep and data["stage"]==stage
                    assert "traceback" not in data
                    seed_key=json.dumps([2026111401,stage,cell,rep,"data"],sort_keys=True)
                    assert seed_key not in seen,"duplicate seed";seen.add(seed_key)
                    row["seed_key"]=seed_key
                    with np.load(cache,allow_pickle=False) as arr:
                        w=arr["w"];z=arr["z"];f=arr["folds"]
                        assert w.shape==(n,) and z.shape==(n,1)
                        assert all(np.isfinite(arr[key]).all() for key in ("w","z","scales","penalties","w_oracle","z_oracle"))
                        assert np.array_equal(np.bincount(f,minlength=5),np.full(5,n//5))
                        assert np.array_equal(arr["truth"],[1.])
                        assert np.array_equal(f,old.folds(n,5,old.rng(stage,cell,rep,"outer")))
                        if law=="sln100_tau050":
                            r=old.rng(stage,cell,rep,"data");x=old.covariates(n,r);m,ell,_=old.truth(x,1,"scalar");v=r.normal(size=n)
                            g=r.normal(size=(n,1));h=r.normal(size=(n,1))
                            e=(np.exp(g)+.5*h-np.exp(.5))/np.sqrt(np.exp(1)*np.expm1(1)+.25)
                            zo=v[:,None]+e
                        else:
                            x,D,Y,v,zo,_=old.generate(cell,old.rng(stage,cell,rep,"data"))
                        assert np.allclose(v,arr["w_oracle"],rtol=1e-13,atol=1e-13),"DGP treatment mismatch"
                        assert np.allclose(zo,arr["z_oracle"],rtol=1e-13,atol=1e-13),"DGP error mismatch"
                        shared=hashlib.sha256(np.column_stack([w,z]).tobytes()+f.tobytes()).hexdigest()
                        assert shared==data["shared_hash"],"residual hash mismatch"
                        if rep==0:
                            if law=="sln100_tau050":D=m+v;Y=ell+zo
                            res,mods=old.crossfit(x,np.column_stack([D,Y]),f)
                            assert np.allclose(res,np.column_stack([w,z]),rtol=1e-8,atol=1e-8),"Spline replay mismatch"
                    row.update(status="REUSE",cache_sha256=digest(cache),record_sha256=digest(record))
                except Exception as exc:row["reason"]=f"{type(exc).__name__}: {exc}"
                results.append(row)
    destination=ROOT/"audit/existing_data_audit.csv"
    with destination.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    reusable=sum(x["status"]=="REUSE" for x in results)
    (ROOT/"audit/reuse_validation.md").write_text(f"# Historical audit\n\nChecked {len(results)} records; reusable {reusable}; incompatible {len(results)-reusable}.\n\nAll caches checked for identity, finite arrays, balanced folds, regenerated oracle residuals, shared input hash and seed uniqueness. One nuisance replay per source/law verifies the spline implementation. Historical second-stage estimates are NEVER reused. Current runtime is recorded separately; floating-point replay tolerance 1e-8. Old sln100(tau=.2), vector datasets and pilots excluded.\n\nElapsed {time.perf_counter()-start:.2f}s.\n",encoding="utf-8")
    (ROOT/"audit/seed_integrity_report.md").write_text(f"# Seed integrity\n\n{len(seen)} unique historical data seed keys. New namespace scenario-a-20261118-v1 uses SHA256 -> SeedSequence, distinct stage tokens smoke/pilot/main and n/law/rep/stream; it never reuses the historical master key. Historical rows carry their exact seed key and immutable cache digest.\n",encoding="utf-8")
    print("AUDIT",len(results),"REUSE",reusable,"INCOMPATIBLE",len(results)-reusable,flush=True)
if __name__=="__main__":audit()


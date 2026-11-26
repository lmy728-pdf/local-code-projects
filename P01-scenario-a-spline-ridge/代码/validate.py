"""Small deterministic implementation checks; never starts pilot or main."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
import json,sqlite3,hashlib,io,time
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm
import methods as m
import runner as r
ROOT=r.ROOT
checks=[]
def check(name,fn):
    start=time.perf_counter();fn()
    checks.append(dict(name=name,status="PASS",seconds=time.perf_counter()-start))
    print("PASS:",name,flush=True)
def finite_difference():
    for n in (100,500,2000):
        t=.731
        def fun(x):
            u=np.sqrt(n)*(x-t)
            return np.array([5-3*u+u**2+2*u**3-4*u**4,2+7*u])
        assert_allclose(m.fivepoint(fun,t,n),[3,-7],rtol=1e-12,atol=1e-11)
def direct_minimum():
    gen=np.random.default_rng(51);w=gen.normal(size=30);z=w+gen.normal(size=30);f=np.arange(30)%5
    pre=m.old.direct_w(w,z,f);slopes=[];weights=[]
    for k in range(5):
        ww=w[f==k];zz=z[f==k]
        for i in range(len(ww)):
            for j in range(i):
                if ww[i]!=ww[j]:
                    slopes.append((zz[i]-zz[j])/(ww[i]-ww[j]));weights.append(abs(ww[i]-ww[j])/(len(w)*(len(ww)-1)))
    slopes=np.array(slopes);weights=np.array(weights)
    objective=lambda t:np.sum(weights*abs(slopes-t))
    assert objective(pre)<=min(map(objective,slopes))+1e-12
    assert_allclose(m.old.direct_w(w,z+np.array([2,-3,.5,8,1])[f],f),pre,atol=1e-12)
def tie_scores():
    e=np.array([1.,1.,2.,3.,4.]);u=np.arange(1,6)/6
    for kind in ("W","N",1,2,3,4,5,6):
        grid=u-.5 if kind=="W" else norm.ppf(u) if kind=="N" else np.sqrt(2*kind+1)*m.eval_legendre(kind,2*u-1)
        grid-=grid.mean();grid[:2]=grid[:2].mean()
        assert_allclose(m.score_grid(e,kind),grid,atol=1e-14)
        assert abs(m.score_grid(e,kind).sum())<1e-13
def rank_equations():
    gen=np.random.default_rng(57);n=500;w=gen.normal(size=n);z=w+gen.logistic(size=n);f=np.arange(n)%5
    pre=m.old.direct_w(w,z,f)
    out=m.rank_methods(w,z,f,pre,np.random.default_rng(58))
    point,se,meta=out["Rank-N"]
    assert_allclose(point,pre+sum(m.deltas(w,z,f,pre,"N"))/(np.sqrt(n)*meta["slope"]))
    ad,ase,a=out["Rank-Ad"];sl=np.asarray(a["fold_slopes"]);gamma=sl.sum(0)
    # Direct literal fold-by-fold formula, not the vectorized production expression.
    raw=[]
    for L in range(1,7):
        total=0.
        for k in range(5):
            for l in range(L):
                other=sum(sl[j,l] for j in range(5) if j!=k)
                total+=other/(1-np.mean(f==k))*sl[k,l]
        raw.append(total)
        delta=np.array([sum(m.deltas(w,z,f,pre,l)) for l in range(1,L+1)])
        assert_allclose(a["candidates"][L-1],pre+gamma[:L]@delta/(np.sqrt(n)*(gamma[:L]@gamma[:L])))
    assert_allclose(a["raw_corrected_information"],raw)
    corrected=np.maximum.accumulate(np.maximum(raw,.5*raw[0]))
    assert_allclose(a["corrected_information"],corrected)
    chosen=next(L for L in range(6) if 1/corrected[L]<=1.1/corrected[-1])
    assert a["selected_L"]==chosen+1
    assert_allclose(ase,np.sqrt(np.mean(w*w)/(n*corrected[chosen])))
    assert a["max_t_critical"]>norm.ppf(.975)
    # Validate the simulated joint Gaussian correlation independently.
    x=np.cumsum(gen.normal(size=(60000,6))*np.sqrt(np.diff(np.r_[0,corrected])),axis=1)/np.sqrt(corrected)
    expected=np.sqrt(np.minimum.outer(corrected,corrected)/np.maximum.outer(corrected,corrected))
    assert np.max(abs(np.corrcoef(x,rowvar=False)-expected))<.02
def kernel_quantiles():
    gen=np.random.default_rng(204);n=500;f=np.arange(n)%5;w=gen.normal(size=n);z=1.2*w+gen.normal(size=n)
    train=gen.normal(size=140);query=np.linspace(-2,2,19);h=.5
    fhat,der=m.density_kde(train,query,h,True)
    diff=(m.density_kde(train,query+1e-5,h)-m.density_kde(train,query-1e-5,h))/2e-5
    assert_allclose(der,diff,rtol=2e-8,atol=1e-10)
    for taus in ([.5],np.arange(1,10)/10):
        point,se,meta=m.quantile_fit(w,z,f,taus)
        shift=np.array([2,-.3,4,1,-7])
        p2,se2,_=m.quantile_fit(w,z+shift[f]+.3*w,f,taus)
        assert_allclose(p2,point+.3,atol=1e-7);assert_allclose(se2,se,rtol=1e-5)
        def loss(b):
            e=z-b*w;val=0
            for tau in taus:
                for k in range(5):
                    u=e[f==k]-np.quantile(e[f==k],tau)
                    val+=np.sum(np.maximum(tau*u,(tau-1)*u))
            return val
        assert loss(point)<=min(loss(point-.002),loss(point+.002))+1e-7
        assert np.isfinite(se) and se>0
    pre=m.old.direct_w(w,z,f);point,se,meta=m.bickel(w,z,f,pre)
    p2,s2,_=m.bickel(w,z+np.array([3,2,-4,5,1])[f],f,pre)
    assert_allclose([point,se],[p2,s2],atol=1e-10)
    # Literal cross-fold score, Fisher-style normalization, centered sandwich.
    e=z-pre*w
    e-=np.array([np.median(e[f==k]) for k in range(5)])[f]
    wc=w-np.array([w[f==k].mean() for k in range(5)])[f];phi=np.zeros(n)
    for k in range(5):
        tr=e[f!=k];ev=e[f==k];hh=meta["bandwidths"][k]
        a,b=m.density_kde(tr,ev,hh,True);keep=a>=1e-3/hh
        temp=np.zeros(len(ev));temp[keep]=-b[keep]/a[keep];phi[f==k]=temp
    A=np.mean(wc**2*phi**2)
    assert_allclose([point,se],[pre+np.mean(wc*phi)/A,np.std(wc*phi/A)/np.sqrt(n)])
def densities():
    inf=r.information()
    for law,want in (("gaussian",1.),("logistic",np.pi**2/9),("t3",2.),("cauchy",.5),("t15",2.5/4.5)):
        assert_allclose(inf[law]["fisher"],want,rtol=1e-6)
    assert_allclose(inf["gaussian"]["rank_w_variance"],np.pi/3,rtol=1e-6)
    assert_allclose(inf["gaussian"]["rank_n_variance"],1,rtol=1e-6)
    results=[]
    for law in m.LAWS:
        order=inf[law]["order"];x=np.linspace(-2.5,5,47)
        ff,score,_=m.density(law,x,order)
        fp=m.density(law,x+1e-5,order)[0];fm=m.density(law,x-1e-5,order)[0]
        ok=ff>1e-10
        err=np.max(abs(score[ok]+(fp[ok]-fm[ok])/(2e-5*ff[ok])))
        assert err<.003,(law,err)
        if law.startswith("sln"):
            other=m.density(law,x,6144)[1]
            score_err=np.max(abs(score[ok]-other[ok]))
            assert score_err<.005,(law,score_err)
        else:score_err=0
        results.append(dict(law=law,score_derivative_max_error=err,score_refinement_max_error=score_err,order=order))
    r.atomic(ROOT/"smoke/density_validation.json",results)
def outputs_reproducible():
    with sqlite3.connect(ROOT/"smoke/records.sqlite") as db:
        rows=[json.loads(x[0]) for x in db.execute("SELECT payload FROM records")]
        assert len(rows)==18
        assert db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
        assert len({(x["n"],x["law"],x["replication"]) for x in rows})==18
        for rec in rows:
            assert len(rec["rows"])==9 and all(row["valid_interval"] for row in rec["rows"])
            payload=db.execute("SELECT payload FROM cache_batches WHERE id=?",(rec["cache_batch"],)).fetchone()[0]
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:
                prefix=f'n{rec["n"]}_{rec["law"]}_r{rec["replication"]}'
                assert all(prefix+"__"+k in z for k in ("w","z","folds"))
        for law in ("gaussian","sln100_tau050"):
            rec=next(x for x in rows if x["law"]==law and x["replication"]==0)
            again,_=r.worker(("smoke",100,law,0,None,r.information()[law]))
            assert again["input_hash"]==rec["input_hash"]
            for a,b in zip(again["rows"],rec["rows"]):
                assert_allclose([a["estimate"],a["se"]],[b["estimate"],b["se"]],rtol=0,atol=1e-12)
                if a["method"]=="Rank-Ad":assert a["metadata"]["max_t_critical"]==b["metadata"]["max_t_critical"]
    import pandas as pd
    flat=pd.read_csv(ROOT/"汇总表_CSV/smoke/replications.csv")
    assert len(flat)==162 and len(flat.groupby(["n","law","rep"]))==18
    files=list((ROOT/"图/smoke").glob("*.pdf"))
    assert len(files)>=81 and all(p.read_bytes().startswith(b"%PDF") for p in files)
def historical():
    reuse=r.load_reuse();inf=r.information();results=[]
    for n,law in ((500,"gaussian"),(1000,"sln100_tau050"),(2000,"t3")):
        rec,cache=r.worker(("validation-reuse",n,law,0,reuse[(n,law,0)],inf[law]))
        assert "dataset_failure" not in rec
        assert len(rec["rows"])==9
        assert all(row["valid_estimator"] for row in rec["rows"])
        results.append(r.clean(rec))
    r.atomic(ROOT/"smoke/historical_second_stage_checks.json",results)
def failures():
    # Inject one family failure; remaining seven methods must be retained.
    cache,_=m.nuisance("failure-test",100,"gaussian",0)
    saved=m.old.regression
    def broken(*a,**k):raise RuntimeError("deliberate test failure")
    try:
        m.old.regression=broken
        rows=m.estimate_all(cache,"gaussian",m.rng("validation",100,"gaussian",0,"max"),r.information()["gaussian"])
        assert len(rows)==9 and all(not x["valid_interval"] for x in rows[:2])
        assert all(x["valid_estimator"] for x in rows[2:])
    finally:m.old.regression=saved
    saved=m.nuisance
    try:
        m.nuisance=broken
        rec,_=r.worker(("failure-test",100,"gaussian",0,None,r.information()["gaussian"]))
        assert len(rec["rows"])==9 and all(not x["valid_interval"] for x in rec["rows"])
        assert "dataset_failure" in rec
    finally:m.nuisance=saved
def notebook():
    import nbformat
    path=ROOT/"代码/Scenario_A_Spline.ipynb"
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=="code":compile(cell.source,str(path),"exec")
if __name__=="__main__":
    for name,fn in [
        ("Five-point sign and local sqrt(n) scale",finite_difference),
        ("Exact weighted-median W minimizes full pairwise objective",direct_minimum),
        ("Ties average scores; exact finite-grid centering",tie_scores),
        ("Rank-N sign; Rank-Ad point, correction, selection and max-t covariance",rank_equations),
        ("KDE derivative; LAD/CQR shared optimum and equivariance; Bickel sandwich",kernel_quantiles),
        ("Analytic information and smoothed-lognormal density/score precision",densities),
        ("18 smoke replications; CSV/PDF/SQLite round-trip; sequential versus process reproducibility",outputs_reproducible),
        ("Three audited historical caches execute new second stages",historical),
        ("Injected estimator and dataset failures are explicit and recoverable",failures),
        ("Notebook schema and executable cell syntax",notebook)]:
        check(name,fn)
    r.atomic(ROOT/"smoke/validation.json",dict(signature=r.signature(),status="PASS",checks=checks,
        note="Implementation checks and smoke only; neither pilot nor main has been run."))
    r.atomic(ROOT/"config/code_validation.json",dict(signature=r.signature(),status="PASS",purpose="Scientific implementation validation"))
    print("ALL IMPLEMENTATION CHECKS PASSED. No pilot/main started.",flush=True)


"""Scenario D estimators. The verified Scenario A functions supply common pieces."""
import dgp
import numpy as np
from scipy.stats import norm
import legacy_core as legacy
import methods as common
from time import perf_counter
import hashlib

Z95 = norm.ppf(.975)
RANKS = ("Rank-W","Rank-N","Rank-Ad","Odd-only","Fixed mixed")
AD_DRAWS = 50000

def rank_values(w,z,f,theta,kinds):
    e=z-theta*w
    mats=[]
    for kind in kinds:
        values=np.empty(len(f))
        for j in np.unique(f):
            mask=f==j
            values[mask]=common.score_grid(e[mask],kind)
        mats.append(values)
    return np.column_stack(mats)

def sandwich_meat(w,scores):
    return np.mean((w*scores)**2)

def interval(point,se,reference,failure=None,critical=Z95):
    valid=bool(failure is None and point is not None and np.isfinite(point) and
               se is not None and np.isfinite(se) and se>0 and np.isfinite(critical))
    out=dict(se=None,lower=None,upper=None,length=None,valid=False,covered=False,
             failure=failure or (None if valid else "invalid point, variance or interval"))
    if valid:
        lo,hi=float(point-critical*se),float(point+critical*se)
        out.update(se=float(se),lower=lo,upper=hi,length=hi-lo,valid=True,covered=bool(lo<=reference<=hi))
    return out

def regular_rank(name,w,z,f,pre,reference,sandwich):
    kinds={"Rank-W":("W",),"Rank-N":("N",),"Fixed mixed":(1,2)}[name]
    n=len(w)
    def statistic(theta):
        s=rank_values(w,z,f,theta,kinds).sum(1)
        return w@s/np.sqrt(n)
    gamma=float(common.fivepoint(statistic,pre,n))
    point=pre if name=="Rank-W" else None
    variance=sand=None
    reason=None
    try:
        common.positive(gamma,"nonpositive_rank_slope")
        point=pre if name=="Rank-W" else pre+statistic(pre)/(np.sqrt(n)*gamma)
        kappa2={"Rank-W":1/12,"Rank-N":1.,"Fixed mixed":2.}[name]
        variance=np.mean(w*w)*kappa2/gamma**2
        if sandwich:
            values=rank_values(w,z,f,point,kinds).sum(1)
            sand=sandwich_meat(w,values)/gamma**2
    except ValueError as exc:
        reason=str(exc)
    cis={"naive":interval(point,None if variance is None else np.sqrt(variance/n),reference,reason)}
    if sandwich:
        cis["sandwich"]=interval(point,None if sand is None else np.sqrt(sand/n),reference,reason)
    return point,cis,dict(slope=gamma,score_degrees=list(kinds))

def adaptive(w,z,f,pre,reference,kinds,rng,sandwich_rng=None):
    n=len(w)
    lam=np.bincount(f,minlength=5)/n
    def stats(theta):
        scores=rank_values(w,z,f,theta,kinds)
        return np.array([w[f==k]@scores[f==k]/np.sqrt(n) for k in range(5)])

    fold_slope=common.fivepoint(stats,pre,n)
    g=fold_slope.sum(0)
    G=np.cumsum(g*g)
    info=np.cumsum(np.sum(fold_slope*(g-fold_slope)/(1-lam[:,None]),axis=0))
    delta=stats(pre).sum(0)
    candidate=pre+np.divide(np.cumsum(g*delta),np.sqrt(n)*G,
                            out=np.full(len(G),np.nan),where=G>1e-12)
    meta=dict(degrees=list(kinds),slopes=g.tolist(),fold_slopes=fold_slope.tolist(),
              raw_information=G.tolist(),cross_fitted_information=info.tolist(),
              candidates=candidate.tolist())
    names=["naive","max_t"]
    if sandwich_rng is not None:
        names+=["sandwich","sandwich_max_t"]
    try:
        common.positive(float(info[0]),"nonpositive_corrected_I1")
        corrected=np.maximum.accumulate(np.maximum(info,.5*info[0]))
        selected=int(np.flatnonzero(corrected>=corrected[-1]/1.1)[0])
        point=float(candidate[selected])
        if not np.isfinite(point):
            raise ValueError("nonfinite adaptive candidate")
        selected_weights=g[:selected+1]
        even=np.array(kinds[:selected+1])%2==0
        even_share=float(np.sum(selected_weights[even]**2)/G[selected])
        even_increment=float(np.sum((g*delta)[:selected+1][even])/(np.sqrt(n)*G[selected]))
        variance=np.mean(w*w)/corrected
        se=float(np.sqrt(variance[selected]/n))
        increments=np.diff(np.r_[0,corrected])
        samples=np.cumsum(rng.normal(size=(AD_DRAWS,len(G)))*np.sqrt(increments),axis=1)
        critical=float(np.quantile(np.max(abs(samples/np.sqrt(corrected)),axis=1),.95))
        cis=dict(naive=interval(point,se,reference),max_t=interval(point,se,reference,critical=critical))
        meta.update(selected_L=int(kinds[selected]),selected_count=selected+1,
                    selected_weights=selected_weights.tolist(),even_weight_share=even_share,
                    even_one_step_increment=even_increment,corrected_information=corrected.tolist(),
                    max_t_critical=critical)
    except ValueError as exc:
        return None,{k:interval(None,None,reference,str(exc)) for k in names},meta

    if sandwich_rng is not None:
        try:
            # Apply the same correction G/I to the fixed-combination sandwich.
            contribution=np.column_stack([
                w*(rank_values(w,z,f,candidate[l],kinds[:l+1])@g[:l+1]) /
                np.sqrt(G[l]*corrected[l]) for l in range(len(G))
            ])
            cov=contribution.T@contribution/n
            s=np.sqrt(np.diag(cov))
            common.positive(np.min(s),"degenerate_sandwich_covariance")
            corr=cov/np.outer(s,s)
            eig,U=np.linalg.eigh((corr+corr.T)*.5)
            factor=U*np.sqrt(np.maximum(eig,0.))
            draws=sandwich_rng.normal(size=(AD_DRAWS,len(G)))@factor.T
            sandwich_critical=float(np.quantile(np.max(abs(draws),axis=1),.95))
            cis.update(sandwich=interval(point,s[selected]/np.sqrt(n),reference),
                       sandwich_max_t=interval(point,s[selected]/np.sqrt(n),reference,critical=sandwich_critical))
            meta.update(sandwich_max_t_critical=sandwich_critical)
        except (ValueError,np.linalg.LinAlgError) as exc:
            cis.update({k:interval(point,None,reference,str(exc)) for k in ("sandwich","sandwich_max_t")})
    return point,cis,meta

def huber_scales(data,mode,scenario,n,rep):
    f=data["folds"]
    y=np.column_stack([data["d"],data["y"]])
    scales=[]
    for k in range(5):
        train=f!=k
        inner=legacy.folds(int(train.sum()),3,dgp.rng(mode,scenario,n,rep,f"inner{k}"))
        if data["x"] is None:
            ww,zz=y[train].T
        else:
            residual,_=legacy.crossfit(data["x"][train],y[train],inner)
            ww,zz=residual.T
        pre=legacy.direct_w(ww,zz,inner)
        error=zz-pre*ww
        scales.append(1.482602218505602*np.median(abs(error-np.median(error))))
    return np.array(scales)

def variants(method,scenario):
    if method in ("Rank-Ad","Odd-only"):
        return ("naive","max_t","sandwich","sandwich_max_t") if scenario=="D1" else ("naive","max_t")
    if method in RANKS:
        return ("naive","sandwich") if scenario=="D1" else ("naive",)
    return ("wald",)

def method_names(sc):
    return dgp.METHODS+(dgp.D3_METHODS if sc=="D3" else ())

def one_rep(job):
    mode,scenario,n,rep=job
    reference=dgp.target(scenario)
    methods=method_names(scenario)
    outer_start=perf_counter()
    failure=None
    result=dict(mode=mode,scenario=scenario,n=n,rep=rep,theta0=reference,
                seeds={s:dgp.seed_key(mode,scenario,n,rep,s) for s in dgp.STREAMS})
    try:
        data=dgp.generate(*job)
        f=data["folds"]
        response=np.column_stack([data["d"],data["y"]])
        if data["x"] is None:
            residual=response
        else:
            residual,models=legacy.crossfit(data["x"],response,f)
            result["outer_penalties"]=[m.alpha.tolist() for m in models]
        if not np.isfinite(residual).all() or np.ptp(residual[:,0])<=1e-12:
            raise ValueError("nonfinite or degenerate nuisance residuals")
        w,z=residual.T
        digest=hashlib.sha256(residual.tobytes()+f.tobytes()).hexdigest()
        result.update(input_sha256=digest,fold_sizes=np.bincount(f).tolist())
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        failure=f"data or nuisance: {exc}"
    result["nuisance_seconds"]=perf_counter()-outer_start
    pre=None
    pre_failure=failure
    start=perf_counter()
    if pre_failure is None:
        try:
            pre=float(legacy.direct_w(w,z,f))
        except (ValueError,FloatingPointError,IndexError) as exc:
            pre_failure=f"Wilcoxon preliminary: {exc}"
    pre_time=perf_counter()-start
    result["methods"]=[]
    for name in methods:
        start=perf_counter()
        point=None
        cis={}
        meta={}
        try:
            if failure:
                raise ValueError(failure)
            if (name in RANKS or name=="Bickel") and pre is None:
                raise ValueError(pre_failure)
            if name in ("OLS","Huber"):
                c=None if name=="OLS" else 1.345*huber_scales(data,*job)[f]
                point,influence,meta=legacy.regression(w,z,f,c)
                se=np.sqrt(np.mean(influence**2)/n)
                cis["wald"]=interval(point,se,reference)
            elif name in ("LAD","CQR","Bickel"):
                if name=="Bickel":
                    point,se,meta=common.bickel(w,z,f,pre)
                else:
                    taus=[.5] if name=="LAD" else list(np.arange(1,10)/10)
                    point,se,meta=common.quantile_fit(w,z,f,taus)
                cis["wald"]=interval(point,se,reference)
            elif name in ("Rank-Ad","Odd-only"):
                kinds=(1,2,3,4,5,6) if name=="Rank-Ad" else (1,3,5)
                stream="rank_ad_max_t" if name=="Rank-Ad" else "odd_max_t"
                srng=dgp.rng(*job,"rank_ad_sandwich_max_t") if scenario=="D1" else None
                point,cis,meta=adaptive(w,z,f,pre,reference,kinds,dgp.rng(*job,stream),srng)
            else:
                point,cis,meta=regular_rank(name,w,z,f,pre,reference,scenario=="D1")
        except (ValueError,ArithmeticError,np.linalg.LinAlgError,RuntimeError) as exc:
            cis={v:interval(point,None,reference,f"{type(exc).__name__}: {exc}") for v in variants(name,scenario)}
        valid=bool(point is not None and np.isfinite(point))
        duration=perf_counter()-start+(pre_time if name=="Rank-W" else 0.)
        result["methods"].append(dict(name=name,point=float(point) if valid else None,
            valid_point=valid,intervals=cis,diagnostics=meta,seconds=duration))
    return result


"""Scenario A methods. Equations: Rank DML.tex Sections 2,3,6.
Numerical conventions are frozen in config/implementation_conventions.md.
"""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
import time
import hashlib
import json
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import ndtri, eval_legendre, expit
from scipy.stats import norm, t, logistic
from scipy.integrate import quad_vec
from scipy.special import roots_legendre as leggauss
import legacy_core as old

LAWS=("gaussian","logistic","t3","contaminated","bimodal","sln075","sln100_tau050","t15","cauchy")
METHODS=("OLS","Huber","LAD","CQR","Bickel","Rank-W","Rank-N","Rank-Ad","Oracle")
VERSION="scenario-a-20261118-v1"
def rng(stage,n,law,rep,stream):
    key=json.dumps([VERSION,20261118,stage,n,law,rep,stream])
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(hashlib.sha256(key.encode()).digest(),dtype="<u4")))

def error_sample(law,n,r):
    if law=="gaussian": return r.normal(size=n)
    if law=="logistic": return r.logistic(size=n)*np.sqrt(3)/np.pi
    if law=="t3": return r.standard_t(3,size=n)/np.sqrt(3)
    if law=="t15": return r.standard_t(1.5,size=n)
    if law=="cauchy": return r.standard_cauchy(size=n)
    if law=="contaminated": return r.normal(size=n)*np.where(r.random(n)<.1,5,1)/np.sqrt(3.4)
    if law=="bimodal": return (r.normal(scale=.5,size=n)+r.choice([-1.5,1.5],size=n))/np.sqrt(2.5)
    s,tau=(.75,.2) if law=="sln075" else (1.,.5)
    return (np.exp(s*r.normal(size=n))+tau*r.normal(size=n)-np.exp(s*s/2))/np.sqrt(np.exp(s*s)*np.expm1(s*s)+tau*tau)

def generate(stage,n,law,rep):
    r=rng(stage,n,law,rep,"data")
    x=old.covariates(n,r); m,ell,truth=old.truth(x,1,"scalar")
    v=r.normal(size=n); e=error_sample(law,n,r)
    return x,m+v,ell[:,0]+v+e,v,v+e,old.folds(n,5,rng(stage,n,law,rep,"outer"))

def nuisance(stage,n,law,rep):
    x,d,y,wo,zo,f=generate(stage,n,law,rep)
    start=time.perf_counter()
    res,models=old.crossfit(x,np.column_stack([d,y]),f)
    scales=[]; metadata=[]
    for k,mod in enumerate(models):
        train=f!=k
        inner=old.folds(sum(train),3,rng(stage,n,law,rep,f"inner-{k}"))
        rr,_=old.crossfit(x[train],np.column_stack([d[train],y[train]]),inner)
        pre=old.direct_w(rr[:,0],rr[:,1],inner)
        er=rr[:,1]-pre*rr[:,0]
        scales.append(1.482602218505602*np.median(abs(er-np.median(er))))
        metadata.append(dict(knots=[b.t.tolist() for _,_,b in mod.bases],mean=mod.mean.tolist(),
            sd=mod.sd.tolist(),outcome_mean=mod.ym.tolist(),coefficients=mod.beta.tolist(),alpha=mod.alpha.tolist(),gcv=mod.gcv))
    fresh=old.covariates(10000,rng(stage,n,law,rep,"diagnostic"))
    m,ell,_=old.truth(fresh,1,"scalar"); diag=[]
    for mod in models:
        pred=mod.predict(fresh); b=pred[:,0]-m; q=pred[:,1]-ell[:,0]; a=q-b
        diag.append(dict(mse_m=float(np.mean(b*b)),mse_ell=float(np.mean(q*q)),
            alignment_variance=float(np.var(a)),alignment_product=float(np.sqrt(np.mean(b*b)*np.var(a))),
            cov_b_a=float(np.mean((b-b.mean())*(a-a.mean())))))
    return dict(w=res[:,0],z=res[:,1],folds=f,w_oracle=wo,z_oracle=zo,scales=np.array(scales)),dict(
        nuisance_seconds=time.perf_counter()-start,models=metadata,diagnostics=diag,seed_namespace=VERSION)

def fivepoint(fun,theta,n,c=2.):
    # Half-width c: nodes -c,-c/2,+c/2,+c on the sqrt(n) local scale.
    h=c/2
    return (-fun(theta-2*h/np.sqrt(n))+8*fun(theta-h/np.sqrt(n))
            -8*fun(theta+h/np.sqrt(n))+fun(theta+2*h/np.sqrt(n)))/(12*h)

def score_grid(e,kind):
    m=len(e);u=np.arange(1,m+1)/(m+1)
    if kind=="W": grid=u-.5
    elif kind=="N": grid=ndtri(u)
    else: grid=np.sqrt(2*kind+1)*eval_legendre(kind,2*u-1)
    grid-=grid.mean(); order=np.argsort(e,kind="stable"); values=e[order]
    tol=64*np.finfo(float).eps*np.max(abs(values-np.median(values)))
    starts=np.r_[0,np.flatnonzero(np.diff(values)>tol)+1]; ends=np.r_[starts[1:],m]
    result=np.empty(m);result[order]=np.repeat(np.add.reduceat(grid,starts)/(ends-starts),ends-starts)
    return result

def deltas(w,z,f,theta,kind):
    return np.array([w[f==k]@score_grid((z-theta*w)[f==k],kind)/np.sqrt(len(w)) for k in range(5)])

def positive(x,label):
    if not np.isfinite(x) or x<=1e-12: raise ValueError(label)
    return x

def rank_methods(w,z,f,pre,r):
    n=len(w);s2=np.mean(w*w);out={}
    for name,kind,kappa in (("Rank-W","W",1/12),("Rank-N","N",1.)):
        started=time.perf_counter()
        g=np.sum(fivepoint(lambda th:deltas(w,z,f,th,kind),pre,n))
        # Retain valid point even if studentization fails.
        try:
            positive(g,"nonpositive_rank_slope")
            point=pre if kind=="W" else pre+sum(deltas(w,z,f,pre,kind))/(np.sqrt(n)*g)
            out[name]=(point,np.sqrt(s2*kappa/g**2/n),dict(slope=g,computation_seconds=time.perf_counter()-started))
        except ValueError as exc:
            out[name]=(pre if kind=="W" else None,None,dict(failure=str(exc),slope=g,computation_seconds=time.perf_counter()-started))
    started=time.perf_counter()
    slopes=np.column_stack([fivepoint(lambda th:deltas(w,z,f,th,l),pre,n) for l in range(1,7)])
    gamma=slopes.sum(0);G=np.cumsum(gamma**2)
    numer=np.cumsum(gamma*np.array([sum(deltas(w,z,f,pre,l)) for l in range(1,7)]))
    points=pre+np.divide(numer,np.sqrt(n)*G,out=np.full(6,np.nan),where=np.isfinite(G)&(G>1e-12))
    lamb=np.bincount(f,minlength=5)/n
    cross=(gamma[None,:]-slopes)/(1-lamb[:,None])
    info=np.cumsum(np.sum(cross*slopes,axis=0))
    meta=dict(candidates=points.tolist(),uncorrected_information=G.tolist(),raw_corrected_information=info.tolist(),
              fold_slopes=slopes.tolist())
    try:
        if not np.isfinite(points).all():raise ValueError("invalid_candidate_information")
        positive(info[0],"nonpositive_corrected_I1")
        corrected=np.maximum.accumulate(np.maximum(info,.5*info[0]))
        variances=s2/corrected
        selected=int(np.flatnonzero(variances<=1.1*variances[-1])[0])
        # Brownian increments give precisely sqrt(I_min/I_max) correlation, including ties.
        increments=np.diff(np.r_[0,corrected])
        draws=np.cumsum(r.normal(size=(50000,6))*np.sqrt(increments),axis=1)/np.sqrt(corrected)
        critical=float(np.quantile(np.max(abs(draws),axis=1),.95))
        se=np.sqrt(variances/n)
        meta.update(corrected_information=corrected.tolist(),candidate_se=se.tolist(),
                    selected_L=selected+1,max_t_critical=critical,max_t_halfwidth=critical*se[selected])
        out["Rank-Ad"]=(points[selected],se[selected],meta)
    except ValueError as exc:
        meta["failure"]=str(exc);out["Rank-Ad"]=(None,None,meta)
    out["Rank-Ad"][2]["computation_seconds"]=time.perf_counter()-started
    return out

def bandwidth(e,mult=1.):
    s=min(np.std(e),np.subtract(*np.percentile(e,[75,25]))/1.349)
    return positive(mult*1.06*s*len(e)**(-.2),"zero_kde_bandwidth")

def density_kde(train,query,h,derivative=False):
    d=(np.asarray(query)[:,None]-train[None,:])/h
    weights=norm.pdf(d)
    f=weights.mean(1)/h
    if derivative:return f,-(weights*d).mean(1)/h**2
    return f

def quantile_fit(w,z,f,taus):
    n=len(w);q=len(taus);p=1+5*q; N=n*q
    rows=np.arange(N); obs=np.tile(np.arange(n),q); lev=np.repeat(np.arange(q),n)
    design=sparse.coo_matrix((np.r_[w[obs],np.ones(N)],
        (np.r_[rows,rows],np.r_[np.zeros(N,int),1+lev*5+f[obs]])),shape=(N,p)).tocsr()
    A=sparse.hstack([design,sparse.eye(N),-sparse.eye(N)],format="csr")
    cost=np.r_[np.zeros(p),np.repeat(taus,n),np.repeat(1-np.array(taus),n)]
    fit=linprog(cost,A_eq=A,b_eq=z[obs],bounds=[(None,None)]*p+[(0,None)]*(2*N),
                method="highs",options={"dual_feasibility_tolerance":1e-8,"primal_feasibility_tolerance":1e-8,"threads":1,"parallel":False})
    if not fit.success: raise ValueError("quantile_optimizer:"+fit.message)
    beta=fit.x[:p];res=z[obs]-design@beta
    if np.max(abs(A@fit.x-z[obs]))>1e-6*(1+np.max(abs(z))):raise ValueError("quantile_primal_residual")
    bread=np.zeros((p,p));score=np.zeros((n,p)); loc=z-beta[0]*w
    for j,tau in enumerate(taus):
        for k in range(5):
            mask=f==k; e=loc[mask];h=bandwidth(e)
            dens=float(density_kde(e,np.array([beta[1+j*5+k]]),h)[0])
            vec=np.column_stack([w[mask],np.ones(sum(mask))]);idx=[0,1+j*5+k]
            bread[np.ix_(idx,idx)]+=dens*(vec.T@vec)/n
            # LP duals provide the valid subgradient at exact fitted quantiles;
            # strict residual signs at zero would make SE depend on roundoff.
            psi=fit.eqlin.marginals[j*n:(j+1)*n][mask]
            score[np.ix_(np.flatnonzero(mask),idx)]+=vec*psi[:,None]
    direction=np.linalg.solve(bread,np.eye(p)[0]);infl=score@direction;infl-=infl.mean()
    return beta[0],np.sqrt(np.mean(infl**2)/n),dict(quantiles=list(taus),intercepts=beta[1:].tolist(),optimizer=fit.message)

def bickel(w,z,f,pre):
    e=z-pre*w; centered=e.copy();wc=w.copy()
    for k in range(5):
        centered[f==k]-=np.median(e[f==k]);wc[f==k]-=w[f==k].mean()
    phi=np.empty(len(w));trim=0;hs=[]
    for k in range(5):
        train=centered[f!=k];ev=f==k;h=bandwidth(train,1.5);hs.append(h)
        dens,der=density_kde(train,centered[ev],h,True)
        # Fixed dimensionally consistent density cutoff; no true density.
        keep=dens>=1e-3/h;score=np.zeros(sum(ev));score[keep]=-der[keep]/dens[keep]
        phi[ev]=score;trim+=int(sum(~keep))
    A=positive(np.mean(wc**2*phi**2),"nonpositive_bickel_information")
    influence=wc*phi/A;point=pre+np.mean(influence)
    return point,np.std(influence)/np.sqrt(len(w)),dict(bandwidths=hs,trimmed=trim,information=A)

_QUAD={}
def density(law,e,order=192):
    e=np.atleast_1d(e).astype(float)
    if law=="gaussian":return norm.pdf(e),e,norm.cdf(e)
    if law=="logistic":
        s=np.sqrt(3)/np.pi;u=expit(e/s);return u*(1-u)/s,(2*u-1)/s,u
    if law in ("t3","t15","cauchy"):
        nu={"t3":3.,"t15":1.5,"cauchy":1.}[law];s=1/np.sqrt(3) if law=="t3" else 1.
        return t.pdf(e/s,nu)/s,(nu+1)*e/(nu*s*s+e*e),t.cdf(e/s,nu)
    if law in ("contaminated","bimodal"):
        means,sd,weight=([0,0],[1/np.sqrt(3.4),5/np.sqrt(3.4)],[.9,.1]) if law=="contaminated" else (
            [-1.5/np.sqrt(2.5),1.5/np.sqrt(2.5)],[.5/np.sqrt(2.5)]*2,[.5,.5])
        parts=np.array([wt*norm.pdf((e-m)/s)/s for m,s,wt in zip(means,sd,weight)])
        f=parts.sum(0);num=sum(a*(e-m)/s**2 for a,m,s in zip(parts,means,sd))
        return f,np.divide(num,f,out=np.zeros_like(f),where=f>1e-300),sum(wt*norm.cdf((e-m)/s) for m,s,wt in zip(means,sd,weight))
    if len(e)>128:
        blocks=[density(law,e[j:j+128],order) for j in range(0,len(e),128)]
        return tuple(np.concatenate([block[k] for block in blocks]) for k in range(3))
    sigma,tau=(.75,.2) if law=="sln075" else (1.,.5)
    scale=np.sqrt(np.exp(sigma*sigma)*np.expm1(sigma*sigma)+tau*tau);mu=np.exp(sigma*sigma/2)
    if order not in _QUAD:
        nodes,weights=leggauss(order);_QUAD[order]=(nodes*10,weights*10*norm.pdf(nodes*10))
    nodes,weights=_QUAD[order]
    u=e[:,None]*scale+mu-tau*nodes
    good=u>0; safe=np.maximum(u,1e-300);log=np.log(safe)
    pdf=np.where(good,norm.pdf(log/sigma)/(sigma*safe),0)
    slope=np.where(good,pdf*(1+log/sigma**2)/safe,0)
    raw=pdf@weights
    score=np.divide(slope@weights,raw,out=np.zeros_like(raw),where=raw>1e-300)*scale
    return raw*scale,score,np.where(good,norm.cdf(log/sigma),0)@weights

def theory(law,order=192,tol=2e-7):
    def integrand(x):
        f,phi,F=density(law,[x],order);ff=f[0];ph=phi[0];u=np.clip(F[0],1e-14,1-1e-14)
        return np.r_[ff,ff*ph*ph,ff*ff,ff*ph*ndtri(u),
            [ff*ph*np.sqrt(2*l+1)*eval_legendre(l,2*u-1) for l in range(1,7)]]
    value,error=quad_vec(integrand,-np.inf,np.inf,epsabs=tol,epsrel=tol,limit=500)
    fisher=positive(value[1],"fisher")
    return dict(law=law,mass=float(value[0]),fisher=float(fisher),efficient_variance=1/fisher,
                rank_w_variance=1/(12*value[2]**2),rank_n_variance=1/value[3]**2,
                rank_ad_variance=(1/np.cumsum(value[4:]**2)).tolist(),quadrature_error=float(error),order=order)

def estimate_all(cache,law,random,information):
    w,z,f=cache["w"],cache["z"].reshape(-1),cache["folds"];n=len(w);out=[]
    started=time.perf_counter();pre_error=None
    try:pre=old.direct_w(w,z,f)
    except Exception as exc:pre=None;pre_error=str(exc)
    pre_seconds=time.perf_counter()-started
    ranks=None
    for name in METHODS:
        start=time.perf_counter();meta={};point=se=None
        try:
            if name in ("Bickel","Rank-W","Rank-N","Rank-Ad") and pre is None:raise ValueError("preliminary:"+pre_error)
            if name in ("OLS","Huber"):
                point,influence,meta=old.regression(w,z,f,None if name=="OLS" else 1.345*cache["scales"][f].reshape(-1))
                se=np.sqrt(np.mean(influence**2)/n)
            elif name in ("LAD","CQR"):point,se,meta=quantile_fit(w,z,f,[.5] if name=="LAD" else np.arange(1,10)/10)
            elif name=="Bickel":point,se,meta=bickel(w,z,f,pre)
            elif name.startswith("Rank-"):
                if ranks is None:ranks=rank_methods(w,z,f,pre,random)
                point,se,meta=ranks[name]
            else:
                v=cache["w_oracle"];zz=cache["z_oracle"].reshape(-1)
                initial=old.direct_w(v,zz,f);dens,phi,_=density(law,zz-initial*v,information["order"])
                if law.startswith("sln") and (np.any(dens<=1e-300) or not np.isfinite(phi).all()):raise ValueError("oracle_density_underflow")
                info=information["fisher"]
                point=initial+np.mean(v*phi)/info
                se=np.sqrt(1/(n*info));meta=dict(preliminary=initial,fisher=info)
            valid_point=point is not None and np.isfinite(point)
            valid_ci=valid_point and se is not None and np.isfinite(se) and se>0
            if not valid_ci and "failure" not in meta:meta["failure"]="invalid_estimate_or_interval"
        except Exception as exc:
            valid_point=point is not None and np.isfinite(point);valid_ci=False;meta["failure"]=f"{type(exc).__name__}: {exc}"
        elapsed=meta.get("computation_seconds",time.perf_counter()-start)
        if name in ("Bickel","Rank-W","Rank-N","Rank-Ad"):
            meta["shared_preliminary_seconds"]=pre_seconds;elapsed+=pre_seconds
        out.append(dict(method=name,estimate=float(point) if valid_point else None,se=float(se) if valid_ci else None,
            valid_estimator=bool(valid_point),valid_interval=bool(valid_ci),covered=bool(abs(point-1)<=1.959963984540054*se) if valid_ci else False,
            length=float(2*1.959963984540054*se) if valid_ci else None,
            max_t_covered=bool(abs(point-1)<=meta["max_t_halfwidth"]) if valid_ci and "max_t_halfwidth" in meta else None,
            max_t_length=float(2*meta["max_t_halfwidth"]) if valid_ci and "max_t_halfwidth" in meta else None,
            seconds=elapsed,metadata=meta))
    return out



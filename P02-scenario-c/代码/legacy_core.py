"""Rank-DML, guide equations; all feasible routines take observed data only."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import hashlib, json, time
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh
from scipy.special import ndtri, gammaln
from scipy.stats import chi2

LAWS=('gaussian','t3','sln075','sln100')
METHODS=('LS','H','HT','W','N')
MULT=np.array([.3,.5,.75,1,1.345,2,3,5,10])

def rng(stage,cell,rep,stream):
    key=json.dumps([2026111401,stage,cell,rep,stream],sort_keys=True)
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:16],'little'))

def folds(n,k,r):
    f=np.empty(n,int); f[r.permutation(n)]=np.arange(n)%k; return f

def covariates(n,r):
    return r.normal(size=(n,10))@np.linalg.cholesky(.5**np.abs(np.subtract.outer(np.arange(10),np.arange(10)))).T

def truth(x,d,experiment):
    m=.6*np.sin(x[:,0])+.4*x[:,1]+.3*x[:,2]*x[:,3]
    theta=1/np.arange(1,d+1)
    if experiment=='scalar':
        g=(.7*np.cos(x[:,0])+.5*x[:,1]**2+.3*np.sin(x[:,2])+.25*x[:,3]*x[:,4])[:,None]
    else:
        j=np.arange(1,d+1)[None,:]
        g=.7/j*np.cos(x[:,0,None])+.3*np.sin(x[:,1,None]+j/3)+.25/j*(x[:,2]*x[:,3])[:,None]+.2*np.cos(x[:,4,None])
    return m,m[:,None]*theta+g,theta

def errors(n,d,rho,law,r):
    R=rho**np.abs(np.subtract.outer(np.arange(d),np.arange(d)))
    L=np.linalg.cholesky(R); g=r.normal(size=(n,d))@L.T
    if law=='gaussian': return g
    if law=='t3': return g/np.sqrt(r.chisquare(3,size=(n,1)))
    sigma=.75 if law=='sln075' else 1.
    return (np.exp(sigma*g)+.2*(r.normal(size=(n,d))@L.T)-np.exp(sigma*sigma/2))/np.sqrt(np.exp(sigma*sigma)*np.expm1(sigma*sigma)+.04)

def generate(cell,r):
    x=covariates(cell['n'],r);m,ell,t=truth(x,cell['d'],cell['experiment'])
    v=r.normal(size=len(x)); e=errors(len(x),cell['d'],cell['rho'],cell['law'],r)
    return x,m+v,ell+v[:,None]*t+e,v,v[:,None]*t+e,t

class Ridge:
    """Open clamped cubic basis: five knots INCLUDING endpoints, drop last basis.
    Linear continuation via boundary value and derivative; population training SD.
    """
    def features(self,x):
        blocks=[]
        for j,(lo,hi,b) in enumerate(self.bases):
            u=x[:,j]; clip=np.clip(u,lo,hi)
            a=b(clip)+(u-clip)[:,None]*b(clip,nu=1)
            blocks.append(a[:,:-1])
        blocks.append(np.column_stack([x[:,a]*x[:,b] for a in range(5) for b in range(a+1,5)]))
        return np.column_stack(blocks)
    def fit(self,x,y):
        self.bases=[]
        for j in range(10):
            lo,hi=x[:,j].min(),x[:,j].max()
            knots=np.r_[np.repeat(lo,3),np.linspace(lo,hi,5),np.repeat(hi,3)]
            self.bases.append((lo,hi,BSpline(knots,np.eye(len(knots)-4),3)))
        a=self.features(x); self.mean=a.mean(0);self.sd=a.std(0);self.sd[self.sd==0]=1
        a=(a-self.mean)/self.sd;self.ym=y.mean(0);yc=y-self.ym
        val,vec=eigh(a.T@a); val=np.maximum(val,0); proj=vec.T@(a.T@yc)
        best=np.full(y.shape[1],np.inf);self.beta=np.zeros((a.shape[1],y.shape[1]));self.alpha=np.zeros(y.shape[1])
        self.gcv=[]
        for alpha in (.1,1,10,100):
            beta=vec@(proj/(val[:,None]+alpha)); df=1+np.sum(val/(val+alpha))
            crit=np.mean((yc-a@beta)**2,0)/(1-df/len(x))**2
            take=crit<=best;self.beta[:,take]=beta[:,take];self.alpha[take]=alpha;best[take]=crit[take]
            self.gcv.append(crit.tolist())
        return self
    def predict(self,x):return (self.features(x)-self.mean)/self.sd@self.beta+self.ym

def crossfit(x,y,f):
    pred=np.empty_like(y); models=[]
    for k in range(f.max()+1):
        train=f!=k;mod=Ridge().fit(x[train],y[train]);pred[~train]=mod.predict(x[~train]);models.append(mod)
    return y-pred,models

def direct_w(w,z,f):
    ss=[];ww=[];n=len(w)
    for k in np.unique(f):
        ix=np.flatnonzero(f==k);a,b=np.triu_indices(len(ix),1);dw=w[ix[a]]-w[ix[b]]
        # Exact zero exclusion: no positive pair weights are approximated away.
        ok=dw!=0
        ss.append((z[ix[a[ok]]]-z[ix[b[ok]]])/dw[ok]);ww.append(abs(dw[ok])/(n*(len(ix)-1)))
    s=np.concatenate(ss);weight=np.concatenate(ww)
    if not len(s) or not np.isfinite(s).all() or weight.sum()<=0:raise ValueError('pair_weight')
    order=np.argsort(s,kind='stable');return s[order[np.searchsorted(np.cumsum(weight[order]),weight.sum()/2)]]

def scores(e,f,method):
    out=np.empty(len(e))
    for k in np.unique(f):
        ix=np.flatnonzero(f==k);order=np.argsort(e[ix],kind='stable');values=e[ix[order]];m=len(ix)
        grid=np.arange(1,m+1)/(m+1);grid=grid-.5 if method=='W' else ndtri(grid);grid-=grid.mean()
        # A pair attaining the W median has equal mathematical residuals.
        # Roundoff in z-t*w must not turn that tie into an arbitrary ordering.
        # Centered residual scale preserves fold translation and positive scaling.
        tie_tol=64*np.finfo(float).eps*np.max(abs(values-np.median(values)))
        starts=np.r_[0,np.flatnonzero(np.diff(values)>tie_tol)+1];ends=np.r_[starts[1:],m]
        tied=np.repeat(np.add.reduceat(grid,starts)/(ends-starts),ends-starts)
        out[ix[order]]=tied
    return out

def regression(w,z,f,c=None):
    q=np.column_stack([w,np.eye(f.max()+1)[f]]);n=len(w)
    beta=np.linalg.lstsq(q,z,rcond=None)[0];converged=c is None;iteration=0
    if c is not None:
        c=np.broadcast_to(c,(n,))
        if not np.isfinite(c).all() or (c<=0).any():raise ValueError('scale_or_threshold')
        for iteration in range(1,1001):
            u=z-q@beta;weight=np.minimum(1,c/np.maximum(abs(u),1e-12))
            new=np.linalg.solve(q.T@(q*weight[:,None]),q.T@(z*weight))
            change=np.max(abs(new-beta));beta=new
            if change<1e-8*(1+np.max(abs(beta))):converged=True;break
    u=z-q@beta;psi=u if c is None else np.clip(u,-c,c)
    score=np.max(abs(q.T@psi/n))
    if not converged or score>1e-6*(1+np.mean(abs(z))):raise ValueError('optimizer')
    a=q.T@q/n if c is None else q.T@(q*(abs(u)<c)[:,None])/n
    influence=(q@np.linalg.solve(a,np.eye(q.shape[1])[0]))*psi;influence-=influence.mean()
    return beta[0],influence,dict(iterations=iteration,score_residual=score,converged=converged)

def training_choices(x,d,y,f,stage,cell,rep):
    scales=np.full((5,y.shape[1]),np.nan);chosen=scales.copy();details=[];aux_time=0;tune_time=0
    for k in range(5):
        start=time.perf_counter();mask=f!=k;fi=folds(mask.sum(),3,rng(stage,cell,rep,'inner_'+str(k)))
        res,mods=crossfit(x[mask],np.column_stack([d[mask],y[mask]]),fi);w=res[:,0]
        row=dict(fold=k,inner_folds=fi.tolist(),penalties=[m.alpha.tolist() for m in mods],coordinates=[])
        for j in range(y.shape[1]):
            z=res[:,j+1];t=direct_w(w,z,fi);e=z-t*w;s=1.4826022185*np.median(abs(e-np.median(e)))
            scales[k,j]=s;row['coordinates'].append(dict(scale=s,preliminary=t,candidates=[]))
        aux_time+=time.perf_counter()-start;start=time.perf_counter()
        for j in range(y.shape[1]):
            best=np.inf
            for a in MULT:
                try:
                    _,eta,meta=regression(w,res[:,j+1],fi,a*scales[k,j]);v=np.mean(eta**2)
                    if not np.isfinite(v) or v<=0:raise ValueError('invalid_variance')
                    candidate=dict(multiplier=a,criterion=v,**meta)
                    if v<=best:best=v;chosen[k,j]=a
                except (ValueError,np.linalg.LinAlgError) as ex:candidate=dict(multiplier=a,criterion=None,converged=False,failure=str(ex))
                row['coordinates'][j]['candidates'].append(candidate)
        tune_time+=time.perf_counter()-start;details.append(row)
    return scales,chosen,details,aux_time,tune_time

def estimate(w,z,f,method,threshold=None,c=1.):
    n,d=z.shape;point=np.full(d,np.nan);V=np.full((d,d),np.nan);meta=[];fail=[]
    eta=np.full((n,d),np.nan);gamma=np.full(d,np.nan)
    for j in range(d):
        try:
            if method in ('LS','H','HT'):
                point[j],eta[:,j],detail=regression(w,z[:,j],f,None if method=='LS' else threshold[:,j]);meta.append(detail)
            else:
                t=direct_w(w,z[:,j],f)
                def delta(t):return w@scores(z[:,j]-t*w,f,method)/np.sqrt(n)
                gamma[j]=(delta(t-c/np.sqrt(n))-delta(t+c/np.sqrt(n)))/(2*c)
                if method=='W':point[j]=t
                if not np.isfinite(gamma[j]) or gamma[j]<=0:raise ValueError('rank_slope')
                if method=='N':point[j]=t+delta(t)/(np.sqrt(n)*gamma[j])
                eta[:,j]=scores(z[:,j]-point[j]*w,f,method)
                meta.append(dict(preliminary=t,gamma=gamma[j]))
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as ex:fail.append(dict(coordinate=j+1,type=str(ex)))
    if method in ('LS','H','HT'):V=eta.T@eta/n
    else:
        C=eta.T@eta/n if d>1 else np.array([[1/12 if method=='W' else 1.]])
        V=np.mean(w*w)*C/(gamma[:,None]*gamma[None,:]);V[(gamma<=0),:]=np.nan;V[:,gamma<=0]=np.nan
    return point,V,meta,fail

def inference(point,V,true,n):
    d=len(true);V=(V+V.T)/2;se=np.sqrt(np.diag(V)/n);valid=np.isfinite(point)&np.isfinite(se)&(se>0)
    lo=point-ndtri(.975)*se;hi=point+ndtri(.975)*se
    out=dict(se=se,valid_interval=valid,coverage=valid&(lo<=true)&(true<=hi),length=hi-lo,
             interval_score=hi-lo+40*np.maximum(lo-true,0)+40*np.maximum(true-hi,0),contrasts={})
    if d>1:
        for name,a in [('plus',np.ones(d)/np.sqrt(d)),('minus',np.r_[1,-1,np.zeros(d-2)]/np.sqrt(2))]:
            t=a@point;s=np.sqrt(a@V@a/n);target=a@true;length=2*ndtri(.975)*s;ok=bool(np.isfinite(s) and s>0 and np.isfinite(t))
            out['contrasts'][name]=dict(estimate=t,se=s,valid=ok,coverage=ok and abs(t-target)<=length/2,length=length,interval_score=length+40*max(target-t-length/2,0)+40*max(t-length/2-target,0))
        out.update(joint_valid=False,joint_coverage=False,volume=np.nan,log_volume=np.nan,min_eigenvalue=np.nan,condition=np.nan,all_marginal=bool(out['coverage'].all()))
        if np.isfinite(V).all():
            out['min_eigenvalue']=np.linalg.eigvalsh(V).min();out['condition']=np.linalg.cond(V)
            try:
                L=np.linalg.cholesky(V);quad=n*np.sum(np.linalg.solve(L,point-true)**2)
                logvol=d/2*np.log(np.pi)-gammaln(d/2+1)+d/2*np.log(chi2.ppf(.95,d))+np.log(np.diag(L)).sum()-d/2*np.log(n)
                out.update(joint_valid=bool(np.isfinite(quad)),joint_coverage=bool(quad<=chi2.ppf(.95,d)),volume=np.exp(logvol),log_volume=logvol)
            except np.linalg.LinAlgError:pass
    return out

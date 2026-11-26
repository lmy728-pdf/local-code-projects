"""Deterministic quadrature for the fixed-F and re-ranked population moments."""
import dgp
import numpy as np
from scipy.special import ndtr, ndtri, eval_legendre, roots_legendre
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.stats import gamma
from functools import lru_cache

@lru_cache(maxsize=8)
def nodes(order):
    return roots_legendre(order)

def cdf(x, theta, scenario):
    if scenario=="D2":
        plus = -np.expm1(-np.maximum(0.,x+1+theta))
        minus = np.exp(np.minimum(0.,x-1-theta))
    else:
        plus = ndtr((x+theta)/2)
        minus = ndtr((x-theta)/.5)
    return .5*(plus+minus)

def score(theta, scenario, order=512, fixed=False):
    """Order: W, N, then standardized Legendre degrees 1 through 6."""
    base, weight = nodes(order)
    total = np.zeros(8)
    for sign in (1,-1):
        if scenario=="D2":
            boundary=2+theta if fixed else 2+2*theta
            breaks=sorted({0.,40.,float(np.clip(boundary,0,40))})
        else:
            breaks=[-11.,-3.,0.,3.,11.]
        for a,b in zip(breaks,breaks[1:]):
            t=(a+b)/2+(b-a)/2*base
            if scenario=="D2":
                residual=sign*(t-1-theta)
                density=np.exp(-t)
            else:
                residual=(2 if sign==1 else .5)*t-sign*theta
                density=np.exp(-t*t/2)/np.sqrt(2*np.pi)
            u=cdf(residual,0. if fixed else theta,scenario)
            s=np.array([u-.5,ndtri(np.clip(u,1e-15,1-1e-15))]+
                    [np.sqrt(2*l+1)*eval_legendre(l,2*u-1) for l in range(1,7)])
            total+=sign*.5*(b-a)/2*(s @ (weight*density))
    return total

def slopes(theta, scenario, order, fixed=False, step=1e-4):
    fun=lambda x:score(x,scenario,order,fixed)
    return (-fun(theta-2*step)+8*fun(theta-step)-8*fun(theta+step)+fun(theta+2*step))/(12*step)

def mirror_baselines(w):
    """Population convex-loss slopes with the same common intercept convention."""
    mad=brentq(lambda q: cdf(q,w,"D2")-cdf(-q,w,"D2")-.5,0,8)
    threshold=1.345*1.482602218505602*mad
    def huber(theta):
        center=1+theta
        points=[x for x in (center-threshold,center+threshold) if 0<x<40]
        return quad(lambda z: np.clip(z-center,-threshold,threshold)*np.exp(-z),
                    0,40,points=points,epsabs=1e-12)[0]
    def cqr(theta):
        moments=[]
        for tau in np.arange(1,10)/10:
            a=brentq(lambda q: cdf(q,theta,"D2")-tau,-30,30)
            p=-np.expm1(-max(0.,a+1+theta))
            m=np.exp(min(0.,a-1-theta))
            moments.append((m-p)/2)
        return sum(moments)
    return dict(OLS=0.,Huber=float(brentq(huber,-1,1)),LAD=float(np.log(2)-1),
                CQR=float(brentq(cqr,-1,1)),Bickel=None)

def calculate(scenario, order, step):
    w = gamma.ppf(.5,a=2)/2-1 if scenario=="D2" else 0.
    values=score(w,scenario,order)
    derivative=slopes(w,scenario,order,step=step)
    g,delta=derivative[2:],values[2:]

    def adaptive(subset):
        h,z=g[subset],delta[subset]
        info=np.cumsum(h*h)
        selected=int(np.flatnonzero(info>=info[-1]/1.1)[0])+1
        return float(w+np.dot(h[:selected],z[:selected])/info[selected-1]),selected

    ad,selected=adaptive(np.arange(6))
    odd,odd_selected=adaptive(np.array([0,2,4]))
    targets=mirror_baselines(w) if scenario=="D2" else dict.fromkeys(dgp.METHODS[:5],0.)
    targets.update({"Rank-W":float(w),"Rank-N":float(w+values[1]/derivative[1]),"Rank-Ad":ad})
    if scenario=="D3":
        targets.update({"Odd-only":odd,"Fixed mixed":float(w+(delta[0]+delta[1])/(g[0]+g[1]))})
    roots={}
    score_columns={"Rank-W":[0],"Rank-N":[1]}
    if scenario=="D3":
        score_columns["Fixed mixed"]=[2,3]
    for name, columns in score_columns.items():
        roots[name]={}
        for fixed,label in ((False,"reranked"),(True,"fixed_F0")):
            fun=lambda x:score(x,scenario,order,fixed)[columns].sum()
            root=brentq(fun,-.8,.8,xtol=1e-13)
            roots[name][label]=dict(root=float(root),moment_at_root=float(fun(root)))
    return dict(targets=targets,rank_score_roots=roots,score_at_zero=score(0.,scenario,order).tolist(),
                slopes_at_W=g.tolist(),selected_L=selected,odd_selected_count=odd_selected,
                paper_wilcoxon_score_at_zero=float(2*score(0.,scenario,order)[0]))

def benchmarks():
    output={"D1":{"targets":dict.fromkeys(dgp.METHODS,1.)}}
    for sc in ("D2","D3"):
        high=calculate(sc,768,1e-4)
        low=calculate(sc,384,2e-4)
        flat=lambda obj: [*(v for v in obj["targets"].values() if v is not None),*obj["score_at_zero"],*obj["slopes_at_W"],
                          *(x["root"] for v in obj["rank_score_roots"].values() for x in v.values())]
        diff=float(np.max(np.abs(np.array(flat(high))-np.array(flat(low)))))
        assert diff<2e-6, (sc,diff)
        high["quadrature_max_difference"]=diff
        high["quadrature_orders"]=[384,768]
        output[sc]=high
    return output


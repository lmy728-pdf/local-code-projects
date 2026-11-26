"""Saved evidence for guide Section 10, isolated check seeds."""
from core import *
from run import save,ROOT,worker
import inspect

def checks():
    evidence={};r=np.random.default_rng(8126101)
    # T1: independent blocks quantify uncertainty including heavy-tail extremes.
    moments=[]
    for d in (1,2,3):
        for rho in ((0,) if d==1 else (0,.5)):
            R=rho**np.abs(np.subtract.outer(np.arange(d),np.arange(d)))
            for law in LAWS:
                means=[];covs=[];maxima=[]
                for _ in range(40):
                    e=errors(25000,d,rho,law,r);means.append(e.mean(0));covs.append(np.atleast_2d(np.cov(e,rowvar=False)));maxima.append(abs(e).max())
                target=R
                if law.startswith('sln'):
                    s=.75 if law=='sln075' else 1
                    target=(np.exp(s*s)*np.expm1(s*s*R)+.04*R)/(np.exp(s*s)*np.expm1(s*s)+.04)
                means=np.array(means);covs=np.array(covs)
                moments.append(dict(d=d,rho=rho,law=law,draws=1000000,mean=means.mean(0),mean_mcse=means.std(0,ddof=1)/np.sqrt(40),covariance=covs.mean(0),covariance_block_se=covs.std(0,ddof=1)/np.sqrt(40),target=target,max_abs=max(maxima)))
    evidence['T1']=dict(status='passed_with_reported_sampling_uncertainty',moments=moments,warning='t3 fourth moment infinite: block covariance SE is descriptive, not a Gaussian precision guarantee; lognormal extremes retained')
    cell=dict(experiment='scalar',n=100,law='gaussian',d=1,rho=0)
    x,D,Y,wo,zo,t=generate(cell,r);f=folds(100,5,r);train=f!=0;y=np.column_stack([D,Y])
    first=Ridge().fit(x[train],y[train]);alter=x.copy();alter[~train]+=1000;ya=y.copy();ya[~train]-=999
    second=Ridge().fit(alter[train],ya[train]);assert np.array_equal(first.beta,second.beta);assert np.array_equal(first.alpha,second.alpha)
    # Validation perturbation leaves all inputs to this fold's auxiliary choice unchanged.
    fi=folds(train.sum(),3,r);inner=[]
    for k in range(3):
        tr=fi!=k;xx=x[train].copy();yy=y[train].copy();a=Ridge().fit(xx[tr],yy[tr]);xx[~tr]+=100;yy[~tr]-=100;b=Ridge().fit(xx[tr],yy[tr]);assert np.array_equal(a.beta,b.beta);inner.append(True)
    assert np.array_equal(x[train],alter[train]) and np.array_equal(y[train],ya[train])
    evidence['T2']=dict(status='passed',outer_basis_beta_penalty_unchanged=True,inner_checks=inner,training_choice_input_unchanged=True,source=inspect.getsource(training_choices))
    w=wo;z=zo[:,0];tw=direct_w(w,z,f)
    def objective(t):
        val=0
        for k in np.unique(f):
            ix=np.flatnonzero(f==k);a,b=np.triu_indices(len(ix),1);val+=np.sum(abs((z[ix[a]]-z[ix[b]])-t*(w[ix[a]]-w[ix[b]])))/(len(w)*(len(ix)-1))
        return val
    test=np.r_[tw,tw+np.array([-1,-.1,-.001,.001,.1,1])];values=np.array([objective(a) for a in test]);assert values[0]<=values[1:].min()+1e-12
    tie=scores(np.array([0.,0.,1.,2.]),np.zeros(4,int),'N');grid=ndtri(np.arange(1,5)/5);grid-=grid.mean();assert np.isclose(tie[0],grid[:2].mean());assert abs(tie.sum())<1e-12
    evidence['T3']=dict(status='passed',slopes=test,objectives=values,tie_scores=tie)
    translations=[];scaling=[];shift=np.arange(5)[f]*1.3
    for method in METHODS:
        c=np.ones((100,1))*1.345
        a,V,_,fa=estimate(w,zo,f,method,c);b,Vb,_,fb=estimate(w,zo+shift[:,None],f,method,c)
        assert not fa and not fb;assert np.allclose(a,b,atol=1e-7);translations.append(dict(method=method,original=a,shifted=b))
        b,Vb,_,fb=estimate(w,zo*2.7,f,method,c*2.7,c=2.7);assert not fb;assert np.allclose(b,2.7*a,atol=1e-7);assert np.allclose(Vb,2.7**2*V,rtol=1e-6);scaling.append(dict(method=method,point_error=max(abs(b-2.7*a)),covariance_error=abs(Vb-2.7**2*V).max()))
    evidence['T4']=dict(status='passed',results=translations);evidence['T6']=dict(status='passed',results=scaling)
    delta=lambda t:w@scores(z-t*w,f,'N')/10
    gamma=(delta(tw-.1)-delta(tw+.1))/2
    point,_,_,_=estimate(w,zo,f,'N');assert gamma>0;assert np.isclose(point[0],tw+delta(tw)/(10*gamma))
    evidence['T5']=dict(status='passed',left=delta(tw-.1),right=delta(tw+.1),gamma=gamma,one_step=point)
    point,V,_,_=estimate(w,zo,f,'LS');q=np.column_stack([w,np.eye(5)[f]]);beta=np.linalg.lstsq(q,z,rcond=None)[0];u=z-q@beta;inv=np.linalg.inv(q.T@q);hc=inv@(q.T@(q*u[:,None]**2))@inv
    assert np.isclose(V[0,0]/100,hc[0,0]);evidence['T7']=dict(status='passed',variance=V[0,0]/100,hc0=hc[0,0])
    oracle=[]
    for n in (500,1000,2000):
        pts={m:[] for m in ('LS','W','N')};covs={m:[] for m in pts}
        for rep in range(200):
            w=r.normal(size=n);e=errors(n,3,.5,'gaussian',r);true=1/np.arange(1,4);z=w[:,None]*true+e;f=folds(n,5,r)
            for method in pts:
                p,V,_,fail=estimate(w,z,f,method);assert not fail;pts[method].append(p);covs[method].append(V)
        for method in pts:
            pp=np.array(pts[method]);oracle.append(dict(n=n,method=method,R=200,bias=pp.mean(0)-true,bias_mcse=pp.std(0,ddof=1)/np.sqrt(200),empirical_root_n_covariance=n*np.cov(pp,rowvar=False),mean_estimated_root_n_covariance=np.mean(covs[method],0)))
    evidence['T8']=dict(status='diagnostic_review_required',vector_gaussian=oracle,benchmarks='LS,N diagonal 1; W diagonal pi/3; N full matrix R; 200 runs finite MC uncertainty')
    worker(('checks',cell,0,str(ROOT/'checks_run')))
    import json
    data=json.loads(next((ROOT/'checks_run'/'records').glob('*.json')).read_text(encoding='utf-8'));assert 'traceback' not in data
    for pipeline in ('learned','oracle'):
        rows=[a for a in data['records'] if a['pipeline']==pipeline];assert len(rows)==5;assert len(set(a['input_hash'] for a in rows))==1
    p,V,_,fail=estimate(np.zeros(100),np.ones((100,1)),np.arange(100)%5,'W');assert fail;inf=inference(p,V,np.ones(1),100);assert not inf['coverage'][0]
    evidence['T9']=dict(status='passed',shared_hashes=True,failed_fit_retained=True,failed_interval_noncoverage=True,record_count=len(data['records']))
    save(ROOT/'reports'/'implementation_checks.json',evidence)
    np.savez_compressed(ROOT/'reports'/'check_inputs.npz',x=x,D=D,Y=Y,outer_folds=np.arange(100)%5)
    print('T1-T7,T9 passed; T8 saved for Monte Carlo review',flush=True)

if __name__=='__main__':checks()

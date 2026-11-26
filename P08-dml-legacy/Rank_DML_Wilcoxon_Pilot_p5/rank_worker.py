# -*- coding: utf-8 -*-
"""独立 Wilcoxon Rank-DML；只拟合新方法，旧 Worker 仅调用数据和 Spline nuisance 函数。"""
import os,sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import ast,hashlib,html,importlib.util,json,math,multiprocessing,sqlite3,time,traceback
from pathlib import Path
from datetime import datetime
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits,threadpool_info
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
TROOT=BASE/'DML_SplineRidge Student-t全量实验'/'脚本'/'Spline_Ridge_DML_Full_Results'
LROOT=BASE/'DML_SplineRidge Lognormal全量实验'/'脚本'/'Spline_Ridge_DML_Lognormal_Results'
SOURCES={0:TROOT/'spline_full_worker.py',7:TROOT/'spline_full_worker.py',14:LROOT/'spline_lognormal_worker.py'}
ERRORS={0:'Gaussian',7:'t(2.1)',14:'LN(2)'}
METHODS=['M1','M2','M3','S','Rank']
MODULES={}
REFS=None

def sha(data):return hashlib.sha256(data).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def atomic_csv(path,frame):
    temp=Path(str(path)+'.tmp');frame.to_csv(temp,index=False,encoding='utf-8-sig');os.replace(temp,path)
def log(where,text):
    with (where/'run.log').open('a',encoding='utf-8') as f:
        f.write(datetime.now().isoformat(timespec='seconds')+' '+text+'\n')
def module(error):
    path=SOURCES[error]
    if str(path) not in MODULES:
        name='historical_rank_source_'+str(error)
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
        MODULES[str(path)]=mod
    return MODULES[str(path)]
def ranks(e):
    e=np.asarray(e,dtype=float)
    if e.ndim!=1 or len(e)<3 or not np.isfinite(e).all():raise ValueError('无效残差')
    return rankdata(e,method='average')
def rank_score(w,e):
    a=ranks(e)/(len(e)+1)-.5
    return float(np.mean(w*a))
def dispersion(u,w,theta):
    e=np.sort(u-theta*w)
    return float(np.dot(2*np.arange(1,len(e)+1)-len(e)-1,e))
def rank_point(u,w):
    """q=1 凸目标的单调次梯度二分；每次 O(n log n)，不构造 n×n 距离矩阵。"""
    u=np.asarray(u,dtype=float);w=np.asarray(w,dtype=float)
    if not np.isfinite(np.r_[u,w]).all() or np.var(w)<=1e-14:raise ValueError('处理残差退化')
    wc=w-w.mean()
    anchor=float(np.dot(wc,u-u.mean())/np.dot(wc,wc)) # 仅数据驱动 OLS 初值
    width=max(.05,float(np.std(u)/np.std(w)))
    evaluations=0
    def g(t):
        nonlocal evaluations
        evaluations+=1
        return rank_score(w,u-t*w)
    lo,hi=anchor-width,anchor+width
    for _ in range(60):
        gl,gh=g(lo),g(hi)
        if gl>=0 and gh<=0:break
        width*=2;lo,hi=anchor-width,anchor+width
    else:raise ArithmeticError('凸目标未能括住最优点')
    for _ in range(80):
        mid=(lo+hi)/2;gm=g(mid)
        if gm==0:lo=hi=mid;gl=gh=0;break
        if gm>0:lo=mid;gl=gm
        else:hi=mid;gh=gm
        if hi-lo<=1e-7*(1+abs(mid)):break
    else:raise ArithmeticError('次梯度二分未收敛')
    theta=(lo+hi)/2
    assert gl>=0 and gh<=0
    gap=2*len(w)*(len(w)+1)*max(abs(gl),abs(gh))*(hi-lo)
    return dict(theta_hat=theta,optimizer_evaluations=evaluations,bracket_width=hi-lo,
                score_left=gl,score_right=gh,objective=dispersion(u,w,theta),
                objective_gap_bound=gap,point_score=rank_score(w,u-theta*w),optimizer_converged=True)
def rank_test(u,w,hypothesis):
    """原假设处重新排名。方差是秩得分理论基准，不使用 Huber 或模拟真值校准。"""
    e=u-hypothesis*w
    g=rank_score(w,e)
    variance=float(np.mean((w-w.mean())**2)/12)
    if not np.isfinite(variance) or variance<=0:raise ArithmeticError('检验方差无效')
    t=math.sqrt(len(w))*g/math.sqrt(variance)
    return dict(T=t,score_mean=g,sigma2=variance,accepted=abs(t)<=1.96,
                ties=len(e)-len(np.unique(e)),w_mean=float(w.mean()))
def init_worker():
    global REFS,_thread_guard
    _thread_guard=threadpool_limits(limits=1)
    REFS=json.loads((ROOT/'historical_reference.json').read_text(encoding='utf-8'))
def run_rep(task):
    d,e,r=task;start=time.perf_counter()
    row=dict(dgp_id=d,error_id=e,error=ERRORS[e],replication_id=r,method='Rank',
        theta_hat=None,T=None,point_status='failed',test_status='failed',nuisance_status='failed',
        failures=[],pid=os.getpid(),source='Pilot, R=200')
    try:
        mod=module(e);cfg=mod.config_for('full') # 保留历史 mode/seed 规则，不把 Pilot R 写入生成器
        # p5 Lognormal 原 Worker 在 generate() 中分派到 generate_pilot_observed。
        observed,folds,fold_ids,audit=mod.generate(cfg,task)
        X,D,Y=observed
        assert X.shape==(1000,5)
        ref=REFS[f'{d},{e},{r}']
        for key in ('data_hash','fold_hash'):assert audit[key]==ref[key],key
        for key,value in audit['seeds'].items():assert value==ref['seed_'+key]
        row.update(audit)
        prediction,alphas=mod.spline_oof(observed,folds,cfg,task)
        ph=mod.sha(prediction.tobytes())
        assert ph==ref['prediction_hash'],'Spline OOF 预测与历史 S 不一致'
        row.update(prediction_hash=ph,alpha_d=[x[0] for x in alphas],alpha_y=[x[1] for x in alphas],
            nuisance_status='ok',matched_historical=True,**mod.prediction_errors(observed,d,*prediction))
        w=D-prediction[0];u=Y-prediction[1]
    except Exception:
        row['failures'].append(dict(stage='nuisance_or_replay',error=traceback.format_exc()))
        row['seconds']=time.perf_counter()-start;return row
    try:
        row.update(rank_point(u,w));row['point_status']='ok'
    except Exception:row['failures'].append(dict(stage='point',error=traceback.format_exc()))
    try:
        row.update(rank_test(u,w,1.0));row['test_status']='ok'
    except Exception:row['failures'].append(dict(stage='test',error=traceback.format_exc()))
    row['seconds']=time.perf_counter()-start
    row['numeric_threads']=[x['num_threads'] for x in threadpool_info()]
    return row

def history():
    cols=['dgp_id','error_id','replication_id','method','theta_hat','T','data_hash','fold_hash',
          'prediction_hash','seed_X','seed_v','seed_epsilon','seed_folds']
    raw=[];tables=[]
    for path,errors in ((TROOT,[0,7]),(LROOT,[14])):
        a=pd.read_csv(path/'full_replication_results.csv',usecols=cols)
        raw.append(a[a.error_id.isin(errors)])
        a=pd.read_csv(path/'full_summary.csv');tables.append(a[a.error_id.isin(errors)])
    return pd.concat(raw,ignore_index=True),pd.concat(tables,ignore_index=True)
def prepare():
    ROOT.mkdir(exist_ok=True)
    raw,summary=history()
    references={}
    for row in raw[(raw.method=='S') & (raw.replication_id<200)].to_dict('records'):
        key=f"{row['dgp_id']},{row['error_id']},{row['replication_id']}"
        references[key]={k:row[k] for k in ('data_hash','fold_hash','prediction_hash','seed_X','seed_v','seed_epsilon','seed_folds')}
    assert len(references)==1200
    protected={}
    for root in (TROOT,LROOT):
        for name in ('configuration.json','full_summary.csv','full_replication_results.csv'):
            p=root/name;protected[str(p)]=sha(p.read_bytes())
    for p in set(SOURCES.values()):protected[str(p)]=sha(p.read_bytes())
    for name in ('rank_worker.py','summary.csv','configuration.json'):
        p=BASE/'Rank_DML_Wilcoxon_Pilot_p10'/name;protected[str(p)]=sha(p.read_bytes())
    config=dict(n=1000,p=5,q=1,R=200,errors=ERRORS,workers=8,threads=1,outer_folds=5,
        variance='mean((w-mean(w))**2)/12',critical=1.96,optimizer='convex monotone subgradient bisection',
        theta_tolerance=1e-7,seed_mode='historical full; original error ids 0,7,14',
        source_hashes=protected,worker_hash=sha(Path(__file__).read_bytes()))
    config['config_hash']=sha(json.dumps(config,sort_keys=True).encode())
    configfile=ROOT/'configuration.json'
    if configfile.exists():assert json.loads(configfile.read_text(encoding='utf-8'))==config,'配置与已有运行不同'
    else:atomic_json(configfile,config)
    if not (ROOT/'historical_reference.json').exists():atomic_json(ROOT/'historical_reference.json',references)
    return config
def checks():
    rng=np.random.default_rng(20261110)
    tied=np.array([3.,1.,1.,5.,2.]);a=ranks(tied)
    assert np.array_equal(a,[4,1.5,1.5,5,3]) and a.sum()==15
    assert np.array_equal(a,ranks(tied+7))
    for n in (5,12,25):
        for _ in range(4):
            w=rng.normal(size=n);u=.7*w+rng.normal(size=n)
            point=rank_point(u,w);shifted=rank_point(u+12.5,w)
            assert abs(point['theta_hat']-shifted['theta_hat'])<1e-6
            assert np.isclose(rank_score(w,u),rank_score(w,u+12.5))
            i,j=np.triu_indices(n,1)
            slopes=(u[i]-u[j])/(w[i]-w[j]);weights=abs(w[i]-w[j])
            order=np.argsort(slopes);exact=slopes[order][np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)]
            direct=lambda t:float(np.abs((u-t*w)[i]-(u-t*w)[j]).sum())
            assert np.isclose(dispersion(u,w,.23),direct(.23),rtol=1e-12)
            assert direct(point['theta_hat'])-direct(exact)<=point['objective_gap_bound']+1e-7
            h=1e-6
            deriv=(direct(.23+h)-direct(.23-h))/(2*h)
            assert np.isclose(deriv,-2*n*(n+1)*rank_score(w,u-.23*w),rtol=1e-5,atol=1e-6)
            assert not np.array_equal(ranks(u-.23*w),ranks(u-4*w))
    # 独立 Gaussian 数值自检，40 个小样本，不是正式比较方法，不调参。
    Ts=[];Gs=[]
    for _ in range(40):
        X=rng.multivariate_normal(np.zeros(5),module(0).ar1_covariance(5),400)
        m,g=module(0).nuisance_functions(X,module(0).DGP_NAMES[1])
        v=rng.normal(size=400);eps=rng.normal(size=400)
        D=m+v;Y=D+g+eps
        out=rank_test(Y-(m+g),D-m,1.)
        Ts.append(out['T']);Gs.append(out['score_mean'])
    assert abs(np.mean(Ts))<.8 and .5<np.std(Ts,ddof=1)<1.7,'独立 Gaussian 自检尺度异常'
    mod=module(0);X=rng.normal(size=(30,5))
    assert all(np.array_equal(a,b) for a,b in zip(mod.nuisance_functions(X,mod.DGP_NAMES[1]),module(14).nuisance_functions(X,mod.DGP_NAMES[1])))
    f,_=mod.spline_features(X[:20],X[20:]);assert f.shape[1]==45
    # 两份源码的第一阶段函数、超参数一致；直接调用，不复制改写。
    left=module(0);right=module(14)
    for name in ('spline_features','ridge_cv_alphas','spline_oof','seed','nuisance_functions'):
        import inspect
        assert ast.dump(ast.parse(inspect.getsource(getattr(left,name))))==ast.dump(ast.parse(inspect.getsource(getattr(right,name))))
    assert np.array_equal(left.ALPHAS,right.ALPHAS)
    return dict(passed=True,rank_ties=True,shift_invariance=True,direct_pairwise_objective=True,
        convex_optimality_certificate=True,candidate_reranking=True,gaussian_validation_R=40,
        oracle_mean_G=float(np.mean(Gs)),oracle_mean_T=float(np.mean(Ts)),oracle_sd_T=float(np.std(Ts,ddof=1)),
        spline_features=45,source_functions_identical=True)

def database(where,config):
    (where/'checkpoint').mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(where/'checkpoint'/'replications.sqlite3')
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('CREATE TABLE IF NOT EXISTS manifest(hash TEXT PRIMARY KEY); CREATE TABLE IF NOT EXISTS results(d INTEGER,e INTEGER,r INTEGER,payload TEXT,PRIMARY KEY(d,e,r));')
    old=db.execute('SELECT hash FROM manifest').fetchall()
    if old:assert old==[(config['config_hash'],)]
    else:
        db.execute('INSERT INTO manifest VALUES(?)',(config['config_hash'],));db.commit()
    return db
def collect(where,config,tasks):
    where.mkdir(exist_ok=True)
    with closing(database(where,config)) as db:
        done={tuple(x) for x in db.execute('SELECT d,e,r FROM results')}
        assert done<=set(tasks)
        todo=iter([t for t in tasks if t not in done])
        futures={}
        def submit(pool):
            task=next(todo,None)
            if task is not None:futures[pool.submit(run_rep,task)]=task
        with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool:
            for _ in range(24):submit(pool)
            with tqdm(total=len(tasks),initial=len(done),desc='Wilcoxon Rank-DML',unit='组',mininterval=15) as bar:
                while futures:
                    ready,_=wait(futures,return_when=FIRST_COMPLETED)
                    for future in ready:
                        task=futures.pop(future);row=future.result()
                        with db:db.execute('INSERT INTO results VALUES(?,?,?,?)',(*task,json.dumps(row,ensure_ascii=False,allow_nan=False)))
                        done.add(task);bar.update()
                        if len(done)%40==0 or len(done)==len(tasks):
                            atomic_json(where/'progress.json',dict(completed=len(done),total=len(tasks),status='running',updated=datetime.now().isoformat()))
                        submit(pool)
        return [json.loads(x[0]) for x in db.execute('SELECT payload FROM results ORDER BY d,e,r')]

def summarize(rows):
    raw=pd.DataFrame(rows);out=[]
    for (d,e),a in raw.groupby(['dgp_id','error_id']):
        t=a['T'].dropna();delta=a.theta_hat.dropna()-1
        coverage=float((abs(t)<=1.96).mean())
        out.append(dict(dgp_id=d,error_id=e,error=ERRORS[e],method='Rank',source='Pilot, R=200',planned_n=len(a),
            point_n=len(delta),test_n=len(t),point_fail=len(a)-len(delta),test_fail=len(a)-len(t),
            point_failure_rate=1-len(delta)/len(a),test_failure_rate=1-len(t)/len(a),
            coverage_pct=100*coverage,coverage_MCSE_pp=100*math.sqrt(coverage*(1-coverage)/len(t)) if len(t) else np.nan,
            bias=delta.mean(),rmse=np.sqrt(np.mean(delta**2)),mean_T=t.mean(),sd_T=t.std(ddof=1),
            left_reject_pct=100*(t<-1.96).mean(),right_reject_pct=100*(t>1.96).mean(),
            y_nuisance_mse=a.get('y_nuisance_mse',pd.Series(dtype=float)).mean(),
            d_nuisance_mse=a.get('d_nuisance_mse',pd.Series(dtype=float)).mean()))
    summary=pd.DataFrame(out)
    base=summary[summary.error_id.eq(0)].set_index('dgp_id').coverage_pct
    summary['deterioration_pp']=[base[d]-c for d,c in zip(summary.dgp_id,summary.coverage_pct)]
    return raw,summary
def export(rows,config):
    raw,summary=summarize(rows)
    failures=[dict(dgp_id=r['dgp_id'],error_id=r['error_id'],replication_id=r['replication_id'],**f) for r in rows for f in r['failures']]
    failure=pd.DataFrame(failures,columns=['dgp_id','error_id','replication_id','stage','error'])
    flat=raw.copy()
    for col in flat.columns:
        if flat[col].map(lambda x:isinstance(x,(dict,list))).any():flat[col]=flat[col].map(lambda x:json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else x)
    atomic_csv(ROOT/'replication_results.csv',flat);atomic_csv(ROOT/'summary.csv',summary);atomic_csv(ROOT/'failures.csv',failure)
    old,old_summary=history()
    old_summary['source']='Historical Full, R=1000'
    old_summary['coverage_MCSE_pp']=100*np.sqrt((old_summary.coverage_pct/100)*(1-old_summary.coverage_pct/100)/old_summary.test_n)
    comparison=pd.concat([old_summary,summary],ignore_index=True,sort=False)
    paired=[]
    for (d,e),new in raw.groupby(['dgp_id','error_id']):
        for method in ['M1','M2','M3','S']:
            ref=old[(old.dgp_id==d)&(old.error_id==e)&(old.method==method)]
            a=new.merge(ref,on=['dgp_id','error_id','replication_id'],suffixes=('_rank','_old'),validate='one_to_one')
            matched=len(a)==200 and all(a[k+'_rank'].eq(a[k+'_old']).all() for k in ['data_hash','fold_hash'])
            assert matched,'不能验证配对数据'
            test=a[['T_rank','T_old']].dropna();point=a[['theta_hat_rank','theta_hat_old']].dropna()-1
            dif=100*((abs(test.T_rank)<=1.96).astype(int)-(abs(test.T_old)<=1.96).astype(int))
            rm=np.sqrt((point**2).mean())
            paired.append(dict(dgp_id=d,error_id=e,error=ERRORS[e],reference=method,test_pairs=len(test),point_pairs=len(point),
                coverage_diff_pp=dif.mean(),coverage_diff_MCSE_pp=dif.std(ddof=1)/np.sqrt(len(dif)),
                bias_diff=point.theta_hat_rank.mean()-point.theta_hat_old.mean(),
                rmse_diff=rm.theta_hat_rank-rm.theta_hat_old,rmse_ratio=rm.theta_hat_rank/rm.theta_hat_old,
                mean_T_rank_paired=test.T_rank.mean(),mean_T_old_paired=test.T_old.mean(),
                reference_source='Historical matched replications 0–199; not Full R1000 summary'))
    paired=pd.DataFrame(paired)
    atomic_csv(ROOT/'comparison_historical.csv',comparison);atomic_csv(ROOT/'paired_comparisons.csv',paired)
    m=module(0);m.configure_plot_style()
    colors={'M1':'#52759b','M2':'#d48629','M3':'#8866ab','S':'#00847d','Rank':'#c73350'}
    panels=[('coverage_pct','Coverage (%)',95),('bias','Bias',0),('rmse','RMSE',None),
            ('mean_T','Mean(T)',0),('sd_T','SD(T)',1),('deterioration_pp','覆盖下降 (pp)',0)]
    for d in range(2):
        sub=comparison[comparison.dgp_id.eq(d)]
        atomic_csv(ROOT/f'DGP{d+1}_结果表.csv',sub)
        fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
        for ax,(metric,label,target) in zip(axes.ravel(),panels):
            for method in METHODS:
                a=sub[sub.method.eq(method)].set_index('error_id').reindex([0,7,14])
                ax.plot(range(3),a[metric],marker='o',label=method+(' Pilot R200' if method=='Rank' else ' Historical R1000'),color=colors[method])
                if method=='Rank':
                    for x,y in enumerate(a[metric]):ax.annotate(f'{y:.3g}',(x,y),xytext=(3,7),textcoords='offset points',fontsize=9)
            if target is not None:ax.axhline(target,color='gray',ls=':',lw=1)
            ax.set_xticks(range(3),list(ERRORS.values()));ax.set_ylabel(label);ax.grid(alpha=.2);ax.margins(y=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle(f'DGP{d+1} '+('线性' if d==0 else '非线性')+' · Wilcoxon Rank-DML 独立 Pilot')
        fig.savefig(ROOT/f'DGP{d+1}_对照图.png',dpi=180);plt.close(fig)
    theory=(ROOT/'统计口径说明.md').read_text(encoding='utf-8')
    sections=[]
    for title,frame in [('Rank 六场景结果',summary),('历史全量对比（重复次数不同）',comparison),
                        ('相同数据前200次配对比较',paired),('失败清单',failure)]:
        sections.append('<h2>'+title+'</h2><div class="scroll">'+frame.to_html(index=False,float_format=lambda x:f'{x:.5g}',na_rep='—')+'</div>')
    discussion=[]
    for _,r in summary.iterrows():
        s=comparison[(comparison.method=='S')&(comparison.dgp_id==r.dgp_id)&(comparison.error_id==r.error_id)].iloc[0]
        discussion.append(f'DGP{int(r.dgp_id)+1} {r.error}：Rank Coverage={r.coverage_pct:.1f}%，Bias={r.bias:+.5f}，RMSE={r.rmse:.5f}，Mean(T)={r.mean_T:+.3f}，SD(T)={r.sd_T:.3f}；历史 S-Huber Coverage={s.coverage_pct:.1f}%，RMSE={s.rmse:.5f}。')
    discussion.append('先看配对表的 MCSE，再判断覆盖差异。Mean(T) 偏移和 SD(T) 偏离1分别提示中心偏移和尺度问题；仅凭 Coverage 不能区分二者。固定复杂度 nuisance 的偏差不能靠放大方差来消除。')
    discussion.append('q=2 只有在当前结果和数学验证均可接受时才值得做受控小型试验；需使用二维 Jaeckel 凸优化、协方差矩阵及二维联合检验临界值，不能沿用标量 1.96。不得把本次 q=1 实验当作 q=2 理论保证。')
    notes='\n\n'.join(discussion)
    (ROOT/'结果分析.md').write_text(notes,encoding='utf-8')
    markup='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Wilcoxon Rank-DML Pilot</title>
<style>body{font:15px/1.7 "Microsoft YaHei",sans-serif;margin:32px;color:#193342;background:#f6f8fa}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;background:white}td,th{border:1px solid #ccd7df;padding:6px}th{background:#175267;color:white}img{max-width:100%}pre{white-space:pre-wrap}</style>
<h1>p=5 · Wilcoxon Rank-DML 独立 Pilot</h1><p>仅新方法运行 R=200；旧方法读取 Historical Full, R=1000。配对分析另用匹配的前200次，未重跑旧方法。没有 Wald 区间，也没有置信区间长度。</p>'''
    markup+='<h2>结果解读</h2>'+''.join('<p>'+html.escape(x)+'</p>' for x in discussion)
    markup+=''.join(sections)+'<h2>图表</h2><img src="DGP1_对照图.png"><img src="DGP2_对照图.png">'
    markup+='<h2>统计公式、假设与来源</h2><pre>'+html.escape(theory)+'</pre></html>'
    (ROOT/'中文报告.html').write_text(markup,encoding='utf-8')
    for path,h in config['source_hashes'].items():assert sha(Path(path).read_bytes())==h,'历史文件被修改'
    atomic_json(ROOT/'completed.json',dict(passed=True,data_n=len(rows),point_n=int(summary.point_n.sum()),
        test_n=int(summary.test_n.sum()),failure_events=len(failure),historical_unchanged=True,
        matched_pairs=len(raw),finished=datetime.now().isoformat()))
    print(summary.to_string(index=False),flush=True)

def run():
    config=prepare()
    check=checks();atomic_json(ROOT/'correctness_checks.json',check)
    smoke=ROOT/'smoke'
    tasks=[(d,e,0) for d in range(2) for e in ERRORS]+[(d,0,1) for d in range(2)]
    rows=collect(smoke,config,tasks)
    assert all(r['point_status']==r['test_status']=='ok' and r['matched_historical'] for r in rows)
    assert all(all(t==1 for t in r['numeric_threads']) for r in rows)
    again=collect(smoke,config,tasks);assert rows==again
    atomic_json(smoke/'validation.json',dict(passed=True,data_n=8,method_n=8,
        worker_pids=sorted(set(r['pid'] for r in rows)),checkpoint_resume=True,old_methods_run=False,
        source_prediction_hashes_match=True))
    log(ROOT,'正确性检查及 Smoke 通过；开始正式 1200 组')
    tasks=[(d,e,r) for r in range(200) for d in range(2) for e in ERRORS]
    rows=collect(ROOT,config,tasks)
    assert len(rows)==1200
    export(rows,config)
    atomic_json(ROOT/'progress.json',dict(status='complete',completed=1200,total=1200,updated=datetime.now().isoformat()))
    log(ROOT,'完成全部模拟、配对核验及中文报告')
if __name__=='__main__':
    multiprocessing.freeze_support()
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with module(0).run_lock(ROOT):run()
    except BaseException:
        atomic_json(ROOT/'execution_error.json',dict(error=traceback.format_exc()));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


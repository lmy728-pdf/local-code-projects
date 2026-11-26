"""CSV + concise Markdown + separate vector PDF figures; no simulation."""
import os
# Notebook launches may inherit an unavailable inline backend.
os.environ["MPLBACKEND"] = "Agg"
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,sys,shutil,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'settings.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(Path(cfg['archive'])/'plot_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':15,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
METHODS=['LS','H','HT','W','N'];COLORS=['#64748b','#d68b20','#a64a83','#007f78','#2563eb']

def audit_records(folder):
    import core
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'));seen=set();replays=0
    for path in (folder/'records').glob('*.json'):
        data=json.loads(path.read_text(encoding='utf-8'));cell=data['cell'];rep=data['replication'];key=(json.dumps(cell,sort_keys=True),rep)
        assert key not in seen;seen.add(key)
        if 'traceback' in data:continue
        for pipe in ('learned','oracle'):
            rows=[x for x in data['records'] if x['pipeline']==pipe]
            assert {x['method'] for x in rows}==set(METHODS) and len({x['input_hash'] for x in rows})==1
            for row in rows:
                V=np.array(row['covariance'],float);p=np.array(row['estimate'],float);truth=np.array(row['truth'],float)
                check=core.inference(p,V,truth,cell['n'])
                assert np.allclose(np.array(row['se'],float),check['se'],equal_nan=True)
                assert row['coverage']==check['coverage'].tolist()
                if cell['d']>1:assert row['joint_coverage']==check['joint_coverage']
        if rep in (0,499,999):
            cache=Path(data['source_cache']) if 'source_cache' in data else folder/'cache'/(path.stem+'.npz')
            with np.load(cache) as a:
                for row in data['records']:
                    oracle=row['pipeline']=='oracle';m=row['method'];w=a['w_oracle' if oracle else 'w'];z=a['z_oracle' if oracle else 'z'];f=a['folds']
                    threshold=a['scales'][f]*(1.345 if m=='H' else a['multipliers'][f]) if m in ('H','HT') else None
                    point,V,_,_=core.estimate(w,z,f,m,threshold,c=data.get('rank_step',1.))
                    assert np.allclose(point,np.array(row['estimate'],float),atol=1e-12,equal_nan=True)
                    assert np.allclose(V,np.array(row['covariance'],float),atol=1e-12,equal_nan=True)
                    replays+=1
    assert len(seen)==len(config['cells'])*1000
    return dict(datasets=len(seen),replayed_records=replays,shared_inputs_verified=True,inference_recomputed=True)

def markdown(df):
    def val(v):return f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def export(folder,target,stage,archive):
    audit=audit_records(folder)
    summary=folder/'summary';tables=target/'汇总表_CSV'/stage;tables.mkdir(parents=True,exist_ok=True)
    results=target/'结果_MD';results.mkdir(exist_ok=True);plots=target/'图'/stage;plots.mkdir(parents=True,exist_ok=True)
    frames={}
    for p in summary.glob('*.csv'):
        shutil.copy2(p,tables/p.name)
        if p.stat().st_size>5:frames[p.stem]=pd.read_csv(p)
    combined=pd.concat([df.assign(table=name,stage=stage) for name,df in frames.items()],ignore_index=True,sort=False)
    combined.to_csv(tables/'汇总表.csv',index=False,encoding='utf-8-sig')
    current=frames['coordinates'];base=[]
    for s in ('g1_scalar_n1000','g2_vector_n1000','g3_n500'):
        p=archive/'runs'/s/'summary'/'coordinates.csv'
        if p.exists():base.append(pd.read_csv(p).assign(source=s))
    references=pd.concat(base,ignore_index=True)
    if stage=='g4_n2000':comparison=pd.concat([references,current.assign(source=stage)],ignore_index=True)
    elif stage=='g5_rf':comparison=pd.concat([references[(references.n==1000)&((references.d==1)|((references.d==3)&(references.rho==.5)))].assign(learner='spline_ridge'),current.assign(source=stage,learner='random_forest')],ignore_index=True)
    elif stage.startswith('g6b_'):comparison=pd.concat([references[(references.n==1000)&(references.law.isin(['sln075','sln100']))&((references.d==1)|((references.d==3)&(references.rho==.5)))].assign(c=1),current.assign(source=stage,c=float(stage.split('_c')[1]))],ignore_index=True)
    else:comparison=pd.concat([references[(references.n==1000)&(references.law=='sln100')&((references.d==1)|((references.d==3)&(references.rho==.5)))].assign(tau=.2),current.assign(source=stage,tau=.5)],ignore_index=True)
    comparison.to_csv(tables/'正式对照.csv',index=False,encoding='utf-8-sig')
    index=[]
    specifications=[('coordinates',['RMSE','coverage','SEratio','mean_length','bias'],['coordinate']),('vectors',['vector_RMSE','coverage','median_log_volume'],[]),('contrasts',['coverage','mean_length'],['contrast'])]
    for name,metrics,extra in specifications:
        df=frames.get(name)
        if df is None:continue
        if name=='vectors':df=df[df.d>1]
        groups=['law','n','d','rho','pipeline']+extra
        for key,q in df.groupby(groups):
            q=q.set_index('method').loc[METHODS]
            for metric in metrics:
                if metric not in q or q[metric].isna().all():continue
                fig,ax=plt.subplots(figsize=(11,8));fig.subplots_adjust(left=.16,right=.96,bottom=.17,top=.78)
                values=q[metric].to_numpy();x=np.arange(5)
                if metric=='coverage':
                    ax.errorbar(x,values,yerr=1.96*q.coverage_mcse,fmt='o',capsize=5);ax.axhline(.95,color='red',ls='--');ax.set_ylim(max(0,min(.8,float(np.nanmin(values))-.08)),1.04)
                else:
                    ax.bar(x,values,color=COLORS)
                    if metric=='SEratio':ax.axhline(1,color='red',ls='--')
                ax.set_xticks(x,METHODS);ax.set_ylabel(metric);ax.grid(axis='y',alpha=.2)
                title=stage+' · '+name+' · '+metric;subtitle=' · '.join(f'{k}={v}' for k,v in zip(groups,key))
                fig.suptitle(title+'\n'+subtitle,fontsize=16)
                fig.text(.5,.05,'R=1000；coverage 误差条为 ±1.96 MC SE；无效区间按未覆盖计。',ha='center',fontsize=11)
                filename=name+'_'+'_'.join(map(str,key))+'_'+metric+'.pdf';fig.savefig(plots/filename);plt.close(fig);index.append(filename)
    # Separate paired ratio figures, including bootstrap uncertainty.
    pairs=frames['paired'].drop_duplicates(['law','n','d','rho','pipeline','method','baseline'])
    for key,q in pairs.groupby(['law','n','d','rho','pipeline']):
        fig,ax=plt.subplots(figsize=(11,8));labels=[]
        for i,(m,b) in enumerate((m,b) for m in ('W','N') for b in ('LS','H','HT')):
            row=q[(q.method==m)&(q.baseline==b)].iloc[0];ax.plot([row.ratio_lower,row.ratio_upper],[i,i],lw=2);ax.plot(row.RMSE_ratio,i,'o');labels.append(m+'/'+b)
        ax.set_yticks(range(6),labels);ax.axvline(1,color='red',ls='--');ax.set_xlabel('RMSE ratio · paired bootstrap 95% interval');ax.set_title(stage+'\n'+' · '.join(map(str,key)));fig.tight_layout()
        filename='paired_'+'_'.join(map(str,key))+'.pdf';fig.savefig(plots/filename);plt.close(fig);index.append(filename)
    # Formal comparison figures, separated by coordinate and metric.
    axis='n' if stage=='g4_n2000' else ('learner' if stage=='g5_rf' else ('c' if stage.startswith('g6b_') else 'tau'))
    for key,q in comparison.groupby(['law' if axis!='tau' else 'experiment','d','rho','pipeline','coordinate']):
        for metric in ('RMSE','coverage','SEratio','mean_length'):
            fig,ax=plt.subplots(figsize=(11,8));ticks=sorted(q[axis].unique());positions={v:i for i,v in enumerate(ticks)}
            for method,color in zip(METHODS,COLORS):
                a=q[q.method==method].sort_values(axis);ax.errorbar([positions[v] for v in a[axis]],a[metric],yerr=1.96*a.coverage_mcse if metric=='coverage' else None,fmt='o-',label=method,color=color,capsize=4)
            if metric in ('coverage','SEratio'):ax.axhline(.95 if metric=='coverage' else 1,color='red',ls='--')
            ax.set_xticks(range(len(ticks)),list(map(str,ticks)));ax.set_xlabel(axis);ax.set_ylabel(metric);ax.legend(ncol=5);ax.set_title(stage+' formal comparison\n'+' · '.join(map(str,key)));fig.tight_layout()
            filename='compare_'+'_'.join(map(str,key))+'_'+metric+'.pdf';fig.savefig(plots/filename);plt.close(fig);index.append(filename)
    (plots/'图表索引.md').write_text('# PDF 矢量图\n\n'+'\n'.join(f'- [{s}](<{s}>)' for s in index),encoding='utf-8')
    failure=json.loads((summary/'failures.json').read_text(encoding='utf-8'))
    audit.update(failure_counts=failure,pdf_figures=len(index),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.glob('*.csv')})
    (folder/'delivery_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    short=current[(current.experiment=='scalar')&(current.pipeline=='learned')][['law','method','RMSE','coverage','SEratio']]
    text=f'# {stage} 结果\n\n{audit["datasets"]} 份数据；每条件 R=1000；缓存复算 {audit["replayed_records"]} 条通过。失败分类：{failure}。\n\n'
    text+='全部方法、Oracle 对照、逐坐标、联合椭球、contrast 和配对比较见 CSV。总体 coverage 包含失败；CSV 同时给出有效区间分母。\n\n'
    text+=f'[汇总表 CSV](../汇总表_CSV/{stage}/汇总表.csv) · [正式对照 CSV](../汇总表_CSV/{stage}/正式对照.csv) · [PDF 图索引](../图/{stage}/图表索引.md)\n\n'+markdown(short)+'\n'
    (results/(stage+'_结果.md')).write_text(text,encoding='utf-8')
    print('Delivery complete:',target,flush=True)

"""Readable standalone vector figures from completed summary CSVs only."""
import os,sys,json,hashlib,zipfile,argparse
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'plot_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':17,'axes.titlesize':22,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':15,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path','savefig.facecolor':'white'})
METHODS=['LS','H','HT','W','N'];COLORS=['#64748B','#D68B20','#A64A83','#007F78','#2563EB']
LAWS={'gaussian':'Gaussian','t3':r'$t_3/\sqrt{3}$','sln075':'平滑 lognormal (σ=0.75)','sln100':'平滑 lognormal (σ=1.00)'}
PIPE={'learned':'Learned nuisance','oracle':'Oracle nuisance'}
ALL=[];SOURCES={}
parser=argparse.ArgumentParser();parser.add_argument('--stages',nargs='+',choices=['G0','G1','G2','G3'],required=True);args=parser.parse_args()
def read(stage,file):
    p=ROOT/'runs'/stage/'summary'/file;SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def canvas():
    f,a=plt.subplots(figsize=(11,8));f.subplots_adjust(left=.15,right=.96,bottom=.22,top=.79);a.grid(axis='y',alpha=.17);a.set_axisbelow(True);return f,a
def finish(f,a,stage,name,metric,law,pipe,d=1,rho=0,note='',sample='n=1000 · R=1000'):
    if stage=='G3' and sample=='n=1000 · R=1000':sample='n=500 · R=1000'
    title=metric+'\n'+LAWS[law]+' · '+PIPE[pipe]
    f.suptitle(title,fontsize=22,y=.97,fontweight='bold')
    sub=stage+(' pilot（仅作实现诊断）' if stage=='G0' else ' 正式实验')+' · '+sample
    if d>1:sub+=f' · d={d}, ρ={rho}'
    f.text(.5,.84,sub,ha='center',fontsize=15,color='#475569')
    f.text(.5,.055,note,ha='center',va='center',fontsize=12,color='#475569')
    out=ROOT/'vector_figures'/stage;out.mkdir(parents=True,exist_ok=True)
    path=out/(name+'.svg');f.savefig(path,format='svg')
    # A small number of QA previews, never the official deliverable.
    if (stage=='G1' and law=='sln100' and pipe=='learned' and metric in ('RMSE','95% 区间覆盖率')) or (stage in ('G2','G3') and d==3 and rho==.5 and law=='sln100' and pipe=='learned' and metric in ('Vector RMSE','Wald 椭球联合覆盖率','Vector RMSE · 正式样本量对照','Wald 椭球联合覆盖率 · 正式样本量对照')):
        qa=ROOT/'reports'/'figure_qa';qa.mkdir(exist_ok=True);f.savefig(qa/(name+'.png'),dpi=110)
    plt.close(f);ALL.append(dict(stage=stage,file=path.relative_to(ROOT/'vector_figures').as_posix(),metric=metric,law=law,pipeline=pipe,d=int(d),rho=float(rho),sample=sample))
def category(df,stage,key,metric,law,pipe,d=1,rho=0,coverage=False,reference=None,suffix=''):
    f,a=canvas();x=np.arange(5);df=df.set_index('method').loc[METHODS]
    if coverage:
        for i,m in enumerate(METHODS):
            v=df.loc[m,'coverage'];err=1.96*df.loc[m,'coverage_mcse'];a.errorbar(i,v,yerr=err,fmt='o',color=COLORS[i],ms=10,capsize=6,lw=2)
            a.annotate(f'{v:.3f}',(i,v+err),xytext=(0,10),textcoords='offset points',ha='center',fontsize=14,bbox=dict(facecolor='white',edgecolor='none',pad=1))
        low=max(0,min(.85,float((df.coverage-1.96*df.coverage_mcse).min())-.045));a.set_ylim(low,1.045);reference=.95
        note='红线：名义覆盖率 95%；误差条：±1.96 × Monte Carlo SE。\n无效区间按未覆盖计；这些误差条反映模拟精度。'
    else:
        values=df[key];a.bar(x,values,color=COLORS,width=.6)
        for i,v in enumerate(values):a.annotate(f'{v:.4f}' if abs(v)<.1 else f'{v:.3f}',(i,v),xytext=(0,7 if v>=0 else -17),textcoords='offset points',ha='center',fontsize=14)
        lo=min(0,float(values.min())*1.2);hi=max(0,float(values.max()))
        if reference is not None:hi=max(hi,reference)
        span=max(hi-lo,.01);a.set_ylim(lo-.07*span if lo<0 else 0,hi+.18*span)
        note='五方法使用同一数据、folds 和 nuisance 预测。\n精度和区间大小须结合 coverage 一起解释。'
    if reference is not None:a.axhline(reference,color='#B91C1C',ls='--',lw=1.5)
    a.set_xticks(x,METHODS);a.set_ylabel(metric)
    name=f'{stage}_d{d}_rho{rho}_{law}_{pipe}_{key}{suffix}'
    finish(f,a,stage,name,metric,law,pipe,d,rho,note)
def curves(df,key,metric,law,pipe,d=1,rho=0,coverage=False,reference=None):
    f,a=canvas()
    for m,color in zip(METHODS,COLORS):
        q=df[df.method==m].sort_values('n');assert len(q)==3
        a.errorbar(q.n,q[key],yerr=1.96*q.coverage_mcse if coverage else None,fmt='o-',color=color,label=m,ms=7,lw=2,capsize=4)
    if coverage:reference=.95;a.set_ylim(max(0,min(.8,(df.coverage-1.96*df.coverage_mcse).min()-.04)),1.04)
    elif key!='median_log_volume':a.set_ylim(bottom=0)
    if reference is not None:a.axhline(reference,color='#B91C1C',ls='--',lw=1.5)
    a.set_xticks([500,1000,2000]);a.set_xlabel('样本量 n');a.set_ylabel(metric);a.legend(ncol=5,fontsize=13,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False)
    finish(f,a,'G0',f'G0_d{d}_rho{rho}_{law}_{pipe}_{key}',metric,law,pipe,d,rho,'pilot 每条件仅 R=100，不作为正式样本量趋势结论。\n'+('误差条为 ±1.96 coverage MC SE；红线为 95%。' if coverage else '未与 G1/G2 的正式重复混用。'),sample='n=500,1000,2000 · R=100')

def formal_curves(df,key,title,law,pipe,d=1,rho=0,coverage=False,reference=None):
    f,a=canvas()
    for m,color in zip(METHODS,COLORS):
        q=df[df.method==m].sort_values('n');assert list(q.n)==[500,1000]
        a.errorbar(q.n,q[key],yerr=1.96*q.coverage_mcse if coverage else None,fmt='o-',color=color,label=m,ms=8,lw=2,capsize=4)
    if coverage:reference=.95;a.set_ylim(max(0,min(.8,(df.coverage-1.96*df.coverage_mcse).min()-.04)),1.04)
    elif key!='median_log_volume':a.set_ylim(bottom=0)
    if reference is not None:a.axhline(reference,color='#B91C1C',ls='--',lw=1.5)
    a.set_xticks([500,1000]);a.set_xlabel('正式样本量 n');a.set_ylabel(title);a.legend(ncol=5,fontsize=13,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False)
    finish(f,a,'G3',f'G3_d{d}_rho{rho}_{law}_{pipe}_{key}_formal_n_compare',title+' · 正式样本量对照',law,pipe,d,rho,'n=500 来自 G3；n=1000 来自 G1/G2；每条件均 R=1000。\n'+('误差条为 ±1.96 coverage MC SE；不混入 pilot。' if coverage else '两个样本量使用独立数据；n=2000 正式实验尚未运行。'),sample='n=500 vs 1000 · R=1000')

for stage in args.stages:
    folder={'G0':'g0_pilot','G1':'g1_scalar_n1000','G2':'g2_vector_n1000','G3':'g3_n500'}[stage]
    coords=read(folder,'coordinates.csv')
    if stage=='G0':
        v=read(folder,'vectors.csv')
        for (law,pipe),df in coords[coords.experiment=='scalar'].groupby(['law','pipeline']):
            for key,title,ref,err in [('RMSE','RMSE',None,False),('coverage','95% 区间覆盖率',.95,True),('SEratio','平均 SE / 经验 SD',1,False),('mean_length','平均区间长度',None,False)]:curves(df,key,title,law,pipe,reference=ref,coverage=err)
        for (d,rho,law,pipe),df in v[v.experiment=='vector'].groupby(['d','rho','law','pipeline']):
            for key,title,err in [('vector_RMSE','Vector RMSE',False),('coverage','Wald 椭球联合覆盖率',True),('median_log_volume','椭球 log-volume 中位数',False)]:curves(df,key,title,law,pipe,int(d),rho,coverage=err)
    if stage in ('G1','G3'):
        for (law,pipe),df in coords[coords.experiment=='scalar'].groupby(['law','pipeline']):
            for key,title,ref,err in [('RMSE','RMSE',None,False),('coverage','95% 区间覆盖率',.95,True),('SEratio','平均 SE / 经验 SD',1,False),('mean_length','平均区间长度',None,False),('bias','估计偏差 Bias',0,False)]:category(df,stage,key,title,law,pipe,reference=ref,coverage=err)
    if stage in ('G2','G3'):
        v=read(folder,'vectors.csv');con=read(folder,'contrasts.csv')
        for (d,rho,law,pipe),df in v[v.experiment=='vector'].groupby(['d','rho','law','pipeline']):
            for key,title,err in [('vector_RMSE','Vector RMSE',False),('coverage','Wald 椭球联合覆盖率',True),('median_log_volume','椭球 log-volume 中位数',False)]:category(df,stage,key,title,law,pipe,int(d),rho,coverage=err)
        for (d,rho,law,pipe,j),df in coords[coords.experiment=='vector'].groupby(['d','rho','law','pipeline','coordinate']):
            for key,title,ref,err in [('RMSE','RMSE',None,False),('coverage','95% marginal 覆盖率',.95,True),('SEratio','平均 SE / 经验 SD',1,False)]:category(df,stage,key,title+f' · 坐标 {j}',law,pipe,int(d),rho,coverage=err,reference=ref,suffix=f'_coord{j}')
        for (d,rho,law,pipe,contrast),df in con.groupby(['d','rho','law','pipeline','contrast']):
            for key,title,err in [('coverage','Contrast 覆盖率',True),('mean_length','Contrast 平均区间长度',False)]:category(df,stage,key,title+f' · {contrast}',law,pipe,int(d),rho,coverage=err,suffix='_'+contrast)
    if stage in ('G1','G2','G3'):
        paired=read(folder,'paired.csv').drop_duplicates(['d','rho','law','pipeline','method','baseline'])
        for (d,rho,law,pipe),df in paired.groupby(['d','rho','law','pipeline']):
            f,a=canvas();labels=[]
            for i,(m,b) in enumerate((m,b) for m in ('W','N') for b in ('LS','H','HT')):
                row=df[(df.method==m)&(df.baseline==b)].iloc[0];color=COLORS[3 if m=='W' else 4];a.plot([row.ratio_lower,row.ratio_upper],[i,i],color=color,lw=3);a.plot(row.RMSE_ratio,i,'o',color=color,ms=10);labels.append(f'{m} / {b}')
            a.axvline(1,color='#B91C1C',ls='--');a.set_yticks(range(6),labels);a.invert_yaxis();a.set_xlabel('RMSE ratio（小于 1 表示 rank 方法更低）');a.grid(axis='x',alpha=.15)
            finish(f,a,stage,f'{stage}_d{int(d)}_rho{rho}_{law}_{pipe}_paired_RMSE','配对 RMSE 比值与 bootstrap 95% 区间',law,pipe,int(d),rho,'按 replication 联合重采样 2000 次；W/N 分别对三个基线。\n多元使用 vector RMSE；完整 bootstrap 分母见 paired.csv。')
    if stage=='G3':
        scalar=pd.concat([coords[coords.experiment=='scalar'],read('g1_scalar_n1000','coordinates.csv')])
        for (law,pipe),df in scalar.groupby(['law','pipeline']):
            for key,title,ref,err in [('RMSE','标量 RMSE',None,False),('coverage','标量区间覆盖率',.95,True),('SEratio','标量 MeanSE / SD',1,False),('mean_length','标量区间长度',None,False)]:formal_curves(df,key,title,law,pipe,reference=ref,coverage=err)
        vector=pd.concat([v[v.experiment=='vector'],read('g2_vector_n1000','vectors.csv')])
        for (d,rho,law,pipe),df in vector.groupby(['d','rho','law','pipeline']):
            for key,title,err in [('vector_RMSE','Vector RMSE',False),('coverage','Wald 椭球联合覆盖率',True),('median_log_volume','椭球 log-volume 中位数',False)]:formal_curves(df,key,title,law,pipe,int(d),rho,coverage=err)
    (ROOT/'vector_figures'/stage/'manifest.json').write_text(json.dumps(dict(figures=[x for x in ALL if x['stage']==stage],sources=SOURCES),ensure_ascii=False,indent=2),encoding='utf-8')
    print(stage,'SVG figures complete',flush=True)

out=ROOT/'vector_figures';entries=[]
for p in out.glob('G*/manifest.json'):entries+=json.loads(p.read_text(encoding='utf-8'))['figures']
for e in entries:e['file']=e['file'].replace('\\','/')
html=['<!doctype html><meta charset="utf-8"><title>独立大幅矢量图</title><style>body{font:18px sans-serif;max-width:1250px;margin:35px auto;padding:20px;color:#172033}select{font-size:18px;margin:8px;padding:8px}img{width:100%;max-width:1100px}article{border-top:1px solid #ddd;margin:25px 0;padding-top:20px}nav{position:sticky;top:0;background:white;padding:15px;border-bottom:1px solid #aaa}</style><h1>独立大幅 SVG 矢量图</h1><p>每张图单独绘制。G0 为 pilot；G1/G2/G3 为正式实验。SVG 可无限放大，不嵌入位图。</p><nav>阶段 <select id="stage"><option value="">全部</option>']
html += [f'<option>{x}</option>' for x in sorted({e['stage'] for e in entries})]
html += ['</select>分布 <select id="law"><option value="">全部</option>'+''.join(f'<option>{x}</option>' for x in LAWS)+'</select>流程 <select id="pipe"><option value="">全部</option><option>learned</option><option>oracle</option></select>指标 <select id="metric"><option value="">全部</option>'+''.join(f'<option>{x}</option>' for x in sorted({e['metric'] for e in entries}))+'</select><span id="count"></span></nav>']
for e in entries:html.append(f'<article data-stage="{e["stage"]}" data-law="{e["law"]}" data-pipe="{e["pipeline"]}" data-metric="{e["metric"]}"><h2>{e["stage"]} · {e["metric"]} · {e["law"]} · {e["pipeline"]} · d={e["d"]}, ρ={e["rho"]}</h2><a href="{e["file"]}" target="_blank">打开 / 保存原始 SVG</a><br><img loading="lazy" src="{e["file"]}"></article>')
html+=['<script>function filter(){let n=0;document.querySelectorAll("article").forEach(a=>{let show=["stage","law","pipe","metric"].every(k=>!document.getElementById(k).value||a.dataset[k]===document.getElementById(k).value);a.style.display=show?"":"none";n+=show});document.getElementById("count").textContent=" 显示 "+n+" 张"}document.querySelectorAll("select").forEach(s=>s.onchange=filter);filter()</script>']
(out/'index.html').write_text('\n'.join(html),encoding='utf-8')
for stage in args.stages:
    local=['<!doctype html><meta charset="utf-8"><title>'+stage+' SVG</title><style>body{font:18px sans-serif;max-width:1200px;margin:30px auto}img{width:100%}section{margin:45px 0}</style><h1>'+stage+' 独立大幅 SVG 图表</h1>']
    for e in entries:
        if e['stage']==stage:local.append(f'<section><h2>{e["metric"]} · {e["law"]} · {e["pipeline"]} · d={e["d"]}, ρ={e["rho"]}</h2><a href="{Path(e["file"]).name}">打开 SVG</a><img loading="lazy" src="{Path(e["file"]).name}"></section>')
    (out/stage/'index.html').write_text('\n'.join(local),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'reports'/(stage+'_vector_figures.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in (out/stage).iterdir():z.write(p,stage+'/'+p.name)
print('Vector gallery updated:',len(entries),'standalone figures',flush=True)

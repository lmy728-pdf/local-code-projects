"""Plot saved summaries only. No simulation, refitting, or retuning."""
import os,sys,json,hashlib,zipfile
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'plot_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','savefig.facecolor':'white'})
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
METHODS=['LS','H','HT','W','N'];COLORS=['#64748B','#D68B20','#A64A83','#007F78','#2563EB']
LAWS=['gaussian','t3','sln075','sln100'];LABELS=['Gaussian',r'$t_3 / \sqrt{3}$','平滑 lognormal (σ=0.75)','平滑 lognormal (σ=1.00)']
PIPE={'learned':'Learned nuisance','oracle':'Oracle nuisance'}
SOURCES={};figures=[]
def read(stage,name):
    p=ROOT/'runs'/stage/'summary'/name;SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
g1=read('g1_scalar_n1000','coordinates.csv');pairs=read('g1_scalar_n1000','paired.csv')
g0=read('g0_pilot','coordinates.csv');g0v=read('g0_pilot','vectors.csv')
def finish(fig,name,title,note):
    fig.suptitle(title,fontsize=17,fontweight='bold',y=.985)
    fig.text(.5,.015,note,ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=[.02,.05,.995,.945],h_pad=2.2,w_pad=1.6)
    for ext in ('png','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=180)
    plt.close(fig);figures.append(dict(name=name,title=title,note=note))
def base(ax):
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
def categorical(metric,title,ref=None,errors=False):
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True)
    vals=g1[metric].to_numpy();low=0 if metric!='bias' else min(0,float(vals.min())*1.2)
    high=max(float(vals.max())*1.12,.01)
    if errors:low=max(0,min(.85,float((g1.coverage-1.96*g1.coverage_mcse).min())-.025));high=1.015
    if ref is not None and not errors:high=max(high,ref*1.12)
    for r,pipe in enumerate(PIPE):
        for c,law in enumerate(LAWS):
            ax=axes[r,c];df=g1[(g1.pipeline==pipe)&(g1.law==law)].set_index('method').loc[METHODS];x=np.arange(5)
            if errors:
                for i,m in enumerate(METHODS):ax.errorbar(i,df.loc[m,metric],yerr=1.96*df.loc[m,'coverage_mcse'],fmt='o',color=COLORS[i],capsize=4,markersize=7)
            else:ax.bar(x,df[metric],color=COLORS,width=.66)
            if ref is not None:ax.axhline(ref,color='#B91C1C',ls='--',lw=1.2)
            for i,v in enumerate(df[metric]):
                if not errors:ax.annotate(f'{v:.3f}',(i,v),xytext=(0,4 if v>=0 else -11),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(x,METHODS);ax.set_ylim(low,high);ax.set_title(LABELS[c]+'\n'+PIPE[pipe],fontsize=10);base(ax)
            if c==0:ax.set_ylabel(title)
    note='G1 正式实验 · n=1000 · 每分布 R=1000 · 全部四分布、五方法均保留'
    if errors:note+='\n误差条：±1.96 × coverage MC SE；虚线：名义覆盖率 95%'
    finish(fig,'G1_'+metric,title+'｜G1 正式标量实验',note)
for metric,title,ref,err in [('RMSE','RMSE',None,False),('coverage','95% 区间覆盖率',.95,True),('SEratio','平均 SE / 经验 SD',1,False),('mean_length','平均区间长度',None,False),('bias','估计偏差 Bias',0,False)]:categorical(metric,title,ref,err)

# One exportable dashboard comparing Learned and Oracle within the same panel.
fig,axes=plt.subplots(4,4,figsize=(15,12))
for r,(metric,title,ref) in enumerate([('RMSE','RMSE',None),('coverage','覆盖率',.95),('SEratio','MeanSE / SD',1),('mean_length','区间长度',None)]):
    vals=g1[metric];lo=0;hi=max(vals.max()*1.12,1.1 if ref==1 else .01)
    if metric=='coverage':lo=max(0,min(.85,(g1.coverage-1.96*g1.coverage_mcse).min()-.03));hi=1.015
    for c,law in enumerate(LAWS):
        ax=axes[r,c]
        for pipe,offset,marker in [('learned',-.12,'o'),('oracle',.12,'s')]:
            df=g1[(g1.pipeline==pipe)&(g1.law==law)].set_index('method').loc[METHODS]
            for i,m in enumerate(METHODS):
                ax.errorbar(i+offset,df.loc[m,metric],yerr=1.96*df.loc[m,'coverage_mcse'] if metric=='coverage' else None,fmt=marker,color=COLORS[i],mfc=COLORS[i] if pipe=='learned' else 'white',capsize=2,ms=5)
        if ref is not None:ax.axhline(ref,color='#B91C1C',ls='--',lw=1)
        ax.set_xticks(range(5),METHODS);ax.set_ylim(lo,hi);base(ax)
        if c==0:ax.set_ylabel(title)
        if r==0:ax.set_title(LABELS[c])
finish(fig,'G1_overview','G1 正式实验总览｜n=1000，R=1000','实心圆：Learned；空心方块：Oracle。颜色对应方法，所有分布均保留。\n覆盖率误差条为 ±1.96 MC SE；不能只凭更短区间认定推断更好。')

fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True)
labels=[f'{m}/{b}' for m in ('W','N') for b in ('LS','H','HT')]
for r,pipe in enumerate(PIPE):
    for c,law in enumerate(LAWS):
        ax=axes[r,c];df=pairs[(pairs.pipeline==pipe)&(pairs.law==law)&(pairs.size_metric=='coordinate_1')]
        for i,(m,b) in enumerate((m,b) for m in ('W','N') for b in ('LS','H','HT')):
            a=df[(df.method==m)&(df.baseline==b)].iloc[0]
            ax.plot([a.ratio_lower,a.ratio_upper],[i,i],color=COLORS[3 if m=='W' else 4],lw=2);ax.plot(a.RMSE_ratio,i,'o',color=COLORS[3 if m=='W' else 4])
        ax.axvline(1,color='#B91C1C',ls='--');ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_title(LABELS[c]+'\n'+PIPE[pipe],fontsize=10);ax.grid(axis='x',alpha=.18)
        if r==1:ax.set_xlabel('RMSE ratio（小于 1 表示 rank 方法 RMSE 更低）',fontsize=8)
finish(fig,'G1_paired_RMSE','G1 配对 RMSE 比值及 bootstrap 95% 区间','每个比值按 replication 联合重采样 2000 次；W/N 分别对 LS、H、HT。\n区间使用 bootstrap 2.5%/97.5% 分位数，并非拟合置信区间。')

def curves(df,metric,title,name,ref=None,errors=False,extra=''):
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True)
    for r,pipe in enumerate(PIPE):
        for c,law in enumerate(LAWS):
            ax=axes[r,c]
            for m,color in zip(METHODS,COLORS):
                a=df[(df.pipeline==pipe)&(df.law==law)&(df.method==m)].sort_values('n')
                assert len(a)==3
                ax.errorbar(a.n,a[metric],yerr=1.96*a.coverage_mcse if errors else None,color=color,fmt='o-',lw=1.5,ms=4,capsize=2,label=m)
            if ref is not None:ax.axhline(ref,color='#B91C1C',ls='--',lw=1)
            ax.set_xticks([500,1000,2000]);ax.set_title(LABELS[c]+'\n'+PIPE[pipe],fontsize=10);base(ax)
            if r==1:ax.set_xlabel('样本量 n')
            if c==0:ax.set_ylabel(title)
    if errors:
        low=max(0,min(.8,(df.coverage-1.96*df.coverage_mcse).min()-.025));axes[0,0].set_ylim(low,1.02)
    elif metric!='median_log_volume':axes[0,0].set_ylim(bottom=0)
    fig.legend(handles=[Line2D([0],[0],color=c,lw=2,label=m) for m,c in zip(METHODS,COLORS)],loc='upper center',bbox_to_anchor=(.5,.947),ncol=5,frameon=False)
    finish(fig,name,title+'｜G0 pilot '+extra,'G0 pilot：每条件仅 R=100，用于实现诊断，不是正式样本量趋势结论。\n'+('覆盖率误差条为 ±1.96 MC SE；红色虚线为 95%。' if errors else '正式 G1（R=1000）没有与这些 pilot 曲线混用。'))
scalar=g0[g0.experiment=='scalar']
for metric,title,ref,err in [('RMSE','标量 RMSE',None,False),('coverage','标量区间覆盖率',.95,True),('SEratio','标量 MeanSE / SD',1,False),('mean_length','标量平均区间长度',None,False)]:curves(scalar,metric,title,'G0_scalar_'+metric,ref,err)
for d in (2,3):
    for rho in (0,.5):
        df=g0v[(g0v.experiment=='vector')&(g0v.d==d)&(g0v.rho==rho)]
        for metric,title,ref,err in [('vector_RMSE','Vector RMSE',None,False),('coverage','Wald 椭球联合覆盖率',.95,True),('median_log_volume','椭球 log-volume 中位数',None,False)]:curves(df,metric,title,f'G0_vector_d{d}_rho{rho}_{metric}',ref,err,f'd={d}, ρ={rho}')

manifest=dict(source_csv_sha256=SOURCES,matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,figures=figures,coverage_errorbars='Wald MC uncertainty: +/-1.96*sqrt(p*(1-p)/attempted); binomial intervals separately in CSV',G2='paused; incomplete G2 excluded from plots')
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
md=['# 图表索引','','G1 为正式 n=1000、R=1000 结果；G0 为 R=100 pilot，二者不混用。G2 已暂停，未完成结果不作为正式图交付。','']
html=['<!doctype html><meta charset="utf-8"><title>Rank-DML 图表</title><style>body{font:16px sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#172033}img{width:100%;border:1px solid #ddd}section{margin:48px 0}a{color:#2563eb}</style><h1>Rank-DML 图表</h1><p>G1：正式 n=1000、R=1000。G0：R=100 pilot，仅作实现诊断。G2 已暂停。</p>']
for f in figures:
    n=f['name'];md.extend([f'## {f["title"]}',f'[PNG]({n}.png) · [SVG]({n}.svg)',f'![{f["title"]}]({n}.png)','']);html.append(f'<section><h2>{f["title"]}</h2><p>{f["note"]}</p><a href="{n}.svg">下载 SVG</a><img loading="lazy" src="{n}.png"></section>')
(OUT/'README.md').write_text('\n'.join(md),encoding='utf-8');(OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
with zipfile.ZipFile(ROOT/'reports'/'G0_G1_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file():z.write(p,'figures/'+p.name)
    z.write(Path(__file__),'make_figures.py')
    for p in SOURCES:z.write(ROOT/p,'source_data/'+p.replace('/','_'))
print(f'Created {len(figures)} figures, PNG+SVG, index and ZIP',flush=True)

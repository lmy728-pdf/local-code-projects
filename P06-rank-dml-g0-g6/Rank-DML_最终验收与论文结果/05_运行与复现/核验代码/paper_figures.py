import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_records import OUT,PROJECT,ARCHIVE,save
sys.path.insert(0,str(ARCHIVE/'plot_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':13,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
METHODS=['LS','H','HT','W','N'];COLORS=['#64748b','#d68b20','#a64a83','#007f78','#2563eb']
LAWS=['gaussian','t3','sln075','sln100'];TITLES=['Gaussian','t(3) / √3','平滑 lognormal σ=0.75','平滑 lognormal σ=1']
INDEX=[]

def read(p):return pd.read_csv(OUT/p,encoding='utf-8-sig')

def finish(fig,name,category,title,source,scope):
    path=OUT/'06_论文图表'/category/(name+'.pdf')
    fig.savefig(path);plt.close(fig)
    INDEX.append(dict(file=str(path),title=title,category=category,source_table=source,scope=scope,format='PDF vector',status='new_from_saved_results'))
    print('Figure',len(INDEX),name,flush=True)

def panels(title,pipe,shape=(2,2)):
    fig,axes=plt.subplots(*shape,figsize=(16,12) if shape==(2,2) else (16,8),squeeze=False)
    fig.subplots_adjust(top=.84,bottom=.18 if shape==(1,2) else .12,left=.085,right=.97,hspace=.37,wspace=.25)
    fig.suptitle(title+' · '+pipe,fontsize=22,y=.98)
    return fig,axes.ravel()

def legend(fig,ax):
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=5,frameon=False)

def vector_trends(v):
    for d in [2,3]:
        for rho in [0.,.5]:
            for pipe in ['learned','oracle']:
                for metric,title in [('vector_RMSE','多元整体 RMSE'),('coverage','Wald 联合覆盖率'),('median_log_volume','联合椭球体积的 log 中位数')]:
                    fig,axes=panels(f'{title}：d={d}, ρ={rho:g}',pipe)
                    for ax,law,lawtitle in zip(axes,LAWS,TITLES):
                        q=v[(v.d==d)&(v.rho==rho)&(v.pipeline==pipe)&(v.law==law)]
                        for method,color in zip(METHODS,COLORS):
                            a=q[q.method==method].sort_values('n');assert a.n.tolist()==[500,1000,2000]
                            ax.errorbar(a.n,a[metric],yerr=1.96*a.coverage_mcse if metric=='coverage' else None,fmt='o-',capsize=4,label=method,color=color)
                        ax.set_xticks([500,1000,2000]);ax.set_xlabel('样本量 n');ax.set_ylabel(title);ax.set_title(lawtitle);ax.grid(alpha=.2)
                        if metric=='coverage':ax.axhline(.95,color='#b91c1c',ls='--');ax.set_ylim(0,1.04)
                        elif metric=='vector_RMSE':ax.set_ylim(bottom=0)
                    note='正式独立重复，每条件 R=1000。覆盖率误差条为 ±1.96 Monte Carlo SE；红线为 95%。' if metric=='coverage' else '正式独立重复，每条件 R=1000；保留所有方法、分布和样本量。完整数值见 T02。'
                    legend(fig,axes[0]);fig.text(.5,.035,note,ha='center',fontsize=12)
                    finish(fig,f'vector_d{d}_rho{rho:g}_{pipe}_{metric}','主实验',title,'T02_多元主结果.csv',f'all laws; n=500/1000/2000; d={d}; rho={rho}; {pipe}')

def diagnostics(c,v,contrast,diag,oracle):
    # Every point is an explicitly identified condition/coordinate, not a pooled estimator.
    allc=c[c.stage.isin(['G1','G2','G3','G4','G5'])]
    keys=['batch','experiment','law','n','d','rho','coordinate']
    merged=allc.merge(diag[keys+['mse_m_mean','mse_ell_mean','alignment_variance_mean','LS_population_displacement_mean']],on=keys,validate='many_to_one')
    fig,axes=panels('总体位移与实际 LS Bias','learned',shape=(1,2))
    for ax,learner in zip(axes,['Spline Ridge','RF']):
        q=merged[(merged.method=='LS')&(merged.pipeline=='learned')&(merged.learner==learner)]
        for law,color in zip(LAWS,COLORS):
            a=q[q.law==law];ax.scatter(a.LS_population_displacement_mean,a.bias,c=color,label=law,s=30,alpha=.8)
        lo=min(q.LS_population_displacement_mean.min(),q.bias.min());hi=max(q.LS_population_displacement_mean.max(),q.bias.max());span=hi-lo
        ax.plot([lo-.05*span,hi+.05*span],[lo-.05*span,hi+.05*span],'k--',lw=1)
        ax.set_xlabel('平均 B(pop)LS');ax.set_ylabel('实际平均 Bias');ax.set_title(learner);ax.grid(alpha=.2)
    legend(fig,axes[0]);fig.text(.5,.04,'每点为一个条件/坐标；总体位移不是有限样本平均 Bias 的恒等式。',ha='center')
    finish(fig,'diagnostic_displacement_bias','诊断','总体位移与实际 Bias','T09_Nuisance诊断.csv + T01/T03','all main/RF coordinates; LS learned')
    fig,axes=panels('Nuisance 误差与估计误差','learned')
    settings=[('mse_m_mean','treatment prediction MSE','RMSE','RMSE'),('mse_ell_mean','outcome prediction MSE','RMSE','RMSE'),('alignment_variance_mean','alignment 中心化方差','RMSE','RMSE'),('LS_population_displacement_mean','平均 B(pop)LS','bias','Observed Bias')]
    q=merged[merged.pipeline=='learned']
    for ax,(x,xlabel,y,ylabel) in zip(axes,settings):
        for method,color in zip(METHODS,COLORS):
            a=q[q.method==method];ax.scatter(a[x],a[y],c=color,label=method,s=16,alpha=.55)
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    legend(fig,axes[0]);fig.text(.5,.035,'各点为正式条件/坐标均值，含 Spline 与 RF；关联为描述性，不能据此断言因果。逐次关联见 T09b。',ha='center',fontsize=11)
    finish(fig,'diagnostic_nuisance_error','诊断','Nuisance 与估计误差','T09 + T01/T03/T10','all main/RF coordinates; learned')
    fig,axes=panels('标准误校准与覆盖率','正式主实验 + RF',shape=(1,2))
    for ax,pipe in zip(axes,['learned','oracle']):
        q=allc[allc.pipeline==pipe]
        for m,col in zip(METHODS,COLORS):
            a=q[q.method==m];ax.errorbar(a.SEratio,a.coverage,yerr=1.96*a.coverage_mcse,fmt='none',ecolor=col,alpha=.18,lw=.6);ax.scatter(a.SEratio,a.coverage,c=col,label=m,s=20,alpha=.55)
        ax.axvline(1,color='black',ls=':');ax.axhline(.95,color='#b91c1c',ls='--');ax.set_ylim(0,1.04)
        ax.set_xlabel('Mean SE / Empirical SD');ax.set_ylabel('边际覆盖率');ax.set_title(pipe);ax.grid(alpha=.2)
    legend(fig,axes[0]);fig.text(.5,.04,'每点为条件/方法/坐标；覆盖率和 SE 比率均未按结果调整。完整 Monte Carlo 区间见对应 CSV。',ha='center',fontsize=11)
    finish(fig,'diagnostic_SEratio_coverage','诊断','SE/SD 与覆盖率','T01/T03/T10','main/RF; all methods; learned/oracle')
    fig,axes=panels('覆盖率与区间长度','正式主实验')
    for ax,(pipe,kind) in zip(axes,[(p,k) for p in ['learned','oracle'] for k in ['coordinate','contrast']]):
        q=(allc if kind=='coordinate' else contrast);q=q[(q.role=='main')&(q.pipeline==pipe)]
        for m,col in zip(METHODS,COLORS):
            a=q[q.method==m];ax.errorbar(a.mean_length,a.coverage,yerr=1.96*a.coverage_mcse,fmt='none',ecolor=col,alpha=.18,lw=.6);ax.scatter(a.mean_length,a.coverage,c=col,label=m,s=19,alpha=.6)
        ax.axhline(.95,color='#b91c1c',ls='--');ax.set_ylim(0,1.04);ax.set_xlabel('平均区间长度');ax.set_ylabel('覆盖率');ax.set_title(pipe+' / '+kind);ax.grid(alpha=.2)
    legend(fig,axes[0]);fig.text(.5,.035,'短区间与低覆盖可能同时出现。不同 n、分布和坐标均保留，不将混合点云解释为因果关系。',ha='center',fontsize=11)
    finish(fig,'diagnostic_length_coverage','诊断','区间长度与覆盖率','T01/T03/T04/T06','all main conditions, coordinates and contrasts')
    fig,axes=panels('Learned 与 Oracle RMSE','正式主实验',shape=(1,2))
    for ax,level in zip(axes,['coordinate','joint']):
        q=oracle[(oracle.role=='main')&(oracle.level==level)]
        for m,col in zip(METHODS,COLORS):
            a=q[q.method==m];ax.scatter(a.RMSE_oracle,a.RMSE_learned,c=col,label=m,s=22,alpha=.65)
        hi=max(q.RMSE_oracle.max(),q.RMSE_learned.max())*1.05;ax.plot([0,hi],[0,hi],'k--');ax.set_xlim(0,hi);ax.set_ylim(0,hi)
        ax.set_xlabel('Oracle RMSE');ax.set_ylabel('Learned RMSE');ax.set_title('逐坐标' if level=='coordinate' else '整体 Vector RMSE');ax.grid(alpha=.2)
    legend(fig,axes[0]);fig.text(.5,.04,'Oracle 沿用 learned 的 H/HT 训练选择，只替换 nuisance 残差；不作为可行估计器。',ha='center',fontsize=11)
    finish(fig,'diagnostic_learned_oracle','诊断','Learned 与 Oracle','T08_Learned_Oracle.csv','all main methods and coordinates/vectors')
    fig,axes=panels('协方差条件数与联合覆盖率','正式多元实验',shape=(1,2))
    for ax,pipe in zip(axes,['learned','oracle']):
        q=v[v.pipeline==pipe]
        for m,col in zip(METHODS,COLORS):
            a=q[q.method==m];ax.errorbar(a.cov_condition_median,a.coverage,yerr=1.96*a.coverage_mcse,fmt='none',ecolor=col,alpha=.18,lw=.6);ax.scatter(a.cov_condition_median,a.coverage,c=col,label=m,s=24,alpha=.6)
        ax.set_xscale('log');ax.axhline(.95,color='#b91c1c',ls='--');ax.set_ylim(0,1.04);ax.set_title(pipe)
        ax.set_xlabel('完整协方差条件数的中位数（log 轴）');ax.set_ylabel('Wald 联合覆盖率');ax.grid(alpha=.2)
    legend(fig,axes[0]);fig.text(.5,.04,'联合推断无单一坐标 SE/SD 定义；同时检查各坐标 SE/SD、协方差稳定性及实际联合覆盖率。',ha='center',fontsize=11)
    finish(fig,'diagnostic_joint_stability','诊断','联合协方差稳定性','T02_多元主结果.csv','all d/rho/n/laws and methods')

def sensitivities(rf,tau,step):
    for pipe in ['learned','oracle']:
        fig,axes=panels('RF / Spline RMSE 比及配对 95% 区间',pipe)
        for ax,law,title in zip(axes,LAWS,TITLES):
            q=rf[(rf.pipeline==pipe)&(rf.law==law)]
            for d,level,shift,color,label in [(1,'coordinate',-.09,'#007f78','标量'),(3,'joint',.09,'#2563eb','三维整体')]:
                a=q[(q.d==d)&(q.level==level)].set_index('method').loc[METHODS]
                x=np.arange(5)+shift
                ax.errorbar(x,a.RMSE_ratio_RF_over_Spline,yerr=[a.RMSE_ratio_RF_over_Spline-a.ratio_lower,a.ratio_upper-a.RMSE_ratio_RF_over_Spline],fmt='o',label=label,color=color,capsize=4)
            ax.axhline(1,color='black',ls='--');ax.set_xticks(range(5),METHODS);ax.set_title(title);ax.set_ylabel('RF / Spline RMSE');ax.grid(alpha=.2)
        legend(fig,axes[0]);fig.text(.5,.035,'n=1000；同数据及 folds，按 replication 联合重采样 2000 次。大于 1 表示 RF 误差更大。',ha='center',fontsize=11)
        finish(fig,'RF_RMSE_ratio_'+pipe,'敏感性','RF / Spline 配对比较','T10_RF敏感性.csv',pipe+'; all methods/laws; scalar/vector3 rho0.5')
    for d,level,title in [(1,'coordinate','标量'),(3,'joint','三维联合')]:
        fig,axes=panels('RF 与 Spline 覆盖率：'+title,'learned')
        for ax,law,lawtitle in zip(axes,LAWS,TITLES):
            q=rf[(rf.pipeline=='learned')&(rf.law==law)&(rf.d==d)&(rf.level==level)].set_index('method').loc[METHODS]
            for learner,shift,col in [('RF',-.07,'#d68b20'),('Spline',.07,'#007f78')]:
                ax.errorbar(np.arange(5)+shift,q[learner+'_coverage'],yerr=1.96*q[learner+'_coverage_mcse'],fmt='o-',color=col,label=learner,capsize=4)
            ax.axhline(.95,color='#b91c1c',ls='--');ax.set_ylim(0,1.04);ax.set_xticks(range(5),METHODS);ax.set_title(lawtitle);ax.set_ylabel('覆盖率');ax.grid(alpha=.2)
        legend(fig,axes[0]);fig.text(.5,.035,'n=1000，R=1000；误差条为 ±1.96 MC SE。跨学习器覆盖率差的配对区间另见 T10。',ha='center',fontsize=11)
        finish(fig,f'RF_coverage_d{d}','敏感性','RF / Spline 覆盖率','T10_RF敏感性.csv','learned all methods/laws')
    for pipe in ['learned','oracle']:
        for metric,title,err in [('gamma_mean','Cross-information slope','gamma_mcse'),('SEratio','Mean SE / Empirical SD',None)]:
            fig,axes=panels('步长 c 敏感性：'+title,pipe)
            for ax,(law,d) in zip(axes,[(law,d) for law in ['sln075','sln100'] for d in [1,3]]):
                q=step[(step.pipeline==pipe)&(step.law==law)&(step.d==d)]
                for m,col in [('W','#007f78'),('N','#2563eb')]:
                    for j,style in zip(range(1,d+1),['-','--',':']):
                        a=q[(q.method==m)&(q.coordinate==j)].sort_values('rank_c')
                        ax.errorbar([0,1,2],a[metric],yerr=1.96*a[err] if err else None,fmt='o'+style,color=col,label=m+f' j={j}',capsize=3)
                if metric=='SEratio':ax.axhline(1,color='black',ls='--')
                ax.set_xticks([0,1,2],['0.5','1（主）','2']);ax.set_xlabel('c');ax.set_ylabel(title);ax.set_title(law+f' / d={d}');ax.grid(alpha=.2)
                ax.legend(fontsize=10,ncol=2)
            fig.text(.5,.035,'相同数据/缓存；每条线保留单独坐标。Slope 误差条为均值的 ±1.96 MC SE；不按结果重新选择 c。',ha='center',fontsize=11)
            finish(fig,f'C_{metric}_{pipe}','敏感性','c 敏感性 '+title,'T12_C敏感性.csv',pipe+'; both laws/dimensions; W/N all coordinates')
    for metric,title in [('RMSE','RMSE'),('coverage','覆盖率')]:
        fig,axes=panels('更强平滑 τ 的 Oracle 对照：'+title,'oracle',shape=(1,2))
        for ax,(d,level) in zip(axes,[(1,'coordinate'),(3,'joint')]):
            q=tau[(tau.pipeline=='oracle')&(tau.d==d)&(tau.level==level)]
            for m,col in zip(METHODS,COLORS):
                a=q[q.method==m].sort_values('tau');ax.errorbar(a.tau,a[metric],yerr=1.96*a.coverage_mcse if metric=='coverage' else None,fmt='o-',color=col,label=m,capsize=4)
            ax.set_xticks([.2,.5]);ax.set_xlabel('τ');ax.set_ylabel(title);ax.set_title('标量' if d==1 else '三维整体 / 联合');ax.grid(alpha=.2)
            if metric=='coverage':ax.axhline(.95,color='#b91c1c',ls='--');ax.set_ylim(0,1.04)
            else:ax.set_ylim(bottom=0)
        legend(fig,axes[0]);fig.text(.5,.04,'σ=1，n=1000，R=1000。两种 τ 为不同误差分布的新数据，连线仅用于对照，不表示配对。',ha='center',fontsize=11)
        finish(fig,'Tau_oracle_'+metric,'敏感性','τ Oracle '+title,'T11_Tau敏感性.csv','all five methods; scalar/vector3 rho0.5')

def main():
    scalar=read('01_正式主实验/T01_标量主结果.csv');coords=read('01_正式主实验/T03_多元坐标结果.csv');v=read('01_正式主实验/T02_多元主结果.csv');contrast=read('01_正式主实验/T04_Contrast结果.csv')
    from assemble_tables import load_results,combined
    allcoords=combined(load_results(),'coordinates')
    diag=read('03_Oracle与诊断/T09_Nuisance诊断.csv');oracle=read('03_Oracle与诊断/T08_Learned_Oracle.csv')
    vector_trends(v);diagnostics(allcoords,v,contrast,diag,oracle)
    sensitivities(read('04_敏感性分析/T10_RF敏感性.csv'),read('04_敏感性分析/T11_Tau敏感性.csv'),read('04_敏感性分析/T12_C敏感性.csv'))
    assert len(INDEX)==40,len(INDEX)
    existing=[]
    for stage in ['G0','G1','G2','G3','G4','G5','G6A','G6B']:
        for p in sorted((PROJECT/(stage+'实验')/'图').rglob('*.pdf')):
            existing.append(dict(stage=stage,path=str(p),role='pilot_not_formal' if stage=='G0' else ('formal_main' if stage in ['G1','G2','G3','G4'] else 'sensitivity'),name=p.stem))
    pd.DataFrame(existing).to_csv(OUT/'06_论文图表/已有PDF图表索引.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(INDEX).to_csv(OUT/'06_论文图表/新增论文图表索引.csv',index=False,encoding='utf-8-sig')
    save(OUT/'06_论文图表/figure_manifest.json',dict(new=INDEX,existing_count=len(existing),new_count=len(INDEX)))
    def rel(p):return Path(os.path.relpath(p,OUT/'06_论文图表')).as_posix()
    md='# 论文图表索引\n\n新增 40 张补缺 PDF，全部直接来自本次核验的正式数据汇总；未重新模拟或训练。原有 2768 张 PDF 按原路径索引，不重复搬动。G0 图明确标为 pilot。\n\n'
    md+='## 已有图覆盖\n\n- 标量 n=500/1000/2000 的 RMSE、Coverage、SE/SD 曲线，以及多元逐坐标对应曲线：G4 的 `compare_*` 图。\n- W/N 对 LS/H/HT 的六组 RMSE ratio 与 bootstrap 区间：G1–G4 的 `paired_*` 或 `*paired_RMSE*` 图，涵盖全部条件及 Learned/Oracle。\n- G6A/G6B learned 核心图各四张：原目录保留。\n- [全部原有 PDF 清单](已有PDF图表索引.csv)。\n\n## 新增补缺图\n\n'
    for category in ['主实验','诊断','敏感性']:
        md+='### '+category+'\n\n'+'\n'.join(f'- [{r["title"]} · {r["scope"]}](<{rel(r["file"])}>)' for r in INDEX if r['category']==category)+'\n\n'
    md+='诊断散点为描述性关联。多元没有统一的“Joint SE/SD”标量定义，故保留逐坐标 SE/SD 并补充完整协方差稳定性和联合覆盖率。\n'
    (OUT/'06_论文图表/图表索引.md').write_text(md,encoding='utf-8')

if __name__=='__main__':main()

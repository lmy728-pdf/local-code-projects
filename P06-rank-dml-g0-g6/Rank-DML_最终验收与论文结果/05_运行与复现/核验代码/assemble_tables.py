import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
from audit_records import OUT,PROJECT,ARCHIVE,TMP,STAGES,METHODS,PIPES,folder_for,cell_id,clean,save

ID=['experiment','law','n','d','rho','pipeline','method']
META=['stage','batch','experiment','law','n','d','rho','tau','rank_c','learner','role']

def write(relative,df):
    path=OUT/relative;assert path.resolve().is_relative_to(OUT)
    df.to_csv(path,index=False,encoding='utf-8-sig',na_rep='NA',float_format='%.17g')
    return path

def load_results():return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(TMP.glob('*__*.json'))]

def combined(results,key):
    return pd.DataFrame([dict(row,source_records=str(Path(a['source_folder'])/'records'),recovery_class='already_calculated_not_organized' if key in ['runtime','diagnostics','fold_diagnostics','slopes'] else 'recomputed_from_existing_records') for a in results for row in a[key]])

def levels(coords,vec,contrast):
    a=coords.assign(level='coordinate',component=coords.coordinate.astype(int).astype(str))
    b=vec.assign(level='joint',component='vector',RMSE=vec.vector_RMSE,bias=np.nan)
    c=contrast.assign(level='contrast',component=contrast.contrast)
    return pd.concat([a,b,c],ignore_index=True)

def add_sources(df,sources):
    # Every output row retains a physical original record folder.
    return df

def paired_existing(results,metric_rows):
    allpairs=[]
    for stage,batch in STAGES:
        if stage=='G0':continue
        folder=folder_for(stage,batch);df=pd.read_csv(folder/'summary/paired.csv',encoding='utf-8-sig')
        df['stage']=stage;df['batch']=batch;df['bootstrap_resamples']=2000
        df['bootstrap_seed_policy']='original reporting.py: default_rng(2026111499), joint replication indices'
        df['source_csv']=str(folder/'summary/paired.csv');df['recovery_class']='already_calculated_not_organized'
        df['role']='main' if stage in ['G1','G2','G3','G4'] else 'sensitivity'
        df['rank_c']=float(batch.split('_c')[1]) if stage=='G6B' else 1.
        df['tau']=.5 if stage=='G6A' else .2
        allpairs.append(df)
    pairs=pd.concat(allpairs,ignore_index=True)
    keys=['stage','batch']+ID[:-1]+['method','baseline']
    rmse_columns=[c for c in pairs if c not in ['size_metric','common_interval_success','mean_size_difference','size_lower','size_upper']]
    rmse=pairs.drop_duplicates(keys)[rmse_columns].copy()
    # Point ratios are independently checked against resummarized full vectors or scalars.
    metrics=metric_rows[(metric_rows.level=='joint')|((metric_rows.level=='coordinate')&(metric_rows.d==1))]
    lookup={tuple(r[c] for c in ['batch']+ID):r.RMSE for _,r in metrics.iterrows()}
    checks=[]
    for _,r in rmse.iterrows():
        head=[r.batch]+[r[k] for k in ID[:-1]]
        expected=lookup[tuple(head+[r.method])]/lookup[tuple(head+[r.baseline])]
        assert np.isclose(expected,r.RMSE_ratio,rtol=1e-10,atol=1e-12)
        checks.append(abs(expected-r.RMSE_ratio))
    rmse['point_ratio_reconstruction_max_error']=max(checks)
    size_rows=[]
    mlookup={(r.batch,r.experiment,r.law,r.n,r.d,r.rho,r.pipeline,r.method,r.level,str(r.component)):r for _,r in metric_rows.iterrows()}
    for _,r in pairs.iterrows():
        level='coordinate' if r.size_metric.startswith('coordinate_') else ('joint' if r.size_metric=='ellipsoid_volume' else 'contrast')
        component=r.size_metric.split('_')[-1] if level=='coordinate' else ('vector' if level=='joint' else r.size_metric)
        k=(r.batch,r.experiment,r.law,r.n,r.d,r.rho,r.pipeline)
        a=mlookup[k+(r.method,level,component)];b=mlookup[k+(r.baseline,level,component)]
        size='mean_volume' if level=='joint' else 'mean_length'
        row=r.to_dict();row.update(level=level,component=component,method_coverage=a.coverage,baseline_coverage=b.coverage,
          method_size=a[size],baseline_size=b[size],method_interval_score=a.get('mean_interval_score',np.nan),baseline_interval_score=b.get('mean_interval_score',np.nan),
          interval_score_applicability='NA: ellipsoid has no interval score definition' if level=='joint' else '95% univariate interval score')
        assert np.isclose(row['method_size']-row['baseline_size'],r.mean_size_difference,rtol=1e-9,atol=1e-10)
        size_rows.append(row)
    return rmse,pd.DataFrame(size_rows)

def key_cell(base):return tuple(base[k] for k in ['experiment','law','n','d','rho'])

def cross_learner_bootstrap(results,full):
    baseline={key_cell(a['base']):a for a in results if a['base']['stage'] in ['G1','G2']}
    rf=[a for a in results if a['base']['stage']=='G5'];rows=[]
    for block in rf:
        b=block['base'];old=baseline[key_cell(b)]
        with np.load(block['array_file']) as f:ra={k:f[k] for k in f.files}
        with np.load(old['array_file']) as f:sa={k:f[k] for k in f.files}
        assert np.array_equal(ra['replications'],sa['replications'])
        B=len(ra['replications']);truth=ra['truth'];d=len(truth)
        seed=int.from_bytes(hashlib.sha256(('RF-paired-audit-v1'+json.dumps(key_cell(b))).encode()).digest()[:8],'little')
        gen=np.random.default_rng(seed)
        ix=gen.integers(B,size=(2000,B));counts=np.zeros((2000,B),float)
        np.add.at(counts,(np.arange(2000)[:,None],ix),1)
        weights=counts/B
        q=full[(full.stage=='G5')&(full.experiment==b['experiment'])&(full.law==b['law'])&(full.n==b['n'])&(full.d==b['d'])&(full.rho==b['rho'])]
        bas=full[(full.stage.isin(['G1','G2']))&(full.experiment==b['experiment'])&(full.law==b['law'])&(full.n==b['n'])&(full.d==b['d'])&(full.rho==b['rho'])]
        for pipe,method in itertools.product(PIPES,METHODS):
            index=PIPES.index(pipe)*5+METHODS.index(method)
            for level,component in [('coordinate',str(j+1)) for j in range(d)]+([('joint','vector'),('contrast','plus'),('contrast','minus')] if d>1 else []):
                ar=q[(q.pipeline==pipe)&(q.method==method)&(q.level==level)&(q.component==component)].iloc[0]
                br=bas[(bas.pipeline==pipe)&(bas.method==method)&(bas.level==level)&(bas.component==component)].iloc[0]
                def values(a):
                    if level=='coordinate':
                        j=int(component)-1;e=a['estimate'][:,index,j]-truth[j];coverage=a['coverage'][:,index,j];size=a['length'][:,index,j];se=a['se'][:,index,j];score=a['interval_score'][:,index,j];valid=a['valid'][:,index,j]
                        return e*e,e,coverage.astype(float),size,se,score,valid
                    if level=='contrast':
                        j=0 if component=='plus' else 1;v=np.ones(d)/np.sqrt(d) if j==0 else np.r_[1,-1,np.zeros(d-2)]/np.sqrt(2)
                        e=a['contrast_estimate'][:,index,j]-v@truth
                        return e*e,e,a['contrast_coverage'][:,index,j].astype(float),a['contrast_length'][:,index,j],a['contrast_se'][:,index,j],a['contrast_score'][:,index,j],a['contrast_valid'][:,index,j]
                    e=a['estimate'][:,index]-truth
                    return np.sum(e*e,axis=1),np.full(B,np.nan),a['joint_coverage'][:,index].astype(float),np.exp(a['log_volume'][:,index]),np.full(B,np.nan),np.full(B,np.nan),a['joint_valid'][:,index]
                aa=values(ra);bb=values(sa);ok=np.isfinite(aa[0])&np.isfinite(bb[0]);interval=aa[6]&bb[6]
                # All supplied records pass, and no failure is silently excluded.
                assert ok.all() and interval.all(),'Cross-learner bootstrap needs explicit reduced common-success resampling'
                totals=np.column_stack([aa[0],bb[0],aa[1]-bb[1],aa[2]-bb[2],aa[3]-bb[3],aa[4]-bb[4],aa[5]-bb[5]])
                boot=weights@totals;ratio=np.sqrt(boot[:,0]/boot[:,1]);ci=np.quantile(ratio,[.025,.975])
                row={k:b[k] for k in META};row.update(pipeline=pipe,method=method,level=level,component=component,attempted=B,common_success=B,bootstrap_resamples=2000,bootstrap_seed=str(seed),paired_on='same original dataset, outer folds and replication; whole vector/covariance retained',source_RF=block['source_folder'],source_spline=old['source_folder'])
                for name in ['RMSE','bias','SD','MeanSE','SEratio','coverage','coverage_mcse','mean_length','median_length','mean_interval_score','mean_volume','median_log_volume','valid_estimates','valid_vector_estimates','valid_intervals']:
                    row['RF_'+name]=ar.get(name,np.nan);row['Spline_'+name]=br.get(name,np.nan)
                row.update(RMSE_ratio_RF_over_Spline=np.sqrt(aa[0].mean()/bb[0].mean()),ratio_lower=ci[0],ratio_upper=ci[1],paired_MSE_difference=(aa[0]-bb[0]).mean(),paired_MSE_mcse=(aa[0]-bb[0]).std(ddof=1)/np.sqrt(B))
                for col,name in enumerate(['bias_difference','coverage_difference','size_difference','MeanSE_difference','interval_score_difference'],2):
                    sample=totals[:,col];row[name]=np.mean(sample)
                    row[name+'_lower'],row[name+'_upper']=np.quantile(boot[:,col],[.025,.975])
                if level!='joint':
                    errors=np.column_stack([aa[1],aa[1]**2,bb[1],bb[1]**2,aa[4],bb[4]])
                    moments=weights@errors
                    sd_a=np.sqrt(np.maximum(0,moments[:,1]-moments[:,0]**2)*B/(B-1));sd_b=np.sqrt(np.maximum(0,moments[:,3]-moments[:,2]**2)*B/(B-1))
                    delta=moments[:,4]/sd_a-moments[:,5]/sd_b
                    row['SEratio_difference']=ar.SEratio-br.SEratio;row['SEratio_difference_lower'],row['SEratio_difference_upper']=np.quantile(delta,[.025,.975])
                else:
                    row['SEratio_difference']=row['SEratio_difference_lower']=row['SEratio_difference_upper']=np.nan
                if level=='coordinate':
                    j=int(component)-1
                    for i,name in [(0,'treatment_MSE'),(1,'outcome_MSE'),(2,'alignment_variance')]:
                        row['RF_'+name]=ra['nuisance'][:,j,i].mean();row['Spline_'+name]=sa['nuisance'][:,j,i].mean()
                rows.append(row)
        print('Cross-learner paired bootstrap completed:',key_cell(b),flush=True)
    return pd.DataFrame(rows)

def main():
    results=load_results();assert len(results)==198,len(results)
    coords=combined(results,'coordinates');vectors=combined(results,'vectors');contrasts=combined(results,'contrasts')
    full=levels(coords,vectors,contrasts)
    main=coords.role=='main';vmain=vectors.role=='main';cmain=contrasts.role=='main'
    write('01_正式主实验/T01_标量主结果.csv',coords[main&(coords.d==1)])
    write('01_正式主实验/T02_多元主结果.csv',vectors[vmain])
    write('01_正式主实验/T03_多元坐标结果.csv',coords[main&(coords.d>1)])
    write('01_正式主实验/T04_Contrast结果.csv',contrasts[cmain])
    write('01_正式主实验/T07_样本量趋势.csv',full[full.role=='main'])
    write('00_总览与验收/G0_实现与pilot指标_不用于正式结论.csv',full[full.stage=='G0'])
    paired,sizes=paired_existing(results,full)
    write('02_方法配对比较/T05_配对RMSE比较.csv',paired)
    write('02_方法配对比较/T06_区间与椭球比较.csv',sizes)
    keys=['stage','batch','experiment','law','n','d','rho','tau','rank_c','learner','role','method','level','component']
    measures=['bias','RMSE','SD','MeanSE','SEratio','coverage','coverage_mcse','mean_length','median_length','mean_interval_score','mean_volume','median_log_volume','vector_bias_norm','valid_estimates','valid_vector_estimates','valid_intervals']
    joined=full[full.pipeline=='learned'][keys+measures].merge(full[full.pipeline=='oracle'][keys+measures],on=keys,suffixes=('_learned','_oracle'),validate='one_to_one')
    joined['RMSE_ratio_learned_over_oracle']=joined.RMSE_learned/joined.RMSE_oracle
    joined['coverage_difference_learned_minus_oracle']=joined.coverage_learned-joined.coverage_oracle
    write('03_Oracle与诊断/T08_Learned_Oracle.csv',joined)
    diagnostics=combined(results,'diagnostics');folds=combined(results,'fold_diagnostics');relations=combined(results,'diagnostic_relations')
    write('03_Oracle与诊断/T09_Nuisance诊断.csv',diagnostics)
    write('03_Oracle与诊断/T09a_Foldwise诊断.csv',folds)
    write('03_Oracle与诊断/T09b_Nuisance与估计误差关联.csv',relations)
    slopes=combined(results,'slopes');slopes=slopes.drop(columns=['source_records','recovery_class'])
    cs=coords.merge(slopes,on=META+['pipeline','method','coordinate'],validate='one_to_one')
    selected=full[(full.stage=='G6A')|((full.stage.isin(['G1','G2']))&(full.law=='sln100')&((full.d==1)|((full.d==3)&(full.rho==.5))))].copy()
    selected['law_family']='sln100';keys=['experiment','n','d','rho','pipeline','method','level','component']
    baseline=selected[selected.tau==.2][keys+measures].rename(columns={c:'baseline_'+c for c in measures})
    selected=selected.merge(baseline,on=keys,validate='many_to_one')
    for metric in ['RMSE','MeanSE','SEratio','mean_length','mean_volume']:
        selected[metric+'_relative_change']=selected[metric]/selected['baseline_'+metric]-1
    selected['coverage_difference']=selected.coverage-selected.baseline_coverage
    selected['coverage_difference_mcse_independent']=np.sqrt(selected.coverage_mcse**2+selected.baseline_coverage_mcse**2)
    selected.loc[selected.tau==.2,'coverage_difference_mcse_independent']=0.
    selected['comparison_design']='tau=0.5 new-distribution independent samples; no paired test against tau=0.2'
    write('04_敏感性分析/T11_Tau敏感性.csv',selected)
    step=cs[(cs.stage=='G6B')|((cs.stage.isin(['G1','G2']))&(cs.law.isin(['sln075','sln100']))&((cs.d==1)|((cs.d==3)&(cs.rho==.5))))].copy()
    step=step[step.method.isin(['W','N'])];step['primary_analysis']=step.rank_c==1
    write('04_敏感性分析/T12_C敏感性.csv',step)
    joint_step=vectors[(vectors.stage=='G6B')|((vectors.stage=='G2')&(vectors.law.isin(['sln075','sln100']))&(vectors.d==3)&(vectors.rho==.5))]
    write('04_敏感性分析/T12a_C联合推断.csv',joint_step)
    write('05_运行与复现/T13_失败及收敛.csv',combined(results,'failures'))
    write('05_运行与复现/T14_运行成本.csv',combined(results,'runtime'))
    inventory=json.loads((TMP/'批次扫描.json').read_text(encoding='utf-8'));matrix=[];configuration=[]
    for info in inventory:
        subset=[r for r in results if r['base']['batch']==info['batch']];config=info['config']
        issues=[i for r in subset for i in r['issues']]
        matrix.append(dict(stage=info['stage'],batch=info['batch'],cells=len(subset),replications_per_cell=config['R'],expected=info['expected'],completed=info['actual'],method_records=sum(r['method_records'] for r in subset),cache_checks=sum(r['cache_checks'] for r in subset),cached_method_replays=sum(r['cached_method_replays'] for r in subset),reuse_checks=sum(r['reused_data_checks'] for r in subset),missing_records=len(info['missing']),unexpected_records=len(info['unexpected']),mathematical_inconsistencies=len(issues),status='PASS' if not issues and not info['missing'] and not info['unexpected'] else 'MISMATCH',source=info['source_folder']))
        for key,value in config.items():configuration.append(dict(stage=info['stage'],batch=info['batch'],item=key,value=json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value,source=str(Path(info['source_folder'])/'config.json')))
    write('00_总览与验收/实验完整性矩阵.csv',pd.DataFrame(matrix))
    write('05_运行与复现/T15_配置与核验.csv',pd.DataFrame(configuration))
    source_index={row['path']:row for r in results for row in r['source_files']}
    for parent in [PROJECT,ARCHIVE]:
        for p in parent.rglob('*'):
            if not p.is_file() or p.is_relative_to(OUT):continue
            if any(part in ['dependencies','plot_packages','__pycache__','.git'] for part in p.relative_to(parent).parts):continue
            if str(p) in source_index:continue
            if p.suffix.lower() not in ['.py','.ipynb','.json','.csv','.md','.html','.pdf','.svg','.log','.tex','.npz','.zip','.ps1','.cmd','.txt']:continue
            st=p.stat();source_index[str(p)]=dict(path=str(p),kind=p.suffix.lower().lstrip('.'),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if st.st_size<2_000_000 else None)
    pdf=Path(r'C:\Users\Lenovo\Desktop\Rank_DML1.pdf');source_index[str(pdf)]=dict(path=str(pdf),kind='teacher_pdf',bytes=pdf.stat().st_size,mtime_ns=pdf.stat().st_mtime_ns,sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    write('00_总览与验收/数据来源与文件索引.csv',pd.DataFrame(source_index.values()))
    # Reconcile all original summary values shared with the expanded tables.
    reconciliation=[]
    for stage,batch in STAGES:
        folder=folder_for(stage,batch)
        for name,new,metric_keys in [('coordinates',coords,ID+['coordinate']),('vectors',vectors,ID),('contrasts',contrasts,ID+['contrast'])]:
            path=folder/'summary'/(name+'.csv')
            try:old=pd.read_csv(path,encoding='utf-8-sig')
            except pd.errors.EmptyDataError:continue
            if old.empty:continue
            if name=='vectors':old=old[old.d>1]
            newpart=new[new.batch==batch];shared=[c for c in old if c in newpart and c not in metric_keys and pd.api.types.is_numeric_dtype(old[c])]
            merged=old.merge(newpart,on=metric_keys,suffixes=('_old','_new'),validate='one_to_one')
            assert len(merged)==len(old),(stage,name,'row reconciliation')
            for metric in shared:
                a=merged[metric+'_old'].to_numpy(float);b=merged[metric+'_new'].to_numpy(float)
                equal=np.isclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True)
                reconciliation.append(dict(stage=stage,batch=batch,table=name,metric=metric,rows=len(a),mismatches=int((~equal).sum()),max_absolute_difference=float(np.nanmax(abs(a-b))) if len(a) else np.nan,source=str(path)))
    write('00_总览与验收/原汇总与重算一致性.csv',pd.DataFrame(reconciliation))
    rf=cross_learner_bootstrap(results,full)
    write('04_敏感性分析/T10_RF敏感性.csv',rf)
    save(TMP/'tables_manifest.json',dict(cells=len(results),datasets=sum(r['completed'] for r in results),method_records=sum(r['method_records'] for r in results),issues=[dict(stage=r['base']['stage'],batch=r['base']['batch'],cell=cell_id(r['base']),issues=r['issues']) for r in results if r['issues']],source_files=len(source_index),original_summary_mismatches=sum(r['mismatches'] for r in reconciliation)))
    print('T01-T15 and all supplemental data tables completed.',flush=True)

if __name__=='__main__':main()

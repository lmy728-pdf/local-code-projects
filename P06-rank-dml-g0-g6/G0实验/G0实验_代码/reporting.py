import json,itertools
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from run import save,ROOT

def coverage(v,valid):
    v=np.array(v,bool);valid=np.array(valid,bool);B=len(v);s=int(v.sum());p=s/B
    return dict(attempted=B,valid_intervals=int(valid.sum()),coverage=p,coverage_valid_only=s/valid.sum() if valid.sum() else np.nan,coverage_mcse=np.sqrt(p*(1-p)/B),coverage_lower=beta.ppf(.025,s,B-s+1) if s else 0,coverage_upper=beta.ppf(.975,s+1,B-s) if s<B else 1)

def summarize(folder):
    folder=Path(folder);groups={};failures=Counter();diagnostics=[]
    for path in sorted((folder/'records').glob('*.json')):
        data=json.loads(path.read_text(encoding='utf-8'));cell=data['cell'];key=tuple(cell[k] for k in ('experiment','law','n','d','rho'))
        groups.setdefault(key,[]).append(data)
        if 'diagnostics' in data:
            diagnostics.append(dict(**cell,replication=data['replication'],LS_population_shift=data['diagnostics']['LS_population_shift'],nuisance_seconds=data['nuisance_seconds'],auxiliary_seconds=data['auxiliary_seconds'],tuning_seconds=data['tuning_seconds'],elapsed=data['elapsed']))
    coords=[];vectors=[];contrasts=[];paired=[];runtime=[];rr=np.random.default_rng(2026111499)
    for key,datasets in groups.items():
        datasets.sort(key=lambda a:a['replication']);cell=dict(zip(('experiment','law','n','d','rho'),key));d=cell['d'];B=len(datasets);lookup={}
        for pipeline in ('learned','oracle'):
            for method in ('LS','H','HT','W','N'):
                rows=[next(a for a in ds['records'] if a['method']==method and a['pipeline']==pipeline) for ds in datasets];lookup[pipeline,method]=rows
                p=np.array([a['estimate'] for a in rows],float);true=np.array(rows[0]['truth']);err=p-true;valid_est=np.isfinite(p);valid_vec=valid_est.all(1);base=dict(**cell,pipeline=pipeline,method=method)
                for row in rows:
                    for failure in row['failures']:failures[pipeline+'|'+method+'|'+failure['type']]+=1
                for j in range(d):
                    mask=valid_est[:,j];e=err[mask,j];sd=np.std(p[mask,j],ddof=1) if mask.sum()>1 else np.nan
                    se=np.array([a['se'][j] for a in rows],float);length=np.array([a['length'][j] for a in rows],float);score=np.array([a['interval_score'][j] for a in rows],float)
                    coords.append(dict(**base,coordinate=j+1,valid_estimates=int(mask.sum()),metric_population='conditional-on-success',bias=np.mean(e),SD=sd,RMSE=np.sqrt(np.mean(e*e)),MAE=np.mean(abs(e)),median_absolute_error=np.median(abs(e)),MeanSE=np.nanmean(se),SEratio=np.nanmean(se)/sd,mean_length=np.nanmean(length),mean_interval_score=np.nanmean(score),failure_rate=1-np.mean([a['valid_interval'][j] for a in rows]),**coverage([a['coverage'][j] for a in rows],[a['valid_interval'][j] for a in rows])))
                vector=dict(**base,attempted=B,valid_vector_estimates=int(valid_vec.sum()),vector_bias_norm=np.linalg.norm(np.mean(err[valid_vec],0)),vector_RMSE=np.sqrt(np.mean(np.sum(err[valid_vec]**2,1))),metric_population='conditional-on-success')
                if d>1:
                    vector.update(coverage([a['joint_coverage'] for a in rows],[a['joint_valid'] for a in rows]));vector.update(mean_volume=np.nanmean(np.array([a['volume'] for a in rows],float)),median_log_volume=np.nanmedian(np.array([a['log_volume'] for a in rows],float)),all_marginal_coverage=np.mean([a['all_marginal'] for a in rows]))
                    for name in ('plus','minus'):
                        cr=[a['contrasts'][name] for a in rows]
                        contrasts.append(dict(**base,contrast=name,mean_length=np.nanmean(np.array([a['length'] for a in cr],float)),mean_interval_score=np.nanmean(np.array([a['interval_score'] for a in cr],float)),**coverage([a['coverage'] for a in cr],[a['valid'] for a in cr])))
                vectors.append(vector);runtime.append(dict(**base,second_stage_seconds=np.mean([a.get('seconds',np.nan) for a in rows]),full_pipeline_seconds=np.mean([a.get('full_pipeline_seconds',np.nan) for a in rows])))
            for method,baseline in itertools.product(('W','N'),('LS','H','HT')):
                a,b=lookup[pipeline,method],lookup[pipeline,baseline];pa=np.array([x['estimate'] for x in a],float);pb=np.array([x['estimate'] for x in b],float);valid=np.isfinite(pa).all(1)&np.isfinite(pb).all(1);count=int(valid.sum());base=dict(**cell,pipeline=pipeline,method=method,baseline=baseline,attempted=B,common_estimate_success=count,method_failure_rate=1-np.isfinite(pa).all(1).mean(),baseline_failure_rate=1-np.isfinite(pb).all(1).mean())
                ea=np.sum((pa[valid]-true)**2,1);eb=np.sum((pb[valid]-true)**2,1)
                if count:
                    ix=rr.integers(count,size=(2000,count));ratios=np.sqrt(ea[ix].mean(1)/eb[ix].mean(1));diff=ea-eb
                    base.update(RMSE_ratio=np.sqrt(ea.mean()/eb.mean()),ratio_lower=np.quantile(ratios,.025),ratio_upper=np.quantile(ratios,.975),paired_MSE_difference=diff.mean(),paired_MSE_mcse=diff.std(ddof=1)/np.sqrt(count) if count>1 else np.nan)
                sizes=[('coordinate_'+str(j+1),np.array([x['length'][j] for x in a],float),np.array([x['length'][j] for x in b],float)) for j in range(d)]
                if d>1:
                    sizes += [(name,np.array([x['contrasts'][name]['length'] for x in a],float),np.array([x['contrasts'][name]['length'] for x in b],float)) for name in ('plus','minus')]
                    sizes += [('ellipsoid_volume',np.array([x['volume'] for x in a],float),np.array([x['volume'] for x in b],float))]
                for name,sa,sb in sizes:
                    ok=np.isfinite(sa)&np.isfinite(sb);dif=sa[ok]-sb[ok];row=dict(**base,size_metric=name,common_interval_success=int(ok.sum()))
                    if len(dif):
                        ix=rr.integers(len(dif),size=(2000,len(dif)));boot=dif[ix].mean(1);row.update(mean_size_difference=dif.mean(),size_lower=np.quantile(boot,.025),size_upper=np.quantile(boot,.975))
                    paired.append(row)
    target=folder/'summary';target.mkdir(exist_ok=True)
    for name,rows in [('coordinates',coords),('vectors',vectors),('contrasts',contrasts),('paired',paired),('runtime',runtime)]:pd.DataFrame(rows).to_csv(target/(name+'.csv'),index=False,encoding='utf-8-sig')
    save(target/'nuisance_diagnostics.json',diagnostics);save(target/'failures.json',dict(failures))
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'));expected=len(config['cells'])*config['R'];completed=sum(map(len,groups.values()))
    audit=dict(expected_datasets=expected,completed_datasets=completed,cells=len(groups),records=sum(len(ds['records']) for gg in groups.values() for ds in gg),all_cells_expected_R=all(len(g)==config['R'] for g in groups.values()),failure_counts=dict(failures),scientific_conclusions='G0 implementation only; not main experiment evidence')
    assert completed==expected and audit['records']==10*expected and audit['all_cells_expected_R'];save(target/'audit.json',audit)
    print('Summary completed:',folder.name,audit,flush=True)

if __name__=='__main__':
    import sys
    summarize(Path(sys.argv[1]))

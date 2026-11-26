"""G3 audit and delivery, retaining every cell and every failure."""
import sys,json,zipfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import ROOT,save,source_hash
from core import *
import pandas as pd
folder=ROOT/'runs'/'g3_n500'
config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
assert config['status']=='completed' and config['R']==1000 and config['workers']==12
assert config['source_hash']==source_hash()
expected_cells={json.dumps(c,sort_keys=True) for c in config['cells']}
assert len(expected_cells)==20 and all(c['n']==500 for c in config['cells'])
snapshot=ROOT/'source_versions'/'g3_start'
for name in ('core.py','reporting.py','run.py'):
    assert (ROOT/name).read_bytes()==(snapshot/name).read_bytes()
paths=sorted((folder/'records').glob('*.json'));assert len(paths)==20000
seen=set();counts={};replayed=0;failed=0;invalid_candidates=0;joint_valid=0;nonzero_offdiagonal=0
for i,path in enumerate(paths,1):
    data=json.loads(path.read_text(encoding='utf-8'));cell=data['cell'];d=cell['d'];rep=data['replication'];key=(d,cell['rho'],cell['law'])
    assert data['stage']=='g3_n500' and json.dumps(cell,sort_keys=True) in expected_cells and 0<=rep<1000
    assert (key,rep) not in seen;seen.add((key,rep));counts[key]=counts.get(key,0)+1
    assert len(data['records'])==10
    if 'traceback' in data:failed+=1;continue
    cp=folder/'cache'/(path.stem+'.npz')
    with zipfile.ZipFile(cp) as z:assert z.testzip() is None
    for pipe in ('learned','oracle'):
        rows=[a for a in data['records'] if a['pipeline']==pipe]
        assert {a['method'] for a in rows}==set(METHODS)
        assert len({a['input_hash'] for a in rows})==1
        for row in rows:
            V=np.array(row['covariance'],float);p=np.array(row['estimate'],float);true=np.array(row['truth']);se=np.array(row['se'],float)
            assert V.shape==(d,d) and np.allclose(V,V.T,equal_nan=True)
            assert np.allclose(se,np.sqrt(np.diag(V)/500),equal_nan=True)
            nonzero_offdiagonal+=bool(np.isfinite(V).all() and np.any(abs(V-np.diag(np.diag(V)))>1e-12))
            if d>1 and row['joint_valid']:
                L=np.linalg.cholesky(V);stat=500*np.sum(np.linalg.solve(L,p-true)**2)
                assert bool(stat<=chi2.ppf(.95,d))==row['joint_coverage'];joint_valid+=1
                logvol=d/2*np.log(np.pi)-gammaln(d/2+1)+d/2*np.log(chi2.ppf(.95,d))+np.log(np.diag(L)).sum()-d/2*np.log(500)
                assert np.isclose(logvol,row['log_volume'])
                for name,a in [('plus',np.ones(d)/np.sqrt(d)),('minus',np.r_[1,-1,np.zeros(d-2)]/np.sqrt(2))]:
                    cr=row['contrasts'][name];s=np.sqrt(a@V@a/500)
                    assert np.isclose(s,cr['se']) and np.isclose(a@p,cr['estimate'])
                    assert bool(abs(a@(p-true))<=ndtri(.975)*s)==cr['coverage']
    for fold in data['training_choices']:
        for coord in fold['coordinates']:invalid_candidates+=sum(not x['converged'] for x in coord['candidates'])
    if rep in (0,499,999):
        cache=np.load(cp);f=cache['folds']
        for row in data['records']:
            oracle=row['pipeline']=='oracle';w=cache['w_oracle' if oracle else 'w'];z=cache['z_oracle' if oracle else 'z'];m=row['method']
            threshold=None if m not in ('H','HT') else cache['scales'][f]*(1.345 if m=='H' else cache['multipliers'][f])
            p,V,_,_=estimate(w,z,f,m,threshold)
            assert np.allclose(p,np.array(row['estimate'],float),atol=1e-12,equal_nan=True)
            assert np.allclose(V,np.array(row['covariance'],float),atol=1e-12,equal_nan=True)
            replayed+=1
    if i%2000==0:print('G3 audited',i,'/20000',flush=True)
assert len(counts)==20 and all(v==1000 for v in counts.values())
audit=json.loads((folder/'summary'/'audit.json').read_text(encoding='utf-8'))
audit.update(scientific_conclusions='G3 formal scalar/vector n=500 main block; joint ellipsoid distinct from simultaneous marginals',cells_by_key={str(k):v for k,v in counts.items()},replayed_method_records=replayed,failed_datasets=failed,invalid_training_candidates=invalid_candidates,valid_joint_records=joint_valid,nonzero_offdiagonal_records=nonzero_offdiagonal,source_hash=source_hash(),cache_crc_checked=True,shared_inputs_verified=True,contrasts_and_joint_recomputed=True)
save(folder/'summary'/'audit.json',audit);save(ROOT/'reports'/'G3_final_audit.json',audit)
coords=pd.read_csv(folder/'summary'/'coordinates.csv');vectors=pd.read_csv(folder/'summary'/'vectors.csv')
diag=json.loads((folder/'summary'/'nuisance_diagnostics.json').read_text(encoding='utf-8'));dr=[]
for item in diag:
    for j,value in enumerate(item['LS_population_shift']):dr.append(dict(experiment=item['experiment'],d=item['d'],rho=item['rho'],law=item['law'],coordinate=j+1,population_shift=value))
keys=['experiment','d','rho','law','coordinate'];drift=pd.DataFrame(dr).groupby(keys,as_index=False).agg(mean_population_shift=('population_shift','mean'))
for pipe in ('learned','oracle'):
    ls=coords[(coords.method=='LS')&(coords.pipeline==pipe)][keys+['bias','SD','MeanSE','SEratio','coverage']]
    drift=drift.merge(ls.rename(columns={k:k+'_'+pipe for k in ls.columns if k not in keys}),on=keys)
drift.to_csv(folder/'summary'/'LS_drift_comparison.csv',index=False,encoding='utf-8-sig')
base_coords=pd.concat([pd.read_csv(ROOT/'runs'/s/'summary'/'coordinates.csv') for s in ('g1_scalar_n1000','g2_vector_n1000')])
ck=['experiment','d','rho','law','pipeline','method','coordinate'];metrics=['bias','SD','RMSE','MeanSE','SEratio','coverage','coverage_mcse','mean_length']
comparison=coords[ck+metrics].merge(base_coords[ck+metrics],on=ck,suffixes=('_n500','_n1000'),validate='one_to_one')
assert len(comparison)==len(coords)==440
comparison.to_csv(folder/'summary'/'formal_n500_vs_n1000_coordinates.csv',index=False,encoding='utf-8-sig')
v2=pd.read_csv(ROOT/'runs'/'g2_vector_n1000'/'summary'/'vectors.csv');vk=['d','rho','law','pipeline','method'];vm=['vector_RMSE','coverage','coverage_mcse','mean_volume','median_log_volume']
vectors[vectors.experiment=='vector'][vk+vm].merge(v2[vk+vm],on=vk,suffixes=('_n500','_n1000'),validate='one_to_one').to_csv(folder/'summary'/'formal_n500_vs_n1000_joint.csv',index=False,encoding='utf-8-sig')
lines=['# G3 小样本正式实验验收报告','','G3 已完成：n=500，4 个标量条件与 16 个多元条件，各 1000 次全新独立重复。共 20000 份数据、200000 条方法/流程记录，12 个单线程 worker。G4 未启动。','',
f'完整核验记录唯一性、共享输入、cache CRC、covariance 对称性及 SE；重算多元 Wald 二次型、log-volume 与两个 contrasts。有效联合区间 {joint_valid}/160000；缓存复算 {replayed} 条方法记录并重现。', '',
f'数据集失败 {failed}；无效 Huber 候选 {invalid_candidates}；方法失败分类 {audit["failure_counts"]}。总体覆盖率包含所有尝试，失败区间计为未覆盖，另报 valid-only coverage。','',
'## 标量结果','','| law | pipeline | method | Bias | SD | RMSE | MeanSE | Coverage | Length |','|---|---|---|---:|---:|---:|---:|---:|---:|']
for _,a in coords[coords.experiment=='scalar'].sort_values(['law','pipeline','method']).iterrows():lines.append(f'| {a.law} | {a.pipeline} | {a.method} | {a.bias:.4f} | {a.SD:.4f} | {a.RMSE:.4f} | {a.MeanSE:.4f} | {a.coverage:.3f} | {a.mean_length:.4f} |')
lines+=['','## 多元联合结果','','coverage 是 95% Wald ellipsoid coverage，不是所有 marginal 同时覆盖。','',
'| d | rho | law | pipeline | method | Vector RMSE | Joint coverage | MC SE | Mean volume | Median log-volume |','|---:|---:|---|---|---|---:|---:|---:|---:|---:|']
for _,a in vectors[vectors.experiment=='vector'].sort_values(['d','rho','law','pipeline','method']).iterrows():lines.append(f'| {a.d} | {a.rho} | {a.law} | {a.pipeline} | {a.method} | {a.vector_RMSE:.4f} | {a.coverage:.3f} | {a.coverage_mcse:.4f} | {a.mean_volume:.6f} | {a.median_log_volume:.3f} |')
lines+=['','## 图表、样本量对照与复现','',
'- [独立大幅 SVG 图表](../vector_figures/G3/index.html) · [G3 矢量图压缩包](G3_vector_figures.zip)。',
'- runs/g3_n500/summary/coordinates.csv、vectors.csv、contrasts.csv：完整逐坐标、联合椭球和 contrast 指标。',
'- formal_n500_vs_n1000_coordinates.csv、formal_n500_vs_n1000_joint.csv：与 G1/G2 正式 n=1000 的同条件对照，不混用 G0 pilot；两样本量的数据相互独立。',
'- paired.csv：W/N 对 LS/H/HT 的 RMSE ratio、2000 次联合 bootstrap、MSE 差异 MC SE 和区间大小对照，含 common-success 分母。',
'- LS_drift_comparison.csv：条件总体 LS 位移与实际 Bias/SE 校准对照。',
'- records/、cache/：逐次结果、完整 covariance、阈值选择、诊断和压缩残差持续保留。',
'- reports/G3_preflight.json、G3_final_audit.json、G3_run.log 与 source_versions/g3_start：启动检查、验收、日志和源码快照。','',
f'源码 SHA-256：`{source_hash()}`。统计定义与 G0 冻结核心相同，独立 seed namespace 为 g3_n500。', '',
'样本量趋势当前只覆盖两个正式样本量 n=500/1000；n=2000 正式实验尚未运行。覆盖与区间大小应共同解释，较窄但欠覆盖不构成推断优势。']
(ROOT/'reports'/'G3_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('G3 audit/report complete',audit,flush=True)

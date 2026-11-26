"""G2 audit and delivery, retaining every cell and every failure."""
import sys,json,zipfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import ROOT,save,source_hash
from core import *
import pandas as pd
folder=ROOT/'runs'/'g2_vector_n1000'
config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
assert config['status']=='completed' and config['R']==1000 and config['workers']==12
assert config['source_hash']==source_hash()
paths=sorted((folder/'records').glob('*.json'));assert len(paths)==16000
seen=set();counts={};replayed=0;failed=0;invalid_candidates=0;joint_valid=0;nonzero_offdiagonal=0
for i,path in enumerate(paths,1):
    data=json.loads(path.read_text(encoding='utf-8'));cell=data['cell'];d=cell['d'];rep=data['replication'];key=(d,cell['rho'],cell['law'])
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
            assert np.allclose(se,np.sqrt(np.diag(V)/1000),equal_nan=True)
            nonzero_offdiagonal+=bool(np.isfinite(V).all() and np.any(abs(V-np.diag(np.diag(V)))>1e-12))
            if row['joint_valid']:
                L=np.linalg.cholesky(V);stat=1000*np.sum(np.linalg.solve(L,p-true)**2)
                assert bool(stat<=chi2.ppf(.95,d))==row['joint_coverage'];joint_valid+=1
                logvol=d/2*np.log(np.pi)-gammaln(d/2+1)+d/2*np.log(chi2.ppf(.95,d))+np.log(np.diag(L)).sum()-d/2*np.log(1000)
                assert np.isclose(logvol,row['log_volume'])
                for name,a in [('plus',np.ones(d)/np.sqrt(d)),('minus',np.r_[1,-1,np.zeros(d-2)]/np.sqrt(2))]:
                    cr=row['contrasts'][name];s=np.sqrt(a@V@a/1000)
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
    if i%2000==0:print('G2 audited',i,'/16000',flush=True)
assert len(counts)==16 and all(v==1000 for v in counts.values())
audit=json.loads((folder/'summary'/'audit.json').read_text(encoding='utf-8'))
audit.update(scientific_conclusions='G2 formal vector n=1000 main block; joint ellipsoid distinct from simultaneous marginals',cells_by_key={str(k):v for k,v in counts.items()},replayed_method_records=replayed,failed_datasets=failed,invalid_training_candidates=invalid_candidates,valid_joint_records=joint_valid,nonzero_offdiagonal_records=nonzero_offdiagonal,source_hash=source_hash(),cache_crc_checked=True,shared_inputs_verified=True,contrasts_and_joint_recomputed=True)
save(folder/'summary'/'audit.json',audit);save(ROOT/'reports'/'G2_final_audit.json',audit)
coords=pd.read_csv(folder/'summary'/'coordinates.csv');vectors=pd.read_csv(folder/'summary'/'vectors.csv')
diag=json.loads((folder/'summary'/'nuisance_diagnostics.json').read_text(encoding='utf-8'))
dr=[]
for item in diag:
    for j,value in enumerate(item['LS_population_shift']):dr.append(dict(d=item['d'],rho=item['rho'],law=item['law'],coordinate=j+1,population_shift=value))
keys=['d','rho','law','coordinate'];drift=pd.DataFrame(dr).groupby(keys,as_index=False).agg(mean_population_shift=('population_shift','mean'))
for pipe in ('learned','oracle'):
    ls=coords[(coords.method=='LS')&(coords.pipeline==pipe)][keys+['bias','SD','MeanSE','SEratio','coverage']]
    drift=drift.merge(ls.rename(columns={k:k+'_'+pipe for k in ls.columns if k not in keys}),on=keys)
drift.to_csv(folder/'summary'/'LS_drift_comparison.csv',index=False,encoding='utf-8-sig')
lines=['# G2 正式多元实验验收报告','','G2：n=1000；d=2,3；rho=0,0.5；四种误差分布；16 cells 各 1000 次。共 16000 份数据、160000 条方法/流程记录，采用 12 个单线程计算 worker。G3 未启动。','',
f'逐条检查唯一性、方法共享输入、完整 covariance 的对称性和 SE 对角一致性；CRC 核验缓存；重算 Wald 二次型、log-volume 及两 contrasts。有效联合区间 {joint_valid}/160000，含非零非对角项的 covariance {nonzero_offdiagonal} 条。缓存抽样复算 {replayed} 条方法记录并重现。','',
f'数据集失败 {failed}；无效 Huber 调参候选 {invalid_candidates}；方法失败分类：{audit["failure_counts"]}。失败记录保留，总 coverage 以全部尝试次数为分母，另有 valid-only coverage。','',
'## 正式联合结果','','下表 coverage 是 95% Wald ellipsoid coverage，不是所有 marginal 同时覆盖。逐坐标及两种 contrast 的完整指标另附 CSV。','',
'| d | rho | law | pipeline | method | Vector RMSE | Joint coverage | MC SE | Mean volume | Median log-volume |','|---:|---:|---|---|---|---:|---:|---:|---:|---:|']
for _,a in vectors.sort_values(['d','rho','law','pipeline','method']).iterrows():lines.append(f'| {a.d} | {a.rho} | {a.law} | {a.pipeline} | {a.method} | {a.vector_RMSE:.4f} | {a.coverage:.3f} | {a.coverage_mcse:.4f} | {a.mean_volume:.6f} | {a.median_log_volume:.3f} |')
lines+=['','## 表格、矢量图与复现','',
'- [独立大幅 SVG 图表索引](../vector_figures/index.html)：按分布、d/rho、指标、Learned/Oracle 拆图。',
'- [G2 矢量图打包](G2_vector_figures.zip)。',
'- runs/g2_vector_n1000/summary/coordinates.csv：各坐标点估计精度、SE/SD、覆盖和区间长度。',
'- contrasts.csv：a_plus、a_minus 的区间结果；vectors.csv：vector RMSE、联合 coverage/volume、同时 marginal 描述比例。',
'- paired.csv：W/N 对 LS/H/HT 的 vector RMSE ratio、2000 次联合 bootstrap、MSE 差异 MC SE、各 coordinate/contrast/volume 大小比较与共同成功分母。',
'- LS_drift_comparison.csv：nuisance 条件总体位移与实际 LS bias/SE calibration 对照。',
'- records/、cache/：所有数据重复的逐次输出、完整协方差、阈值调参、诊断和压缩残差；持续保留以便复算。',
'- config.json、source_versions/g2_start、G2_preflight.json、G2_resume_audit.json、G2_final_audit.json：配置、版本、启动/恢复/最终核验。','',
f'源码 SHA-256：`{source_hash()}`。统计核心沿用 G0 冻结版本，G2 使用独立 namespace g2_vector_n1000。任务曾因用户要求补图而暂停，恢复前验证并保留 1983 份结果；其余使用原种子补跑，没有重复计数。', '',
'区间更窄而严重欠覆盖不构成推断优势。当前正式实验只覆盖 n=1000；不能从本组推断样本量趋势或 RF 稳健性。']
(ROOT/'reports'/'G2_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('G2 audit/report complete',audit,flush=True)

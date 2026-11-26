import sys,json,zipfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import ROOT,save,source_hash
from core import *
import pandas as pd

folder=ROOT/'runs'/'g1_scalar_n1000'
config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
assert config['status']=='completed' and config['R']==1000 and config['workers']==12
assert config['source_hash']==source_hash()
paths=sorted((folder/'records').glob('*.json'));assert len(paths)==4000
seen=set();counts={};replayed=0;failed_datasets=0;invalid_candidates=0
for path in paths:
    data=json.loads(path.read_text(encoding='utf-8'));law=data['cell']['law'];rep=data['replication']
    assert (law,rep) not in seen;seen.add((law,rep));counts[law]=counts.get(law,0)+1
    assert len(data['records'])==10
    if 'traceback' in data:
        failed_datasets+=1;continue
    cachepath=folder/'cache'/(path.stem+'.npz')
    with zipfile.ZipFile(cachepath) as z:assert z.testzip() is None
    for pipeline in ('learned','oracle'):
        rows=[a for a in data['records'] if a['pipeline']==pipeline]
        assert {a['method'] for a in rows}==set(METHODS)
        assert len({a['input_hash'] for a in rows})==1
        for row in rows:
            V=np.array(row['covariance'],float);se=np.array(row['se'],float)
            assert np.allclose(se,np.sqrt(np.diag(V)/1000),equal_nan=True)
    for fold in data['training_choices']:
        for coord in fold['coordinates']:invalid_candidates+=sum(not x['converged'] for x in coord['candidates'])
    if rep in (0,499,999):
        cache=np.load(cachepath);f=cache['folds']
        for row in data['records']:
            oracle=row['pipeline']=='oracle';w=cache['w_oracle' if oracle else 'w'];z=cache['z_oracle' if oracle else 'z'];method=row['method']
            threshold=None if method not in ('H','HT') else cache['scales'][f]*(1.345 if method=='H' else cache['multipliers'][f])
            p,V,_,_=estimate(w,z,f,method,threshold)
            assert np.allclose(p,np.array(row['estimate'],float),atol=1e-12,equal_nan=True)
            assert np.allclose(V,np.array(row['covariance'],float),atol=1e-12,equal_nan=True)
            replayed+=1
assert set(counts)==set(LAWS) and all(v==1000 for v in counts.values())
audit=json.loads((folder/'summary'/'audit.json').read_text(encoding='utf-8'))
audit.update(scientific_conclusions='G1 formal scalar n=1000 main block; separate independent main seeds; larger/smaller n and sensitivities not yet run',cells_by_law=counts,replayed_method_records=replayed,failed_datasets=failed_datasets,invalid_training_candidates=invalid_candidates,source_hash=source_hash(),cache_crc_checked=True,shared_inputs_verified=True)
save(folder/'summary'/'audit.json',audit);save(ROOT/'reports'/'G1_final_audit.json',audit)
coords=pd.read_csv(folder/'summary'/'coordinates.csv');pairs=pd.read_csv(folder/'summary'/'paired.csv')
diag=json.loads((folder/'summary'/'nuisance_diagnostics.json').read_text(encoding='utf-8'))
drift=pd.DataFrame([dict(law=x['law'],population_shift=x['LS_population_shift'][0]) for x in diag]).groupby('law',as_index=False).agg(mean_population_shift=('population_shift','mean'))
for pipeline in ('learned','oracle'):
    ls=coords[(coords.method=='LS')&(coords.pipeline==pipeline)][['law','bias','SD','MeanSE','SEratio','coverage']]
    drift=drift.merge(ls.rename(columns={k:k+'_'+pipeline for k in ls.columns if k!='law'}),on='law')
drift.to_csv(folder/'summary'/'LS_drift_comparison.csv',index=False,encoding='utf-8-sig')
lines=['# G1 正式标量实验验收报告','','G1 已完成：n=1000，四种误差分布各 1000 次全新独立重复，共 4000 份数据、40000 条方法/流程记录。12 个单线程 worker。G2 未启动。','',
f'所有输入共享、记录唯一性、SE/协方差一致性和缓存 CRC 已核验；从缓存复算 {replayed} 条方法记录并重现。数据集失败 {failed_datasets}；无效 Huber 调参候选 {invalid_candidates}；方法失败分类：{audit["failure_counts"]}。','',
'## 主要结果','','覆盖率均按全部尝试次数计算，失败区间记为不覆盖；CSV 同时报告 valid-only coverage、binomial CI、Monte Carlo SE 和成功分母。覆盖率约 95% 时，1000 次重复的 MC SE 约 0.0069。','',
'| 误差 | 流程 | 方法 | Bias | SD | RMSE | MeanSE | SE/SD | Coverage | Length |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for _,a in coords.sort_values(['law','pipeline','method']).iterrows():
    lines.append(f'| {a.law} | {a.pipeline} | {a.method} | {a.bias:.4f} | {a.SD:.4f} | {a.RMSE:.4f} | {a.MeanSE:.4f} | {a.SEratio:.3f} | {a.coverage:.3f} | {a.mean_length:.4f} |')
lines+=['','## 配对比较与解释','','paired.csv 给出 W/N 分别对 LS/H/HT 的 RMSE ratio、2000 次配对 bootstrap CI、MSE difference 的 MC SE、区间长度差和 common-success 分母。LS_drift_comparison.csv 对照 nuisance 位移诊断与 Learned/Oracle bias、SE 校准。结果只覆盖 n=1000 的标量主实验；尚不能据此判断样本量趋势、学习器稳健性或多元联合推断。','','## 可复现记录','',
f'源码 SHA-256：`{source_hash()}`。统计核心与 G0 冻结版本相同，新增 G1 入口与启动检查。seed namespace 为 g1_scalar_n1000，与 smoke/pilot 独立。','',
'- runs/g1_scalar_n1000/config.json：运行前配置、软件、机器、线程和时间。',
'- runs/g1_scalar_n1000/records、cache：完整逐次结果、训练选择、残差缓存和协方差；持续保留，可复算。',
'- runs/g1_scalar_n1000/summary：coordinates.csv、vectors.csv、paired.csv、runtime.csv、LS_drift_comparison.csv、失败与诊断。',
'- source_versions/g1_start：本批源码快照。',
'- reports/G1_preflight.json、G1_final_audit.json、G1_run.log：启动检查、验收与日志。']
(ROOT/'reports'/'G1_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('G1 audited and report written',audit,flush=True)

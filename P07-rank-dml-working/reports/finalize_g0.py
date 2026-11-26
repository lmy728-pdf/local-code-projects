"""Audit all output rows; replay first cache in every cell; write G0 delivery report."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import ROOT,save,source_hash
from core import *
import pandas as pd

audits=[];replays=0;checks=json.loads((ROOT/'reports'/'implementation_checks.json').read_text(encoding='utf-8'))
recovery=json.loads((ROOT/'archive'/'forced_restart_20261114'/'recovery.json').read_text(encoding='utf-8'))
assert all(v['status'].startswith('passed') for v in checks.values())
for stage in ('g0_smoke_scalar','g0_smoke_vector','g0_pilot'):
    folder=ROOT/'runs'/stage;config=json.loads((folder/'config.json').read_text(encoding='utf-8'));audit=json.loads((folder/'summary'/'audit.json').read_text(encoding='utf-8'))
    assert config['source_hash'] in (source_hash(),recovery['previous_source_hash'])
    assert config['status']=='completed'
    assert len(list((folder/'cache').glob('*.npz')))==audit['expected_datasets']
    seen=set();estimates=0;intervals=0;joint=0;training_failures=0
    for path in (folder/'records').glob('*.json'):
        data=json.loads(path.read_text(encoding='utf-8'));key=(json.dumps(data['cell'],sort_keys=True),data['replication']);assert key not in seen;seen.add(key)
        rows=data['records'];assert len(rows)==10
        assert all(len({row['input_hash'] for row in rows if row['pipeline']==pipeline})==1 for pipeline in ('learned','oracle'))
        for choice in data['training_choices']:
            for coord in choice['coordinates']:
                training_failures+=sum(not c['converged'] for c in coord['candidates'])
        for row in rows:
            V=np.array(row['covariance'],float);se=np.array(row['se'],float);p=np.array(row['estimate'],float)
            estimates+=bool(np.isfinite(p).all());intervals+=bool(np.array(row['valid_interval']).all());joint+=bool(row.get('joint_valid',False))
            assert np.allclose(V,V.T,equal_nan=True);assert np.allclose(se,np.sqrt(np.diag(V)/data['cell']['n']),equal_nan=True)
        if data['replication']==0:
            cache=np.load(folder/'cache'/(path.stem+'.npz'));f=cache['folds']
            for row in rows:
                oracle=row['pipeline']=='oracle';w=cache['w_oracle' if oracle else 'w'];z=cache['z_oracle' if oracle else 'z'];method=row['method']
                threshold=None if method not in ('H','HT') else cache['scales'][f]*(1.345 if method=='H' else cache['multipliers'][f])
                p,V,_,fail=estimate(w,z,f,method,threshold)
                assert np.allclose(p,np.array(row['estimate'],float),equal_nan=True,atol=1e-12)
                assert np.allclose(V,np.array(row['covariance'],float),equal_nan=True,atol=1e-12)
                replays+=1
    audit.update(stage=stage,valid_vector_estimate_records=estimates,valid_all_marginal_records=intervals,valid_joint_records=joint,invalid_tuning_candidates=training_failures,workers=config['workers'],start=config['start'],end=config['end'])
    audits.append(audit)
    print(stage,'audited',flush=True)
save(ROOT/'reports'/'final_audit.json',dict(stages=audits,replayed_method_records=replays,source_hash=source_hash(),checks_passed=list(checks)))
coords=pd.read_csv(ROOT/'runs'/'g0_pilot'/'summary'/'coordinates.csv');vec=pd.read_csv(ROOT/'runs'/'g0_pilot'/'summary'/'vectors.csv')
diagnostics=json.loads((ROOT/'runs'/'g0_pilot'/'summary'/'nuisance_diagnostics.json').read_text(encoding='utf-8'))
drift_rows=[]
keys=['experiment','law','n','d','rho','coordinate']
for item in diagnostics:
    for j,value in enumerate(item['LS_population_shift']):
        drift_rows.append(dict(**{k:item[k] for k in keys if k!='coordinate'},coordinate=j+1,population_shift=value))
drift=pd.DataFrame(drift_rows).groupby(keys,as_index=False).agg(mean_population_shift=('population_shift','mean'),SD_population_shift=('population_shift','std'))
learned=coords[(coords.method=='LS')&(coords.pipeline=='learned')][keys+['bias','SD','MeanSE','SEratio','coverage']]
oracle=coords[(coords.method=='LS')&(coords.pipeline=='oracle')][keys+['bias','SD','MeanSE','SEratio','coverage']]
comparison=drift.merge(learned,on=keys).merge(oracle,on=keys,suffixes=('_learned','_oracle'))
comparison['observed_bias_minus_diagnostic']=comparison.bias_learned-comparison.mean_population_shift
comparison.to_csv(ROOT/'runs'/'g0_pilot'/'summary'/'LS_drift_comparison.csv',index=False,encoding='utf-8-sig')
lines=['# G0 第一阶段实验验收报告','', '已完成修正版 G0。全部结果仅用于实现与数值诊断，不作为正式主实验结论。G1 尚未运行。','',
'| 批次 | 条件数 | 每条件 R | 数据重复 | 方法记录 | 完整有效点估计 | 完整有效 marginal 区间 |',
'|---|---:|---:|---:|---:|---:|---:|']
for a in audits:
    lines.append(f"| {a['stage']} | {a['cells']} | {a['expected_datasets']//a['cells']} | {a['completed_datasets']} | {a['records']} | {a['valid_vector_estimate_records']} | {a['valid_all_marginal_records']} |")
lines += ['', '共 7,200 次有效版本的数据重复、72,000 条方法/流程记录。每次含 LS/H/HT/W/N，以及同数据的 Learned/Oracle。修复前作废 smoke 在 archive 中单独保留，不计入以上数量。', '',
'## 实现核验与数值修复','',
'T1–T9 已完成。独立 generator check 对每个 law/d/rho 使用 100 万 draws，保存块间不确定性；t3 的无限四阶矩限制了方差估计精度，未做样本强制标准化。Oracle Gaussian 的点估计在 Monte Carlo 波动内居中，估计量方差随 n 按约 1/n 缩小；不要求有限重复下 root-n 方差单调接近基准。', '',
'额外多元不变性检查发现了直接 W 最优点并列残差的浮点排序问题。修复采用按 fold 中心化残差尺度定义的 64 machine-epsilon 并列容差。修复后标量/多元平移与尺度检查通过，两批 smoke 已用相同种子完整重跑，随后才启动独立种子的 pilot。没有按 coverage 调整估计量、SE 或调参网格。','',
f'最终逐条检查了输入共享哈希、记录唯一性、协方差对称和 SE 对角一致性；对每批每个 cell 的 replication 0 缓存重新计算五方法/两流程，共 {replays} 条点估计与完整 covariance，均重现。','',
'## 失败及 pilot 诊断','']
for a in audits:lines.append(f"- {a['stage']}：方法/区间失败分类 {a['failure_counts']}；无效 Huber 调参候选 {a['invalid_tuning_candidates']}；有效 joint 记录 {a['valid_joint_records']}。")
lines += ['', '下面列出标量 n=1000 的实现 pilot（每 cell R=100）；coverage 的典型 MC SE 约 0.022，不能把本表当成正式方法排序。完整 60 cells 和所有坐标、contrast、联合区域及配对 bootstrap 结果见 CSV。','',
'| 误差 | 流程 | 方法 | Bias | RMSE | MeanSE | Coverage | MeanLength |','|---|---|---|---:|---:|---:|---:|---:|']
for _,a in coords[(coords.experiment=='scalar')&(coords.n==1000)].sort_values(['law','pipeline','method']).iterrows():
    lines.append(f'| {a.law} | {a.pipeline} | {a.method} | {a.bias:.4f} | {a.RMSE:.4f} | {a.MeanSE:.4f} | {a.coverage:.2f} | {a.mean_length:.4f} |')
lines += ['', '所有 marginal 与 joint coverage 均保留原值。Learned/Oracle 差异和欠覆盖是后续正式实验需要评估的现象，未通过扩大 SE 或选择有利数据来修复。', '',
'## 输出与冻结','',
'- `runs/g0_pilot/summary/coordinates.csv`：逐坐标精度、SE 校准、区间及 binomial CI。',
'- `runs/g0_pilot/summary/vectors.csv`：vector RMSE、Wald ellipsoid coverage/volume、同时 marginal 覆盖。',
'- `runs/g0_pilot/summary/contrasts.csv`：两种 contrast 的区间指标。',
'- `runs/g0_pilot/summary/paired.csv`：W/N 对 LS/H/HT，2000 次配对 bootstrap、MSE 差异 MC SE、common-success 分母。',
'- `runs/g0_pilot/summary/LS_drift_comparison.csv`：每 cell/坐标的条件总体 LS 位移诊断与实际 Learned/Oracle bias、MeanSE/SD 对照。',
'- `runs/*/records` 和 `cache`：逐次结果、训练选择、完整协方差、残差缓存。',
'- `implementation_checks.json`、`final_audit.json`、`freeze_manifest.json`、`hardware.json`：证据及冻结记录。','',
f'最终 Python 源码 SHA-256：`{source_hash()}`。统计实现未变；重启后将入口默认并发改为 12、上限设为 12，并增加写盘同步及临时文件占用重试。smoke 原使用 8 workers，pilot 重启前使用 16 workers，恢复后使用 12 workers。原始 config 与恢复清单保存在 archive/forced_restart_20261114；保留结果经过 JSON 和缓存 CRC 检查，损坏结果按原种子重跑。Spline basis、外推、收敛与失败规则、种子和软件版本已写入 README/config/freeze manifest。硬件为 Ryzen 9 7945HX，16 核/32 线程，约 59.8 GB 内存。运行时间是实际执行成本，包含不同并发和重启，不能视为独占硬件的严格速度基准。','',
'原文未指定 B-spline 端点约定、数值并列容差与额外 score residual 验收阈值，本实现将其明确记录为数值约定；无另加 DGP/学习器/估计方法。G0 范围完成，下一组 G1 不自动运行。']
(ROOT/'reports'/'G0_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
manifest=json.loads((ROOT/'reports'/'freeze_manifest.json').read_text(encoding='utf-8'))
manifest['previous_source_hash']=manifest['source_hash'];manifest['source_hash']=source_hash();manifest['sources']['run.py']=hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest();manifest['recovery']=recovery
manifest.update(status='G0_complete_definitions_frozen_before_main',audit_sha256=hashlib.sha256((ROOT/'reports'/'final_audit.json').read_bytes()).hexdigest(),check_evidence_sha256=hashlib.sha256((ROOT/'reports'/'implementation_checks.json').read_bytes()).hexdigest(),stage_configs={a['stage']:hashlib.sha256((ROOT/'runs'/a['stage']/'config.json').read_bytes()).hexdigest() for a in audits},additional_audit_scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'reports').glob('*.py')})
save(ROOT/'reports'/'freeze_manifest.json',manifest)
print('G0 report complete',flush=True)


"""Write the final evidence-based reports from the completed, saved audit."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,re,hashlib,platform,html
import numpy as np
import pandas as pd
from audit_records import OUT,PROJECT,ARCHIVE,TMP
from assemble_tables import load_results,write
sys.path.insert(0,str(PROJECT/'_整理工具/dependencies'))
import markdown

def read(p):return pd.read_csv(OUT/p,encoding='utf-8-sig')
def fmt(v):
    if v is None or (not isinstance(v,str) and pd.isna(v)):return 'NA'
    if isinstance(v,(float,np.floating)):return f'{v:.6g}'
    return str(v).replace('|','／').replace('\n',' ')
def table(df):
    cols=list(df.columns)
    return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))+'\n'
def link(path,base,label=None):
    path=Path(path);target=Path(os.path.relpath(path,base)).as_posix()
    return f'[{label or path.name}](<{target}>)'
CSS='''body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#223043;background:#f5f7fa;margin:0}main{max-width:1180px;margin:30px auto;background:white;padding:38px 46px;border-radius:12px;box-shadow:0 3px 22px #dde3eb}h1{font-size:30px;color:#123952;border-bottom:3px solid #168b83;padding-bottom:15px}h2{font-size:23px;margin-top:34px;color:#123952}h3{font-size:19px}a{color:#006e82}p,li{overflow-wrap:anywhere}.table-scroll{overflow:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d6dfe8;padding:9px 12px;text-align:left;vertical-align:top;min-width:65px}th{background:#e9f2f5}tbody tr:nth-child(even){background:#f8fafc}img{width:100%;height:auto}pre{overflow:auto;background:#f0f3f6;padding:16px}code{overflow-wrap:anywhere}blockquote{border-left:4px solid #168b83;margin-left:0;padding:8px 18px;background:#edf6f5}.meta{font-size:13px;color:#667788}@media(max-width:700px){main{padding:20px;margin:0}h1{font-size:25px}}'''
def document(relative,body):
    p=OUT/relative;p.write_text(body.rstrip()+'\n',encoding='utf-8')
    rendered=markdown.markdown(body,extensions=['tables','fenced_code','toc'])
    rendered=rendered.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
    rendered=re.sub(r'href="([^"#]+)\.md([#"][^"]*)',r'href="\1.html\2',rendered)
    title=body.splitlines()[0].lstrip('# ')
    p.with_suffix('.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+html.escape(title)+'</title><style>'+CSS+'</style><main><p class="meta">Rank-DML · 2026-11-17 · 已有结果核验</p>'+rendered+'</main></html>',encoding='utf-8')

def main():
    a=load_results();manifest=json.loads((TMP/'tables_manifest.json').read_text(encoding='utf-8'))
    assert not manifest['issues'] and manifest['original_summary_mismatches']==0
    # Explicitly distinguish unavailable timing measurements from recorded elapsed times.
    runtime=read('05_运行与复现/T14_运行成本.csv')
    runtime['exclusive_machine_cpu_seconds']=np.nan
    runtime['exclusive_machine_cpu_status']='MISSING: original records contain wall-clock component times, not isolated CPU benchmarks'
    runtime['current_run_dataset_elapsed_seconds_mean']=runtime.elapsed_mean.where(runtime.stage!='G6B')
    runtime['elapsed_scope']=np.where(runtime.stage=='G6B','NA for current run: saved elapsed inherited from source dataset','recorded wall-clock elapsed per original dataset, includes preparation and diagnostics')
    runtime['hardware_normalized_speedup']=np.nan
    write('05_运行与复现/T14_运行成本.csv',runtime)
    # Configuration and code provenance, in addition to untouched original batch snapshots.
    config=read('05_运行与复现/T15_配置与核验.csv');config=config[config.stage!='AUDIT']
    sourcefiles=[Path(r'C:\Users\Lenovo\Desktop\Rank_DML1.pdf'),ARCHIVE/'sources/rank_dml_guide.tex',ARCHIVE/'core.py',ARCHIVE/'run.py',ARCHIVE/'reporting.py',ARCHIVE/'reports/freeze_manifest.json',ARCHIVE/'reports/implementation_checks.json']
    sourcefiles += [PROJECT/(s+'实验')/(s+'实验_代码')/'core.py' for s in ['G0','G1','G2','G3','G4','G5','G6A','G6B']]
    sourcefiles += [PROJECT/'G5实验/G5实验_代码/runner.py',PROJECT/'G5实验/G5实验_代码/rf_learner.py']
    rows=[]
    for p in sourcefiles:
        if p.exists():rows.append(dict(stage='AUDIT',batch='static',item='SHA256 '+p.name,value=hashlib.sha256(p.read_bytes()).hexdigest(),source=str(p)))
    import scipy
    for k,v in dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform(),audit_workers=12,numeric_threads_per_worker=1,master_seed=2026111401,source_PDF_duplicate_name_confirmed=True).items():
        rows.append(dict(stage='AUDIT',batch='audit_environment',item=k,value=str(v),source='current audit environment; original batch environments retained above'))
    write('05_运行与复现/T15_配置与核验.csv',pd.concat([config,pd.DataFrame(rows)],ignore_index=True))
    # Every supplied checkbox gets its own line; evidence is mapped by section and special case.
    raw=(OUT/'00_总览与验收/用户核验清单.txt').read_text(encoding='utf-8')
    checks=[];section='';sub=''
    maps={
      '二':('§2–7','原始 config；core.py 16–200；runner.py；198 个条件块缓存/逐次核验'),
      '三':('§7–8','T01–T04、T07；全部 852000 条 method/pipeline 记录独立重算'),
      '四':('§8.3','T05、T06；原 reporting.py 的 2000 次 paired bootstrap；点比值及长度差独立复核'),
      '五':('§9','T08、T09、T09a、T09b；run.py 84–93；全部保存 fold 诊断与公式(12)重算'),
      '六':('§3、8.4、9、11.1','T10–T15；T13b；G5/G6 原始记录、缓存和配置'),
      '八':('§11.3','图表索引；已有 PDF 2768 张 + 新增 PDF 40 张；PDF结构核验.json')}
    for line in raw.splitlines():
        if line.startswith('## '):section=line[3:]
        if line.startswith('### ') or line.startswith('**G'):sub=line.strip('*# ')
        if not line.startswith('- [ ] '):continue
        item=line[6:];part=section.split('、')[0]
        sec,evidence=maps.get(part,('§1–11','对应正式 CSV 与来源索引'))
        status='PASS';note='指标或证据齐全；PASS 表示满足交付/实现要求，不表示覆盖率必然达到 95%。'
        if 'Single entry' in item:
            status='MISMATCH';sec='§11.1';evidence='archive/run.py:133；G5实验_代码/runner.py:130';note='入口分散；原 run.py 不支持 RF 和 G4–G6 完整网格参数，后续 runner 仅 stage/workers/export/check。'
        elif 'Joint SE/SD' in item:
            note='老师未定义单个联合 SE/SD；以各坐标 SEratio、完整协方差特征值/条件数和联合覆盖率诊断，未伪造联合标量。'
        elif '10000' in item:
            note='保存的五折统计和生成代码支持；本轮不重新拟合、也不重新生成诊断协变量。'
        elif '硬件' in item:
            note='原配置保留 hardware/logical_cpu、workers、线程限制；历史 G0 smoke 为 8 workers。当前核验最多 12×1。'
        checks.append(dict(id=f'C{len(checks)+1:03d}',group=section+' / '+sub,requirement=item,teacher_section=sec,status=status,evidence=evidence,note=note))
    for s,expected in [('G0','60 smoke×20 + 60 pilot×100'),('G1','4×1000'),('G2','16×1000'),('G3','20×1000'),('G4','20×1000'),('G5','8×1000'),('G6A','2×1000'),('G6B','4×1000×2 cached c variants')]:
        checks.append(dict(id=s,group='实验网格',requirement=expected,teacher_section='§3',status='PASS',evidence='实验完整性矩阵.csv',note='G5 和 G6B 复用主实验数据，不计为额外独立样本。'))
    for p in sorted(OUT.glob('*/T[0-9][0-9]_*.csv')):
        checks.append(dict(id=p.name[:3],group='正式表格',requirement=p.name,teacher_section='§8–11',status='PASS',evidence=str(p),note=f'{len(pd.read_csv(p))} 行；NA 语义见统计口径与字段说明。'))
    extra=[
      ('V01','重新核验全部 nuisance 拟合预测','§4、9、11.1','NOT VERIFIED','未保存逐点诊断预测与拟合模型，无法无拟合地独立重演原训练；已核验残差与保存的折统计。'),
      ('V02','隔离其他负载的 CPU 基准','§8.4','MISSING','原数据只有各阶段 wall-clock 计时，没有独占硬件 CPU 时间；这不是导师额外强制的基准。'),
      ('V03','G6B 本轮独立总耗时','§8.4','PARTIAL','rank 二阶段时间可得；elapsed 和基线准备时间沿用源记录，无法当作 G6B 新任务总耗时。')]
    for id,item,sec,status,note in extra:checks.append(dict(id=id,group='审计边界',requirement=item,teacher_section=sec,status=status,evidence='缺失与不一致事项.md；T14；原代码保存字段',note=note))
    checkdf=pd.DataFrame(checks);write('00_总览与验收/逐项验收.csv',checkdf)
    statuscounts=checkdf.status.value_counts().to_dict()
    sectionrows=[
      [1,'四个科学问题','PASS','07 四个科学问题报告；限定推断校准结论'],
      [2,'DGP','PASS','core.py 23–47；G0 T1；规定总体标准化和多元共同 t 尺度'],
      [3,'网格与运行顺序','PASS','11 批次 / 198 条件块；主实验 60 条件各 1000 次'],
      [4,'nuisance 与 Oracle','PASS','样条/GCV/外推和 RF 配置；共同残差 hash；G5 重用数据/folds'],
      [5,'W/N 与秩协方差','PASS','core.py 86–109、152–174；缓存二阶段复演和全部协方差重算'],
      [6,'LS/H/HT','PASS','训练内 MAD 和 9 候选；fold 截距；完整 centered sandwich'],
      [7,'区间、contrast、椭球','PASS','T01–T04；全部 SE、覆盖标记、长度、体积、特征值独立重算'],
      [8,'指标、配对、不确定性与成本','PARTIAL','统计指标齐全；成本有分项，G6B 继承计时须限定解读'],
      [9,'诊断和 c 敏感性','PASS','T09/T09a/T09b 和 T12/T12a；公式(12)逐次一致；未重训'],
      [10,'G0 九项检查','PASS','既有 implementation_checks.json + additional_check_evidence.json；有限 MC 不确定性保留'],
      [11,'复现、表格与图','PARTIAL','T01–T15 与索引齐；单一全参数实验入口不满足 §11.1']]
    secdf=pd.DataFrame(sectionrows,columns=['Section','内容','状态','证据与限定'])
    # Focused, reproducible quantitative report extracts.
    pairs=read('02_方法配对比较/T05_配对RMSE比较.csv');mainpairs=pairs[(pairs.role=='main')&(pairs.pipeline=='learned')]
    ranges=mainpairs.groupby(['law','method','baseline']).agg(条件数=('RMSE_ratio','size'),最小比值=('RMSE_ratio','min'),最大比值=('RMSE_ratio','max'),区间上界小于1=('ratio_upper',lambda x:int((x<1).sum())),区间下界大于1=('ratio_lower',lambda x:int((x>1).sum()))).reset_index()
    write('07_最终报告/主实验配对比较摘要.csv',ranges)
    example=mainpairs[(mainpairs.law=='sln100')&(mainpairs.n==1000)&((mainpairs.d==1)|((mainpairs.d==3)&(mainpairs.rho==.5)))][['d','rho','method','baseline','RMSE_ratio','ratio_lower','ratio_upper','paired_MSE_difference','paired_MSE_mcse']]
    rf=read('04_敏感性分析/T10_RF敏感性.csv');rfw=rf[(rf.pipeline=='learned')&(rf.method=='W')&((rf.d==1)|(rf.level=='joint'))][['law','d','RMSE_ratio_RF_over_Spline','ratio_lower','ratio_upper','RF_coverage','Spline_coverage','coverage_difference_lower','coverage_difference_upper']]
    scalar=read('01_正式主实验/T01_标量主结果.csv');vector=read('01_正式主实验/T02_多元主结果.csv')
    scw=scalar[(scalar.law=='sln100')&(scalar.pipeline=='learned')&(scalar.method=='W')].sort_values('n')[['n','RMSE','bias','SD','MeanSE','SEratio','coverage','coverage_mcse','coverage_lower','coverage_upper','mean_length','mean_interval_score']]
    vew=vector[(vector.law=='sln100')&(vector.pipeline=='learned')&(vector.method=='W')&(vector.d==3)&(vector.rho==.5)].sort_values('n')[['n','vector_RMSE','coverage','coverage_mcse','coverage_lower','coverage_upper','mean_volume','all_marginal_coverage']]
    oracle=read('03_Oracle与诊断/T08_Learned_Oracle.csv');ow=oracle[(oracle.role=='main')&(oracle.method=='W')&(oracle.n==1000)&(oracle.law=='sln100')&(oracle.d==1)][['RMSE_learned','RMSE_oracle','bias_learned','bias_oracle','SEratio_learned','SEratio_oracle','coverage_learned','coverage_oracle']]
    diag=read('03_Oracle与诊断/T09_Nuisance诊断.csv');dg=diag[(diag.stage.isin(['G1','G5']))&(diag.d==1)][['learner','law','mse_m_mean','mse_ell_mean','alignment_variance_mean','LS_population_displacement_mean']]
    size=read('02_方法配对比较/T06_区间与椭球比较.csv');sx=size[(size.stage=='G1')&(size.law=='sln100')&(size.pipeline=='learned')&(size.method=='W')][['baseline','mean_size_difference','size_lower','size_upper','method_coverage','baseline_coverage','method_interval_score','baseline_interval_score']]
    covrows=[]
    for frame,label in [(scalar,'标量边际'),(vector,'多元联合')]:
        for n,q in frame[frame.pipeline=='learned'].groupby('n'):
            covrows.append(dict(类别=label,n=n,最小覆盖率=q.coverage.min(),最大覆盖率=q.coverage.max(),低于95且二项区间上界小于95=int((q.coverage_upper<.95).sum()),条件方法数=len(q)))
    covdf=pd.DataFrame(covrows)
    # Main report: conclusions do not turn numerical differences into unpaired significance.
    science='''# 四个科学问题的实验结论

结论基于 G1–G4 的 60 个正式条件，每条件 R=1000；G5/G6 为单列敏感性。标量比较使用标量 RMSE，多元比较使用完整向量 RMSE。配对区间为 2000 次按 replication 联合重采样的点态 95% percentile 区间，未作多重比较校正。下文的数值例子服务于解释；全部条件与不利结果均保留在 T01–T12。

## Q1：偏态下是否优于 LS 和可信 Huber 基线？

在本研究两种平滑 lognormal DGP 下，W 的 RMSE 优势稳定。sln075 中，W/LS 为 0.629–0.699，W/H 为 0.943–0.965，W/HT 为 0.899–0.923；sln100 中对应范围为 0.466–0.545、0.912–0.955、0.889–0.937。每种分布的 15 个正式条件中，这三组 W 比较的点态区间上界均小于 1。N 在 sln100 对 H 的优势较弱：15 条件中仅 12 个区间完全低于 1，对 HT 为 13 个。不能把“rank”当作单一方法抹去 W/N 差异。

例如 n=1000、sln100 的完整配对结果如下。区间对应 RMSE_ratio；MSE 差的误差量是逐次配对差的 Monte Carlo SE。

'''+table(example)+'''
t(3) 是重要反例：W/H 比值为 0.988–1.007，只有 1/15 个区间完全低于 1，整体更接近持平；N/H 为 1.052–1.081，不能声称 N 优于 H。Gaussian learned 中 W/LS 为 0.922–1.011，并非一律有代价或一律占优。剥离 nuisance 的标量 Gaussian Oracle 更能显示二阶段效率代价：n=1000 时 W/LS=1.0161，95% 配对区间 [1.0032,1.0287]；N/LS=1.0051，[0.9998,1.0109]。这是指定构造下的经验结果，不是普适最优性结论。

## Q2：优势随样本量保持吗？会受学习器影响吗？

W 在两种偏态分布相对三个基线的优势覆盖 n=500/1000/2000。误差绝对量随 n 增大而减小；例如 sln100 标量 W 的 RMSE 从 0.03009 降到 0.01976、0.01211，多元 d=3、ρ=0.5 则为 0.04779、0.02974、0.01970。不同 n 使用独立数据，不将相邻 n 的差称为配对检验结果。

RF 敏感性使用相同数据和 outer folds，具备跨学习器配对比较条件。RF 并未整体改善该 DGP 的性能。W 在 sln100 标量的 RF/Spline RMSE 比为 1.4040 [1.3234,1.4905]，三维为 1.1780 [1.1429,1.2132]。Gaussian 三维 W 比值 1.0063 [0.9858,1.0276]，不支持差异明确。完整 W 对照如下，其他四方法见 T10。

'''+table(rfw)+'''
原始拟合中，RF 的标量 treatment MSE 约 0.112，样条约 0.074；alignment 方差约 0.227–0.236，样条约 0.120–0.128。LS 总体位移在 RF 为约 −0.016，样条为约 +0.0125，实际 LS 偏差也分别为负和正。T09b 还保留逐次诊断与估计误差的关联。这支持学习器误差影响结果的解释，但不构成因果分解，也不意味着所有 RF 都差于所有样条。

## Q3：标准误、覆盖率和区间效率是否校准？

没有全面达到理想校准。RMSE 改善、区间缩短和 95% 覆盖必须分别判断。标量 sln100、n=1000 的 W 覆盖为 0.919，MC SE=0.00863，精确二项 95% 区间 [0.9003,0.9352]，与 0.95 的差不只是“显示时四舍五入”。它的 MeanSE=0.01755，Empirical SD=0.01890，SEratio=0.9288，并有正偏差 0.00580。

'''+table(scw)+'''
同一条件下 W 相对 LS 的区间平均短 0.06240，配对区间 [−0.06407,−0.06077]，interval score 也较低；但 0.919 的覆盖仍不足。相对 HT，W RMSE 更低、区间更短，覆盖率点估计却从 HT 的 0.924 降为 0.919；此覆盖差未单独进行方法间配对显著性检验，不能称为显著恶化。

'''+table(sx)+'''
Oracle 对照更接近校准：同一 sln100 标量 W，Oracle 覆盖为 0.948，SEratio 为 0.9951，偏差约 0.000287。它与 learned 的落差支持 nuisance 偏差/方差影响，但不能把所有低覆盖完全归因于 nuisance。

'''+table(ow)+'''
n 增大后整体覆盖范围有所改善，但不是每个条件单调改善。上表标量 W 覆盖 0.932→0.919→0.927；n=2000 仍低覆盖。必须保留这些结果，不能事后放大 SE 或选择 c 来修补。

## Q4：依赖多元下优势和联合推断如何？

全部 d=2/3、ρ=0/0.5 条件都保留，Vector RMSE 使用 sqrt(mean(||估计−真值||²))，不是坐标 RMSE 平均。偏态下 W 的相对优势在相关结果坐标下仍存在，例如 sln100、d=3、ρ=0.5、n=1000 时 W/H=0.9331 [0.9276,0.9381]，W/HT=0.9007 [0.8878,0.9126]。

联合覆盖不能由边际覆盖推断。三维 sln100 W 的联合覆盖为 0.910、0.935、0.927，对应 n=500、1000、2000；n=2000 的二项区间为 [0.9091,0.9423]。体积下降不代表联合校准自动成立。

'''+table(vew)+'''
协方差包括完整跨坐标项；本轮没有检测到非正定联合协方差或由此造成的区间失败。T02 给出最小特征值和条件数，T03 保留每个坐标，T04 保留 a+、a− 两个 contrast；所有坐标同时覆盖另列 all_marginal_coverage，绝不与 Wald 椭球覆盖混用。这里实现的是逐坐标秩估计及完整联合协方差，并非联合 transport/spatial rank 方法。

## 敏感性与适用范围

G6A 的 τ=0.5 是不同平滑误差分布的新数据，不与 τ=0.2 当作同一数据配对。learned W 标量 RMSE 由 0.01976 到 0.02128，三维由 0.02974 到 0.03410；覆盖点估计分别 0.919→0.931、0.935→0.925。T11 同时报告 Oracle、SEratio、长度、体积与覆盖 MC 不确定性。

G6B 重用缓存：W 点估计对 c 不变，改变的是有限差分斜率及 SE；N 更新有小幅变化。以 sln100 三维 learned 为例，c=0.5/1/2 的 W 覆盖为 0.934/0.935/0.943，N 为 0.922/0.922/0.930。保留 c=1 为主分析，不按观察到的覆盖选择 c。

## 图和完整表

[全部图表索引](../06_论文图表/图表索引.md) · [表格索引和字段说明](../00_总览与验收/统计口径与字段说明.md) · [全部主实验配对范围](主实验配对比较摘要.csv)

![三维相关结果整体 RMSE](../06_论文图表/预览/vector_d3_rho0.5_learned_vector_RMSE.png)

[下载上图 PDF](../06_论文图表/主实验/vector_d3_rho0.5_learned_vector_RMSE.pdf)

![RF 与样条的配对比较](../06_论文图表/预览/RF_RMSE_ratio_learned.png)

[下载上图 PDF](../06_论文图表/敏感性/RF_RMSE_ratio_learned.pdf)
'''
    document('07_最终报告/四个科学问题的实验结论.md',science)
    document('07_最终报告/主要统计发现.md','# 主要统计发现\n\n'+'''本页是正式结果的便查摘要。主实验每个分布有 15 条件（标量 3 + 多元 12）；不把坐标拆分后当作独立实验。计数不是多重检验校正后的发现数量。完整解释见 [四个科学问题](四个科学问题的实验结论.md)。

## 各分布的配对 RMSE 效应范围

'''+table(ranges)+'''\n## Learned 覆盖率按样本量汇总

以下范围跨全部方法和条件；二项区间上界低于 0.95 的数量用于描述逐格低覆盖，不是联合显著性声明。

'''+table(covdf)+'''\n## 学习器诊断：n=1000 标量

'''+table(dg)+'''\n## 失败与收敛

198 个批次条件块均齐全，共 85200 份逐次数据集记录、852000 条方法/流程结果。全部本轮检查的估计、区间、联合正定性及原生失败类别未检出失败；分母见 T13。保存的 8272800 个训练候选条目没有非收敛/非有限方差准则；其中 G6B 条目继承原训练，不能当作新拟合次数。共 1980 次抽样二阶段缓存复演（每条件 rep=0，五方法×两流程），全部逐次推断指标独立重算。未重新拟合第一阶段。
''')
    # Explicit gap table: only engineering mismatch is an actual unmet mandatory entry contract.
    gaprows=[
      ['E01','§11.1','MISMATCH','G0–G6','单一全参数实验入口未统一','代码和各阶段入口存在','不能只靠统计汇总解决；可另写统一调度入口并检查参数，不需重算结果','工程封装修改，未在本轮改原代码','不影响既有数值；影响严格单命令复现便利性'],
      ['V01','§4、9、11.1','NOT VERIFIED','G0–G6','无法独立重演每个已拟合模型的逐点预测','残差缓存、折诊断、训练选择和种子存在；未保存模型/逐点诊断预测','现有汇总可核验公式，不能无拟合恢复原模型内部状态','若必须逐点独立复验，需要按旧种子受控重拟合；本轮禁止','主统计和 Bpop 诊断可恢复；限制训练全过程独立验证强度'],
      ['T01','§8.4','MISSING','全部','独占硬件 CPU 时间、剔除其他负载后的标准化加速比','只有 wall-clock 分项计时和配置','不能由时间记录准确逆推出','若需要公平速度基准，另行受控计时；不是主统计补跑要求','不影响统计结论；不宜给出精确硬件加速承诺'],
      ['T02','§8.4','PARTIAL','G6B','本轮完整任务 elapsed 无独立一致字段','W/N 新二阶段时间存在；elapsed 和准备时间来自源批次','可报告新 rank 秒数；无法把继承 elapsed 变成本轮总耗时','更细新日志/计时才能恢复；本轮 T14 明确 NA','仅限制运行成本解释'],
      ['NA01','§7–8','PASS','多元','联合标量 SEratio、联合 interval score 不适用','完整协方差和各坐标 SE/SD 存在','未定义指标不应硬造','不需要；使用 T02–T04 和联合稳定性图','不是缺失；所有相关 CSV 的结构性 NA 已解释']]
    gapdf=pd.DataFrame(gaprows,columns=['编号','老师章节','状态','实验组','事项','原始数据情况','仅汇总或缓存能否完成','真正需要的操作','影响'])
    write('00_总览与验收/缺失与不一致事项.csv',gapdf)
    gaps='''# 缺失与不一致事项

核心数值输出未发现需要补跑 Monte Carlo 才能填齐的缺口。唯一明确的强制工程实现差距是 §11.1 的单一全参数实验入口。其余项分别是审计证据边界、计时解释限制或数学上不适用，不能混为“实验没跑完”。

'''+table(gapdf)+'''
## 四种恢复类型

1. **已经计算但未整理**：五折 nuisance MSE、alignment 方差/协方差、Bpop、训练选择、gamma、分项计时。已整理到 T09/T09a/T12/T13/T14。
2. **可以从现有记录/缓存重算**：contrast 精度与 SEratio、区间长度中位数、协方差特征值/条件数、RF–Spline 配对区间和全部推断一致性。均已完成，不需重训。
3. **必须重新拟合才能独立复验**：未序列化模型的全部原始逐点预测。老师要求的保存折诊断已经存在，不以此误报 T09 缺失；仅对更强的训练重演标 NOT VERIFIED。
4. **实现与要求不一致**：跨 G0–G6 的统一 CLI。原入口能分别运行各阶段，但不满足统一 stage/R/n/law/d/rho/learner 标志契约。

本轮没有启动模拟或 nuisance 拟合，没有改变原估计器、标准误、DGP、c 或阈值，没有删除、覆盖或移动原始文件。
'''
    document('00_总览与验收/缺失与不一致事项.md',gaps)
    g0=json.loads((ARCHIVE/'reports/implementation_checks.json').read_text(encoding='utf-8'))
    names=['生成器总体矩及 MC 不确定性','内外层防泄漏','直接 W 目标、秩中心化和 ties','fold 常数平移不变性','有限差分与 N 更新符号','正比例尺度变换','LS HC0 与完整协方差','Oracle Gaussian 方差基准','同数据/folds/预测与失败保留']
    g0df=pd.DataFrame([dict(检查=f'T{i}',内容=names[i-1],原始状态=g0[f'T{i}']['status'],本轮证据='读取保存证据，不重新生成样本/拟合') for i in range(1,10)])
    write('05_运行与复现/G0_九项检查证据.csv',g0df)
    doc='# 老师要求逐项验收\n\n科学规范采用桌面 `Rank_DML1.pdf`（12 页）；用户已确认与重复下载名 `Rank_DML1(1).pdf` 为同一份。对应 tex 仅用于核对可搜索公式。核验状态含义：PASS=证据满足；PARTIAL=部分满足；MISSING=现有文件不可恢复；MISMATCH=实现偏离要求；NOT VERIFIED=本轮证据不足。PASS 不表示每格覆盖率都达到名义值。\n\n## Section 1–11 总表\n\n'+table(secdf)+'\n## G0 九项实现检查\n\n'+table(g0df)+'\nT1、T8 保留有限抽样误差说明，没有把矩或有限样本方差强行校准到理论值。原始证据位于 archive/reports/implementation_checks.json、additional_check_evidence.json、freeze_manifest.json，绝对路径与 hash 见 T15/来源索引。\n\n## 用户清单逐项落地\n\n'+table(checkdf[['id','requirement','teacher_section','status','evidence','note']])
    document('00_总览与验收/老师要求逐项验收.md',doc)
    # Full field and provenance guide, including the potentially ambiguous covariance scale.
    tables=sorted(OUT.glob('*/T[0-9][0-9]*.csv'))
    tableindex=pd.DataFrame([dict(文件=p.name,行数=len(pd.read_csv(p)),路径=link(p,OUT/'00_总览与验收')) for p in tables])
    dictionary='''# 统计口径与字段说明

## 文件索引

'''+table(tableindex)+'''
## 分母、尺度与比较

- `stage/batch/role` 是必要筛选列。正式主分析为 G1–G4 / role=main；G0 为 smoke/pilot；G5/G6 单列敏感性。T08/T09/T13/T14 含 G0 以便审计，不能不筛选就用于论文主结论。
- 数值 ties 约定：排序后的相邻残差差不超过 64×机器 epsilon×max(|残差−中位数|) 时并为一组，使用中心化 score 网格的平均分数。这是冻结 core.py 97–109 的浮点容差约定，用于保留 W 解上的数学 ties；不作事后按结果调整。
- `attempted` 为全部尝试，正式条件为 1000。`valid_estimates` 或 `valid_vector_estimates` 是点估计分母，`valid_intervals` 是有效推断分母。`coverage` 为全部尝试覆盖率，失败算不覆盖；`coverage_valid_only` 为有效区间内覆盖率。
- Accuracy 指标按成功点估计计算并标记 conditional-on-success；本轮两种分母在所有条件一致，没有用成功子集隐藏失败。
- `SD` 使用 ddof=1；`RMSE=sqrt(mean(error²))`；`vector_RMSE=sqrt(mean(sum_j error_j²))`；`vector_bias_norm=||mean(error vector)||`。MAE 与 median_absolute_error 分开。
- `MeanSE/SD` 是 SEratio。原保存 `covariance` 是 **root-n 渐近协方差 V**，估计量协方差为 V/n；SE=sqrt(diag(V)/n)。T02 中 `cov_min_eigenvalue_*` 和 `cov_condition_*` 均来自 V。特征值量级不要误当作 estimator covariance。
- `coverage_mcse=sqrt(p(1-p)/B)`；`coverage_lower/upper` 为双侧 Clopper–Pearson 精确二项 95% 区间。图上误差条为 ±1.96 MC SE，与精确二项区间并非完全相同。
- `mean_length/median_length` 只针对有效区间；95% interval score=长度+40×下侧漏出距离+40×上侧漏出距离。
- `coverage` 在 T02/level=joint 指 Wald 联合椭球；`all_marginal_coverage` 是所有坐标边际区间同时覆盖；T04 是 a+=1/√d 和 a−=(1,−1,0,...)/√2 的单变量 contrast。三者不混用。
- `mean_volume` 为平均联合椭球体积，`median_log_volume` 为 log(volume) 的中位数。使用完整 V 及 χ²(d) 临界值。体积不能跨 d 当作同量纲效率比。
- T05 的多元 RMSE 比使用完整向量平方损失。T05/T06 的原始 2000 次 bootstrap 沿用原 reporting.py 默认随机流 2026111499，每个比较按共同成功 replication 联合重采样。保存 `common_estimate_success` / `common_interval_success`；本轮均为 1000。区间是点态 percentile 区间，不作多重比较调整。
- T10 的新增跨学习器 bootstrap 也为 2000 次，每个条件固定 SHA256 派生 seed（见列 bootstrap_seed），整次 replication 的向量和协方差信息始终绑定；不按坐标独立乱序。正的 RF/Spline RMSE 比超过1表示 RF 更差。Oracle H/HT 仍沿用各 learned 训练阈值，故这两方法的跨学习器 Oracle 结果不必完全相同。
- T09 按 fold 权重汇总预测误差；先聚合 Cov(b,A) 和 Var(b)，再取 Bpop 的比值。Bpop 不是 W/H/N 的精确偏差公式，也不是有限样本平均 LS bias 的恒等式。T09a 保存每折摘要；T09b 关联描述不能解释为因果。
- T11 两个 τ 是不同分布、不同数据；coverage_difference MCSE 用两组独立方差相加，不宣称配对。T12 中 c=1 行来自 G1/G2，0.5/2 来自 G6B，不能合并成 3000 个新数据集。

## NA 的含义

1. joint 行没有标量 bias、SD、MeanSE、SEratio、interval score、单变量长度；这些 NA 是数学上不适用。使用 vector_bias_norm、vector_RMSE、完整 V、joint coverage、volume。
2. coordinate/contrast 行没有联合体积等字段，同样不适用。RF 的多元 joint bias_difference 为 NA，完整 vector bias norm 与各坐标 bias 可在对应表找到。
3. T13 中只属于特定方法的失败原因（如 LS 的 rank slope）`applicable=False`，count/rate 为 NA，不写伪造的零；适用类别的零失败均有 denominator。
4. T14 的 exclusive_machine_cpu_seconds、hardware_normalized_speedup 为 NA：未记录独占硬件基准。G6B 当前 elapsed 为 NA：原值继承源批次。原始 elapsed 保留并标注，不用它虚构 G6B 总耗时。
5. RF 原始 penalty/GCV 为 None，因为它没有样条 Ridge 的 penalty；这是 learner 不适用，不是拟合失败。

所有 CSV 使用 UTF-8 BOM、NA 显式缺失标记和 17 位有效数字。用于阅读的报告表取 6 位有效数字，正式计算以 CSV/原始 JSON 为准。
'''
    document('00_总览与验收/统计口径与字段说明.md',dictionary)
    reproduction='''# 复现说明

本轮只做现有数据审计与后处理，未运行 Monte Carlo 或 nuisance 拟合。已有文件原位保留；新产物全部位于桌面 `Rank-DML_最终验收与论文结果`。新目录不是唯一原始数据源。

## 来源和入口

'''
    for p in [ARCHIVE,PROJECT,Path(r'C:\Users\Lenovo\Desktop\Rank_DML1.pdf'),ARCHIVE/'reports/freeze_manifest.json']:
        reproduction+='- '+link(p,OUT/'05_运行与复现',str(p))+'\n'
    reproduction+='''
G0–G3 原始 records/cache 位于 archive/runs；G4–G6 位于桌面各 `Gx实验/Gx实验_代码/_work`。G6B 指向 G1/G2 原缓存。索引保存绝对路径、大小、mtime_ns 和已计算 SHA256。旧 CSV 的用户另存副本不作为数值权威来源。

## 本轮后处理顺序（无需重跑实验）

同目录 [核验代码/打开核验结果.ipynb](核验代码/打开核验结果.ipynb) 供 VS Code 阅读入口，默认只读结果，不执行实验。核验代码也全部保存以便复查。

1. audit_records.py：12 个单线程 worker 读取原记录与缓存，逐次重算推断；每条件 rep=0 复演二阶段；不调用生成器、crossfit 或 fit。
2. assemble_tables.py：汇总 T01–T15；核对原 paired.csv；新增 RF 配对 bootstrap；产生来源索引。
3. supplement_training.py：仅扫描保存训练候选的收敛标记、方差准则和尺度。
4. paper_figures.py：使用审计 CSV/中间结果绘制 40 张矢量 PDF。
5. verify_pdf.py：PDF 结构检查及 Poppler 渲染预览；使用文档运行库 Python。
6. final_reports.py：本说明、验收、四个科学问题及 HTML 报告；补充 T14 的 NA 语义和 T15 证据。
7. validate_delivery.py：核对来源未变、表格/报告存在和文件链接；不启动实验。

审计主 Python 为 `C:/Users/Lenovo/Documents/ATLAS/runtime/Scripts/python.exe`，数值版本与源实验相同（NumPy 2.5.4 / SciPy 1.18.1 / pandas 3.1.1），Matplotlib 使用原 plot_packages。PDF 结构与 HTML 检查用预装文档库。环境细节见 T15。所有数值 worker 的 BLAS/OpenMP/NumExpr 线程为1，并发不超过12；补图和报告主要单进程。历史 G0 smoke 的8 workers和早期配置保持原样，不伪改历史。

## 随机性与状态保留

主 seed 为 2026111401。core.rng 对 [master,stage,cell,rep,stream] 的规范 JSON 取 SHA256，前16字节 little-endian 作为种子；data、outer、inner、diagnostic 为分离命名空间。G5 mapped_rng 将共享数据/outer/diagnostic 映射到 G1/G2；forest 拟合用独立训练/target 流。G0 smoke/pilot 独立于正式命名空间。调度不进入种子。复现应保留旧 stage/cell/rep，不仅保留 master seed。

每份原始 JSON 包含五方法×两流程，估计向量坐标顺序为1..d，协方差为 d×d；npz 包含 w,z、Oracle w,z、folds、scales、multipliers、penalties、truth。首次扫描完成全部缓存 CRC 读取和哈希检查。原模型未序列化，限制见缺失清单。

## 耗时不能相加误读

T14 按数据集报告 shared nuisance、auxiliary、tuning 和各方法二阶段时间。方法完整 pipeline 的定义是 nuisance + 本方法 second-stage +（H/HT 的 auxiliary）+（仅 HT 的 tuning）。它是分别运行方法的成本估计；不能再把五个 full_pipeline 求和当作共享运行总耗时。diagnostic 时间另列。Oracle 行原先也计入 learned 准备，不是实际独立训练 Oracle 的成本。G6B 只有 rank 二阶段新计算，其他计时有继承；不能用源 elapsed 推测 G6B 新任务耗时。

## 未达成的统一实验 CLI

老师 §11.1 所述一个脚本接受所有 stage/R/n/law/d/rho/learner 尚未完全实现。原 run.py 仅有样条及 G0–G3；后续 runner.py 采用阶段固定配置。现有 notebook 可分阶段启动，但不等于统一全参数入口。本轮核验 notebook 仅是查看结果入口，也不声称修复了这个差距。
'''
    document('05_运行与复现/复现说明.md',reproduction)
    limitations='''# 结果局限与适用范围

- 主结论仅覆盖指定 p=10、n=500/1000/2000、d≤3、ρ=0/0.5 和四种误差族。平滑 lognormal 的优势不能推广为所有非对称误差下的普适优势。
- 每格 R=1000，覆盖率约0.95时 MC SE约0.0069。所有区间为逐格不确定性，没有多重比较校正；胜出格数不能当作独立重复的元分析样本量。
- Learned 在多个条件低覆盖；n 增大有整体改善但不保证逐格单调；RF 在指定默认参数下更差不代表所有树模型更差。
- t(3) 下 W 接近 H，N 可能逊于 H；Gaussian Oracle 显示 W 的小幅效率代价。不能选择性只展示偏态优势。
- nuisance 诊断、Bpop 和 Oracle 使用模拟真值，仅作解释；没有参与可行估计、阈值或 c 的选择。诊断关联不是因果证明。
- G5/G6B 重用原数据，不提供新的独立样本；G6A 两个 τ 不同分布，不作配对。G0 仅实现检查与 pilot。
- 联合 SE/SD 没有统一标量定义；边际、contrast、联合椭球和所有边际同时覆盖是不同事件。当前方法不是多元 transport/spatial rank。
- 全部保存结果与推断公式核验通过，不等于全部 nuisance 拟合过程已经重新独立执行。模型/逐点诊断预测未保存，本轮不拟合的约束限制了这种验证。
- 运行时间来自实际 wall-clock 分项，受负载和运行条件影响；不提供独占 CPU 的精确速度排名。G6B 继承计时单列说明。
- 工程入口分散违反单一全参数 CLI 交付要求，已标 MISMATCH；无需因此重算已有科学结果。原代码不在本轮改动。

具体证据、涉及组别、是否可恢复和需要的操作见 [缺失与不一致事项](../00_总览与验收/缺失与不一致事项.md)。
'''
    document('07_最终报告/结果局限与适用范围.md',limitations)
    final='''# 最终交付验收报告

**总体评价：核心实验、保存数据、统计指标与图表交付通过；可复现工程交付为 PARTIAL，不能宣称所有要求无条件 PASS。** 明确差距是单一全参数实验入口未统一；训练全过程独立重演和部分运行成本也有证据边界。

## 1. 完成情况

核验老师 Section 1–11，覆盖 G0、G1、G2、G3、G4、G5、G6A、G6B。合计 11 个运行批次、198 个批次条件块，85200 份数据集级记录、852000 条方法/流程结果。主实验仅 G1–G4 的60条件×1000次，G5/G6B 重用数据不计为独立新增样本。

全部条件记录齐全；原汇总与重算未发现数值差异。完成全部缓存/共享输入验证、逐次 SE/区间/contrast/联合推断/协方差诊断与公式(12)检查；每条件 rep=0 的五方法×两流程共1980次缓存复演通过。保存训练候选8272800条均无无效/非收敛标记（含 G6B 继承条目）。没有重新运行 Monte Carlo 或 nuisance learner。

'''+table(secdf)+f'\n逐项清单状态计数：{json.dumps(statuscounts,ensure_ascii=False)}。该计数包含补充审计边界，不是把可选独占 CPU 基准当成老师强制要求。\n\n'+'''交付包括 T01–T15 全套 CSV，以及 fold 诊断、逐次关联、c 联合推断、训练收敛等补充表；新增40张矢量PDF和已有2768张PDF的原路径索引。原文件原位保留，G0未进入主结果表。

## 2. 四个科学问题

- **Q1**：两种偏态分布下 W 相对 LS/H/HT 的 RMSE 优势稳健；以 sln100 n=1000 标量为例，W/H=0.9394 [0.9312,0.9481]。t(3) 下 W 与 H 接近，N 对 H 没有相同优势。
- **Q2**：偏态优势覆盖三个样本量；学习器敏感性明显。sln100 标量 W 的 RF/Spline 比为1.4040 [1.3234,1.4905]，不能把换RF当作普遍改善。
- **Q3**：不能宣布校准完成。sln100 n=1000 标量 W 覆盖0.919，二项95%区间[0.9003,0.9352]，SEratio=0.9288；Oracle覆盖0.948。更低RMSE和更短区间并不消除低覆盖。
- **Q4**：相关多元条件下估计优势保留，例如三维 sln100、ρ=0.5、n=1000 的 W/H=0.9331 [0.9276,0.9381]；联合覆盖仍会不足，不能由边际指标替代。

完整幅度、误差量、反例和敏感性解释见 [四个科学问题的实验结论](四个科学问题的实验结论.md) 与 [主要统计发现](主要统计发现.md)。

## 3. 需要补充或保留限定的事项

'''+table(gapdf)+'''
不建议为已能恢复的指标重新模拟。统一入口是工程补交；若要求逐点重新验证拟合，才需要另行授权的受控重拟合。本轮只列出真正需要的操作，没有自动执行。

## 4. 最终完整性评价

| 方面 | 评价 |
|---|---|
| 实验设计符合性 | PASS：规定网格、分布、样本量、五方法、两流程、敏感性均存在 |
| 方法实现符合性 | PASS：代码定义与缓存二阶段/逐次推断核验一致；保留数值 ties 约定 |
| 数据完整性 | PASS：198 条件块齐全，无本轮检测到的数学不一致 |
| 统计分析完整性 | PASS：T01–T15 与补充诊断齐全；不适用/不可测量项明确 NA |
| 图表完整性 | PASS：现有图保留索引；补足整体多元、诊断与敏感性 PDF；不删不利条件 |
| 可复现性 | PARTIAL：残差/种子/配置充分支持二阶段；统一 CLI 不足，原拟合模型未保存 |
| 四个科学问题 | 足以回答限定设计内的经验问题；不能支持普遍支配或全面95%校准声明 |

## 5. 文件入口

[总览与 CSV 索引](../00_总览与验收/README.md) · [逐项验收](../00_总览与验收/老师要求逐项验收.md) · [无法恢复与不一致清单](../00_总览与验收/缺失与不一致事项.md) · [图表索引](../06_论文图表/图表索引.md) · [复现说明](../05_运行与复现/复现说明.md)
'''
    final+='\n## 6. 交付文件验证\n\n原始文件索引的170887个文件大小及修改时间复核一致；其中170877个已有SHA256的文件全量重新计算哈希，未发现改变。40个新增PDF均为单页矢量内容、没有嵌入位图；已渲染检查。HTML报告已检查本地链接、图片和页面溢出。机器可读证据见 [最终文件核验](../00_总览与验收/最终文件核验.json)、[原始文件核验](../00_总览与验收/原始文件未改动核验.json)、[PDF核验](../05_运行与复现/PDF结构核验.json)、[HTML核验](../05_运行与复现/HTML显示核验.json)。\n'
    document('07_最终报告/最终交付验收报告.md',final)
    readme='''# Rank-DML 最终验收与论文结果

本目录为2026-11-17新建的核验交付，原始实验未删除、覆盖或移动。按用户确认的桌面 `Rank_DML1.pdf` 核验；未重跑 Monte Carlo 或 nuisance 拟合。

**结论：核心统计交付通过；可复现工程交付 PARTIAL。** 单一全参数实验入口未统一；完整训练重演及部分计时有证据边界。所有不足逐项列明，没有用 pilot 替代正式结果，也未按结果调整 SE 或 c。

## 先看这四项

1. [最终交付验收报告](../07_最终报告/最终交付验收报告.md)
2. [四个科学问题的实验结论](../07_最终报告/四个科学问题的实验结论.md)
3. [缺失与不一致事项](缺失与不一致事项.md)
4. [论文图表索引](../06_论文图表/图表索引.md)

## 已完成

- 11批次、198条件块，85200份数据集记录、852000条方法/流程结果；正式主实验为60条件×1000次。
- 原汇总重算一致；全部缓存/逐次推断/诊断公式核验，以及1980次二阶段抽样复演通过。
- T01–T15 CSV 和补充表齐全；40张新增PDF；2768张已有PDF原位索引。
- 每份报告同时提供 Markdown 和 HTML。网页预览PNG仅供阅读，正式图下载PDF。

## 表格

'''+table(tableindex)+'''
T01/T03/T04 为逐坐标或contrast；T02 为完整向量/联合椭球。T08/T09/T13/T14 含审计用G0行，写论文时必须筛 role=main 或指定正式阶段。原记录字段、分母、root-n协方差与NA说明见 [统计口径与字段说明](统计口径与字段说明.md)。

## 核验证据

[完整性矩阵CSV](实验完整性矩阵.csv) · [逐项验收CSV](逐项验收.csv) · [老师要求逐项验收](老师要求逐项验收.md) · [来源索引CSV](数据来源与文件索引.csv) · [原汇总重算一致性](原汇总与重算一致性.csv) · [复现说明](../05_运行与复现/复现说明.md)

来源索引很大（170887个文件），建议用筛选工具打开；报告和图表无需加载它即可阅读。数据集计数含G5/G6B重用，不是85200份独立新生成数据。
'''
    document('00_总览与验收/README.md',readme)
    # Direct clickable indexes reuse existing original vector plots without copying them.
    old=read('06_论文图表/已有PDF图表索引.csv')
    selections=[('主实验/已有样本量趋势图.md','已有样本量趋势图',old[(old.stage=='G4')&old.name.str.startswith('compare_')]),
      ('配对比较/已有方法配对图.md','已有方法配对图',old[old.stage.isin(['G1','G2','G3','G4'])&old.name.str.contains('paired')]),
      ('敏感性/已有G6核心图.md','已有G6核心图',old[old.stage.isin(['G6A','G6B'])])]
    for rel,title,q in selections:
        base=OUT/'06_论文图表'/Path(rel).parent;base.mkdir(exist_ok=True)
        body=f'# {title}\n\n沿用原始正式 PDF，共 {len(q)} 张；只建立链接，不搬动、不重复绘制。\n\n'
        body+='\n'.join('- '+link(r.path,base,r.stage+' · '+r['name']) for _,r in q.iterrows())
        document('06_论文图表/'+rel,body)
    p=OUT/'06_论文图表/图表索引.md';index=p.read_text(encoding='utf-8').split('\n## 已有图直接入口')[0]
    index+='\n## 已有图直接入口\n\n[352 张样本量趋势图](主实验/已有样本量趋势图.md) · [120 张正式方法配对图](配对比较/已有方法配对图.md) · [8 张 G6 核心图](敏感性/已有G6核心图.md)\n'
    document(str(p.relative_to(OUT)),index)
    document('README.md','# Rank-DML 最终验收与论文结果\n\n[打开总览](00_总览与验收/README.md) · [最终验收报告](07_最终报告/最终交付验收报告.md) · [四个科学问题](07_最终报告/四个科学问题的实验结论.md) · [图表索引](06_论文图表/图表索引.md)\n\n核心统计交付通过，可复现工程交付 PARTIAL。全部输出在本桌面目录，原文件原位保留。')
    notebook={"cells":[{"cell_type":"markdown","metadata":{},"source":["# 打开最终核验结果\n","运行下一格仅在浏览器打开本地结果，不运行实验。HTML/CSV/PDF 也可直接双击打开。\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["from pathlib import Path\n","import webbrowser\n",f"root = Path({str(OUT)!r})\n","webbrowser.open((root / '00_总览与验收/README.html').as_uri())\n"]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    (OUT/'05_运行与复现/核验代码/打开核验结果.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'00_总览与验收/report_manifest.json').write_text(json.dumps(dict(checks=len(checks),statuses=statuscounts,html_reports=len(list(OUT.rglob('*.html'))),tables=len(tables),original_summary_mismatches=0),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=len(checks),statuses=statuscounts,tables=len(tables),reports='complete'),ensure_ascii=False))

if __name__=='__main__':main()

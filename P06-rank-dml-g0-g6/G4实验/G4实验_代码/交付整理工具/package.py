from pathlib import Path
import json,shutil,hashlib,ast,html,sys
W=Path(__file__).resolve().parents[4];old=W/'rank_dml_study';out=W/'实验交付';tools=Path(__file__).parent
sys.path.insert(0,str(old/'plot_packages'))
import pandas as pd
stages={'G0':['g0_smoke_scalar','g0_smoke_vector','g0_pilot'],'G1':['g1_scalar_n1000'],'G2':['g2_vector_n1000'],'G3':['g3_n500']}
for stage,runs in stages.items():
    target=out/(stage+'实验');code=target/(stage+'实验_代码');tables=target/'汇总表_CSV';result=target/'结果_MD';fig=target/'图'
    for p in (code,tables,result,fig):p.mkdir(parents=True,exist_ok=True)
    snapshot=old/('frozen_source' if stage=='G0' else 'source_versions/'+stage.lower()+'_start')
    for p in snapshot.glob('*.py'):shutil.copy2(p,code/p.name)
    scripts=code/'交付脚本';scripts.mkdir(exist_ok=True)
    for name in ('finalize_'+stage.lower()+'.py','vector_figures.py'):
        p=old/'reports'/name
        if p.exists():shutil.copy2(p,scripts/p.name)
    src=code/'科学规范';src.mkdir(exist_ok=True)
    for name in ('README.md','rank_dml_guide.tex'):shutil.copy2(old/'sources'/name,src/name)
    combined=[]
    for run in runs:
        td=tables/run;td.mkdir(exist_ok=True)
        for p in (old/'runs'/run/'summary').glob('*.csv'):
            shutil.copy2(p,td/p.name)
            if p.stat().st_size>5:combined.append(pd.read_csv(p).assign(table=p.stem,batch=run))
    pd.concat(combined,ignore_index=True,sort=False).to_csv(tables/'汇总表.csv',index=False,encoding='utf-8-sig')
    manifest=json.loads((old/'vector_figures'/stage/'manifest.json').read_text(encoding='utf-8'))
    gallery=['<!doctype html><meta charset="utf-8"><title>'+stage+'</title><style>body{max-width:1100px;margin:30px auto;font:18px sans-serif}img{width:100%}</style><h1>'+stage+' 矢量图</h1>']
    for e in manifest['figures']:
        name=Path(e['file']).name;shutil.copy2(old/'vector_figures'/e['file'],fig/name)
        gallery.append(f'<h2>{html.escape(e["metric"])} · {e["law"]} · {e["pipeline"]} · d={e["d"]} · rho={e["rho"]}</h2><a href="{name}">SVG</a><img loading="lazy" src="{name}">')
    (fig/'图表索引.html').write_text('\n'.join(gallery),encoding='utf-8')
    a=json.loads((old/'reports'/('final_audit.json' if stage=='G0' else stage+'_final_audit.json')).read_text(encoding='utf-8'))
    total=sum(x['completed_datasets'] for x in a['stages']) if stage=='G0' else a['completed_datasets']
    lines=[f'# {stage} 实验结果','',f'已完成 {total:,} 份数据；'+('G0 为 smoke/pilot，不计入正式重复。' if stage=='G0' else '每条件 1,000 次独立重复；失败分类：'+str(a['failure_counts'])+'。'),'', '[汇总表 CSV](../汇总表_CSV/汇总表.csv) · [图表索引](../图/图表索引.html)','', 'table 标识指标类别，batch 标识实验批次；未适用字段留空。分项 CSV 同时保留。coverage 含失败，另报有效区间分母；vectors 的 coverage 为联合椭球覆盖率。','']
    if stage!='G0':
        frame=pd.read_csv(old/'runs'/runs[0]/'summary'/'coordinates.csv');q=frame[(frame.experiment=='scalar')&(frame.pipeline=='learned')]
        if len(q):lines+=['| law | method | RMSE | Coverage |','|---|---|---:|---:|']+[f'| {r.law} | {r.method} | {r.RMSE:.4f} | {r.coverage:.3f} |' for r in q.itertuples()]
        v=pd.read_csv(old/'runs'/runs[0]/'summary'/'vectors.csv');v=v[(v.d>1)&(v.pipeline=='learned')]
        if len(v):lines+=['',f'Learned 多元联合覆盖率范围：{v.coverage.min():.1%}–{v.coverage.max():.1%}。完整条件及 Oracle 对照见 CSV。']
    lines+=['',f'图共 {len(manifest["figures"])} 张，保留原 SVG。原始记录和完整验收材料保留于 `{old}`。']
    (result/(stage+'_结果.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (code/'代码说明.md').write_text(f'# {stage} 代码\n\n根目录为运行时源码快照；交付脚本单列。没有 notebook。历史入口依赖原项目 runs、reports 和冻结清单；完整运行上下文保留于 `{old}`，本目录是源码归档。\n',encoding='utf-8')
    (code/'原项目位置.json').write_text(json.dumps({'archive':str(old),'runs':runs},ensure_ascii=False,indent=2),encoding='utf-8')
for stage in ('G4','G5','G6A','G6B'):
    target=out/(stage+'实验');code=target/(stage+'实验_代码')
    for p in (code,target/'汇总表_CSV',target/'结果_MD',target/'图'):p.mkdir(parents=True,exist_ok=True)
    for name in ('core.py','run.py','reporting.py'):shutil.copy2(old/name,code/name)
    p=code/'reporting.py';p.write_text(p.read_text(encoding='utf-8').replace("scientific_conclusions='G0 implementation only; not main experiment evidence'","scientific_conclusions='Formal experiment/sensitivity; see stage configuration'"),encoding='utf-8')
    for name in ('runner.py','deliver.py','prepare_environment.py'):shutil.copy2(tools/name,code/name)
    if stage.startswith('G6'):shutil.copy2(tools/'compact_output.py',code/'compact_output.py')
    settings={'stage':stage,'archive':str(old),'frozen':{name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in ('core.py','reporting.py')}}
    (code/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    ps=r"""$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' -u '.\runner.py' --workers 12
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect the error above.' }
"""
    (code/('运行'+stage+'.ps1')).write_text(ps,encoding='utf-8-sig')
    (code/('启动'+stage+'.cmd')).write_text('@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0运行'+stage+'.ps1"\r\npause\r\n',encoding='gbk')
    install=ps.replace("-u '.\\runner.py' --workers 12","-m pip install --only-binary=:all: --no-deps --target '.\\dependencies' scikit-learn joblib threadpoolctl")
    (code/'install_G5_dependencies.ps1').write_text(install,encoding='utf-8-sig')
    specs={'G4':'20 条件，n=2000，各 R=1000。','G5':'8 条件，n=1000，各 R=1000；内外层均使用 RF 500 棵树、min_leaf=10、全特征、训练集大小 bootstrap、无限深度。','G6A':'2 条件，n=1000，各 R=1000；平滑 lognormal sigma=1、tau=0.5。','G6B':'4 个既有条件，各复用 1000 份残差，计算 c=0.5/2 与 c=1 对照；不重拟合 nuisance。'}
    (code/'使用说明.md').write_text(f'''# {stage} 使用说明

{specs[stage]}

双击 `启动{stage}.cmd`，或执行 `运行{stage}.ps1`。只运行本组；不要同时启动多组。默认 12 个单线程 worker。

完成后自动生成三个输出：汇总表_CSV、结果_MD、图（独立 PDF 矢量图）。无 notebook。中间记录/缓存保存在代码目录 `_work`，用于续算和验收；重复启动可续算，源码变更或缓存损坏时报错，不静默混用。

仅重新汇总/绘图：使用指定 Python 执行 `runner.py --export-only`。Python 为 `C:\\Users\\Lenovo\\Documents\\ATLAS\\runtime\\Scripts\\python.exe`。绘图库读取原项目 plot_packages。

G5 运行前先执行 `install_G5_dependencies.ps1`，当前环境无 sklearn；安装到本代码目录 dependencies，不修改历史环境。算法种子按 replication、fit 序号（外层/内层）和 target 独立记录；RF 的 penalty 字段不适用，记 null。

settings.json 指定原项目路径。G5/G6B 依赖原 G1/G2 records/cache；G4 对照读取 G1–G3 汇总。移动项目需更新路径，保留原始数据。G5 校验 Oracle 残差、folds 与原缓存完全相同。

G6B 的 LS/H/HT 与 nuisance 诊断、耗时复用历史记录；W/N 二阶段耗时为本次重算，不能当作全部方法重拟合。

已做代码静态检查；未启动新实验或模拟试跑。运行后会自动核验记录、共享输入和推断结果，并缓存复算每条件第 0/499/999 次。
''',encoding='utf-8')
    for p in code.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
(out/'使用说明.md').write_text('''# 实验交付

每组四个子文件夹：Gx实验_代码、汇总表_CSV、结果_MD、图。

G0–G3 已整理，旧图保留 SVG；G4/G5/G6A/G6B 默认输出 PDF。汇总表.csv 合并多类指标，以 table/batch 区分，同时保留分项 CSV。

先双击 `G4实验/G4实验_代码/启动G4.cmd`。各组独立启动；G5 先安装其依赖。不要同时跑多组。

未启动或监测新实验。原始数据和完整历史材料保留在 rank_dml_study；不重复复制大型缓存。
''',encoding='utf-8')
print('Packaging complete')

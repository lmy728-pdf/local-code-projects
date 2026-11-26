"""Build reports and navigable indexes from completed, unchanged summary CSVs."""
from pathlib import Path
import csv, json, sys, os, shutil, hashlib, html, collections, re, subprocess, zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = Path(r'C:\Users\Lenovo\Documents\ChatGPT\毕设\rank_dml_study')
sys.path.insert(0, str(HERE/'dependencies'))
import markdown

STAGES = ['G0','G1','G2','G3','G4','G5','G6A','G6B']
METHODS = ['LS','H','HT','W','N']
LAWS = ['gaussian','t3','sln075','sln100','sln100_tau050']
BATCH = dict(G0='g0_pilot',G1='g1_scalar_n1000',G2='g2_vector_n1000',G3='g3_n500',G4='g4_n2000',G5='g5_rf')
TITLE = dict(G0='实现核验与 pilot',G1='标量主实验',G2='多元主实验',G3='小样本实验',G4='大样本实验',G5='随机森林敏感性',G6A='误差平滑敏感性',G6B='数值步长敏感性')
DESIGN = dict(G0='n=500/1000/2000；12 个标量、48 个多元条件；smoke 每条件 20 次，pilot 每条件 100 次。',
 G1='n=1000；标量；四种分布，每条件 R=1000，共 4000 份数据。',
 G2='n=1000；d=2/3，ρ=0/0.5；四种分布，共 16 条件、16000 份数据。',
 G3='n=500；标量 4 条件、多元 16 条件；每条件 R=1000，共 20000 份数据。',
 G4='n=2000；标量 4 条件、多元 16 条件；每条件 R=1000，共 20000 份数据。',
 G5='n=1000；标量和三维 ρ=0.5，各四种分布；每条件 R=1000，共 8000 次 RF 流程。复用 G1/G2 对应原始数据和外层 folds。',
 G6A='n=1000；标量与三维 ρ=0.5；平滑 lognormal σ=1，τ=0.5；两条件各 R=1000，共 2000 份新数据。对照为 G1/G2 的 τ=0.2。',
 G6B='n=1000；标量与三维 ρ=0.5；σ=0.75/1。复用 4000 份既有数据和残差，各计算 c=0.5、2，共 8000 次参数重算；c=1 为既有主结果。')
COUNT = dict(G0='7200（smoke + pilot）',G1='4000',G2='16000',G3='20000',G4='20000',G5='8000（复用主数据）',G6A='2000（新分布数据）',G6B='4000 份缓存 × 2 个变体')
REPLAY = dict(G0=1200,G1=120,G2=480,G3=600,G4=600,G5=240,G6A=60,G6B=240)
CSS = '''body{font-family:"Microsoft YaHei",sans-serif;max-width:1180px;margin:40px auto;padding:0 25px;color:#172b3a;line-height:1.7;background:#fbfcfe}h1{font-size:30px;border-bottom:3px solid #127b78;padding-bottom:14px}h2{margin-top:35px;color:#126c72}a{color:#086b9d}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0;background:white}th,td{border:1px solid #dbe3eb;padding:8px;text-align:right}th{background:#eaf1f5}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f6f8fa}img{max-width:100%;height:auto}code{background:#edf1f5;padding:2px 5px}input{width:95%;padding:12px;font-size:17px;border:1px solid #bbc9d8;border-radius:5px}details{background:white;padding:12px 18px;margin:15px 0;border:1px solid #dbe3eb}.note{color:#536579}footer{margin-top:40px;border-top:1px solid #ddd;padding-top:15px;font-size:13px} @media print{body{margin:10px}a{color:inherit}table{font-size:10px}h2{break-after:avoid}tr,img{break-inside:avoid}}'''


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row,key):
    return float(row[key])


def rel(path, parent):
    return Path(os.path.relpath(path,parent)).as_posix()


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'] +
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def write_html(path, title, md):
    body = markdown.markdown(md,extensions=['tables','fenced_code'])
    def html_link(match):
        stem=match.group(1)
        name=Path(stem).name
        return 'href="'+stem+('.html' if name.endswith('_实验报告') or name in ['代码索引','G0-G6_实验总览'] else '.md')+'"'
    body = re.sub(r'href="([^"]+)\.md"',html_link,body)
    path.write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style><body>{body}<footer>Rank-DML 实验汇总 · 2026-11-17 · PDF 为正式矢量图，PNG 仅供预览。</footer></body></html>',encoding='utf-8')


def primary_rows(stage):
    if stage.startswith('G6'):
        full=read(ROOT/(stage+'实验')/'汇总表_CSV/汇总表.csv')
        return [dict(r,metric=r['RMSE'] if r['table']=='coordinates' else r['vector_RMSE']) for r in full
                if (r['table']=='coordinates' and number(r,'d')==1) or (r['table']=='vectors' and number(r,'d')>1)]
    tables=ROOT/(stage+'实验')/'汇总表_CSV'/BATCH[stage]
    scalar=[dict(r,metric=r['RMSE']) for r in read(tables/'coordinates.csv') if number(r,'d')==1]
    vector=[dict(r,metric=r['vector_RMSE']) for r in read(tables/'vectors.csv') if number(r,'d')>1]
    return scalar+vector


def key(r):
    return (int(number(r,'d')), number(r,'rho'), LAWS.index(r['law']), number(r,'n'),
            float(r.get('tau') or r.get('c') or 0))


def label(r,stage):
    d=int(number(r,'d'))
    s=('标量' if d==1 else f'd={d},ρ={number(r,"rho"):g}')+' / '+r['law']
    if stage=='G0':s+=f' / n={number(r,"n"):g}'
    if stage.startswith('G6'):
        axis='tau' if stage=='G6A' else 'c'
        s+=f' / {axis}={number(r,axis):g}'
    return s


def grouped(rows,pipe='learned'):
    result=collections.defaultdict(dict)
    for r in rows:
        if r['pipeline']==pipe:result[key(r)][r['method']]=r
    for q in result.values():assert set(q)==set(METHODS)
    return dict(sorted(result.items()))


def primary_figures(stage):
    base=ROOT/(stage+'实验')/'图'
    if stage in ['G0','G1','G3']:
        names=[f'{stage}_d1_rho0_sln100_learned_RMSE.pdf',f'{stage}_d1_rho0_sln100_learned_coverage.pdf']
        if stage!='G1':names += [f'{stage}_d3_rho0.5_sln100_learned_vector_RMSE.pdf',f'{stage}_d3_rho0.5_sln100_learned_coverage.pdf']
        else:names += ['G1_d1_rho0_t3_learned_RMSE.pdf','G1_d1_rho0_sln100_learned_paired_RMSE.pdf']
    elif stage=='G2':
        names=['G2_d3_rho0.5_sln100_learned_vector_RMSE.pdf','G2_d3_rho0.5_sln100_learned_coverage.pdf','G2_d2_rho0.0_gaussian_learned_coverage.pdf','G2_d3_rho0.5_sln100_learned_paired_RMSE.pdf']
    elif stage in ['G4','G5']:
        base=base/BATCH[stage];n=2000 if stage=='G4' else 1000
        names=[f'coordinates_sln100_{n}_1_0.0_learned_1_RMSE.pdf',f'coordinates_sln100_{n}_1_0.0_learned_1_coverage.pdf',f'vectors_sln100_{n}_3_0.5_learned_vector_RMSE.pdf',f'vectors_sln100_{n}_3_0.5_learned_coverage.pdf']
    else:names=['01_scalar_RMSE.pdf','02_scalar_coverage.pdf','03_vector_RMSE.pdf','04_joint_coverage.pdf']
    paths=[base/name for name in names]
    for p in paths:assert p.exists(),str(p)
    return paths


def quantitative_conclusion(stage, rows):
    learned=grouped(rows)
    if stage=='G0':
        return ['九项实现检查全部通过；T1 与 T8 明确保留有限模拟次数的不确定性。',
                '本组只用于实现、数值稳定性和流程诊断；不使用 pilot 的方法排序或覆盖率作为正式结论。',
                '并列残差处理修复后，用相同种子重跑 smoke，再运行独立种子的 pilot；未按覆盖率调大标准误。']
    if stage=='G6A':
        return ['把平滑参数 τ 从 0.2 改为 0.5 后，W 的标量 RMSE 从 0.01976 增至 0.02128，三维整体 RMSE 从 0.02974 增至 0.03410。',
                'W 的标量覆盖率从 91.9% 变为 93.1%，三维联合覆盖率从 93.5% 变为 92.5%；覆盖率没有统一改善。',
                '两组来自不同分布的新数据，不作同数据配对显著性判断；该组不能用于事后选择较有利的平滑参数。']
    if stage=='G6B':
        return ['W 的直接加权中位数点估计不依赖 c，因此三种步长下的 RMSE 完全相同；c 会影响斜率估计、标准误和覆盖率。',
                'N 的 RMSE 对 c=0.5/1/2 的变化很小；不同条件下覆盖率变化的方向不完全一致，详细数值见下表。',
                '主定义继续保留 c=1。该组复用四个条件的残差缓存，不是新增 8000 份独立数据，也不按覆盖率选择步长。']
    winners=collections.Counter(min(q.values(),key=lambda r:number(r,'metric'))['method'] for q in learned.values())
    text=['按未四舍五入的 RMSE 数值排序，'+ '、'.join(f'{m} 在 {winners[m]} 个条件最小' for m in METHODS if winners[m])+f'（共 {len(learned)} 条件）。数值最小不等于对所有方法均有显著优势。']
    gains=[1-number(q['W'],'metric')/number(q['LS'],'metric') for q in learned.values() if q['W']['law']!='gaussian']
    text.append(f'在 t(3) 和两种偏态分布下，W 相对 LS 的 RMSE 降幅为 {min(gains):.1%}–{max(gains):.1%}。与 H/HT 的比较更严格，需同时查看配对区间和覆盖率。')
    for scalar in [True,False]:
        cov=[number(r,'coverage') for q in learned.values() for r in q.values() if (number(r,'d')==1)==scalar]
        if cov:text.append(('标量区间' if scalar else '多元联合')+f'覆盖率范围为 {min(cov):.1%}–{max(cov):.1%}；名义水平为 95%，不能仅凭较小 RMSE 判定推断校准充分。')
    if stage=='G4':text.append('与 n=500、1000 的正式数据配合后，主实验 60 个条件全部完成；样本量趋势表保留各方法原始数值，不以 pilot 替代正式对照。')
    if stage=='G5':text.append('本次 RF 配置未显示整体优于样条 Ridge：W 在三个非 Gaussian 条件的标量 RMSE 高约 19.3%–40.4%，三维整体 RMSE 高约 6.1%–17.8%。这只针对当前预设配置。')
    return text


def copy_provenance(stage):
    code=ROOT/(stage+'实验')/(stage+'实验_代码')
    science=code/'科学规范';science.mkdir(exist_ok=True)
    for name in ['README.md','rank_dml_guide.tex']:
        if not (science/name).exists():shutil.copy2(ARCHIVE/'sources'/name,science/name)
    if stage in ['G0','G1','G2','G3']:
        evidence=code/'运行记录';evidence.mkdir(exist_ok=True)
        batches=['g0_smoke_scalar','g0_smoke_vector','g0_pilot'] if stage=='G0' else [BATCH[stage]]
        for batch in batches:
            source=ARCHIVE/'runs'/batch/'config.json'
            shutil.copy2(source,evidence/(batch+'_config.json'))
        audit=ARCHIVE/'reports'/('final_audit.json' if stage=='G0' else stage+'_final_audit.json')
        shutil.copy2(audit,evidence/'验收.json')
        if stage=='G0':shutil.copy2(ARCHIVE/'reports/implementation_checks.json',evidence/'implementation_checks.json')
    # Notebook discovery must work from the new desktop location. Root Python files stay unchanged.
    for path in code.glob('*.ipynb'):
        nb=json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            cell['source']=[line for line in cell.get('source',[]) if not line.lstrip().startswith('candidates.append(Path(')]
            if cell['cell_type']=='code':
                src=''.join(cell['source'])
                needle='CODE_DIR = next('
                if needle in src:
                    src=src.replace(needle,'candidates.append(Path('+repr(str(code))+'))\n'+needle,1)
                    cell['source']=src.splitlines(keepends=True)
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return code


def generate():
    poppler=Path(r'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\native\poppler\Library\bin\pdftoppm.exe')
    overview=[];code_manifest=[]
    allrows={stage:primary_rows(stage) for stage in STAGES}
    for stage in STAGES:
        target=ROOT/(stage+'实验');code=copy_provenance(stage)
        report=target/'结果_MD'/(stage+'_实验报告.md')
        rows=allrows[stage]
        shown=[r for r in rows if number(r,'n')==1000] if stage=='G0' else rows
        groups=grouped(shown);oracle=grouped(shown,'oracle')
        figs=primary_figures(stage)
        pdfs=sorted((target/'图').rglob('*.pdf'))
        previews=target/'图/报告预览';previews.mkdir(exist_ok=True)
        for i,p in enumerate(figs):
            dest=previews/f'{stage}_重点图{i+1}'
            subprocess.run([str(poppler),'-scale-to','1100','-png','-singlefile',str(p),str(dest)],check=True,capture_output=True)
        csv_path=(target/'汇总表_CSV/汇总表.csv') if stage in ['G0','G1','G2','G3','G6A','G6B'] else (target/'汇总表_CSV'/BATCH[stage]/'汇总表.csv')
        assert csv_path.exists()
        md=f'# {stage} 实验报告：{TITLE[stage]}\n\n整理日期：2026-11-17。'+('**实现与 pilot，不属于正式科学结论。**' if stage=='G0' else '**正式实验 / 预设敏感性结果。**')+'\n\n'
        md+='## 结论\n\n'+'\n'.join('- '+s for s in quantitative_conclusion(stage,rows))+'\n\n'
        md+='## 设计与完成情况\n\n'+DESIGN[stage]+'\n\n'
        md+='五方法为 LS（最小二乘）、H（带 fold 截距的中心化 Huber，乘子 1.345）、HT（训练内调参 Huber）、W（直接 Wilcoxon 秩估计）、N（一步 normal scores）。同次重复共享数据、5 个外层 folds、第一阶段预测和残差；Oracle 仅用于诊断，不进入可行估计与调参。\n\n'
        md+=('第一阶段为固定 RF：500 棵树、叶节点至少 10 个样本、全特征候选、训练集大小 bootstrap、无深度限制；内层辅助拟合使用相同设定。' if stage=='G5' else '第一阶段为样条 Ridge：三次 B-spline、训练集内 knots 与标准化、线性外推及前五变量两两交互；按训练集 GCV 从 α=0.1/1/10/100 选择。')+'\n\n'
        md+=f'完成记录：{COUNT[stage]}；未记录方法/区间失败；已有验收复算 {REPLAY[stage]} 条方法记录通过。逐次数据和审计证据保留，本次只整理报告、图格式与目录，没有重跑实验。\n\n'
        if stage=='G0':
            md+=table(['批次','条件数','每条件 R','数据重复'],[['标量 smoke',12,20,240],['多元 smoke',48,20,960],['完整 pilot',60,100,6000]])+'\n'
            names=['生成分布矩','外层与内层无验证泄漏','W 最小化、中心化与并列处理','fold 平移不变性','有限差分与 N 更新符号','正尺度变换','HC0 与完整协方差一致性','Oracle Gaussian 基准','同数据输入与失败保留']
            checks=json.loads((ARCHIVE/'reports/implementation_checks.json').read_text(encoding='utf-8'))
            md+=table(['检查','内容','结果'],[[f'T{i}',name,checks[f'T{i}']['status']] for i,name in enumerate(names,1)])+'\n'
            md+='历史 smoke 使用 8 workers，pilot 曾在旧设置下运行，重启后统一为最多 12 个单线程 worker；该历史事实保留。下表仅展示 pilot 的 n=1000 切片，其余 n=500/2000 数据全部在 CSV 中。\n\n'
        else:md+='运行设置为最多 12 个计算 worker，每个 worker 的数值库线程为 1。\n\n'
        md+='## 精度结果\n\n下表为 learned 流程；标量使用 RMSE，多元使用整体 vector RMSE。各列均保留五方法，数值越小表示点估计误差越小。\n\n'
        md+=table(['条件']+METHODS,[[label(q['W'],stage)]+[f'{number(q[m],"metric"):.5f}' for m in METHODS] for q in groups.values()])+'\n'
        md+='## 覆盖率\n\n标量为边际 95% 区间，多元为 95% Wald 联合椭球，不能与“所有坐标区间同时覆盖”混用。分母为全部尝试次数，无效区间计未覆盖；有效分母指标另见 CSV。\n\n'
        md+=table(['条件']+METHODS,[[label(q['W'],stage)]+[f'{number(q[m],"coverage"):.1%}' for m in METHODS] for q in groups.values()])+'\n'
        md+=('R=100 的 pilot 在 coverage≈95% 时 MC SE≈2.18 个百分点。' if stage=='G0' else 'R=1000 在 coverage≈95% 时 MC SE≈0.69 个百分点。')+'图中保留 95% 参考线与 ±1.96 MC SE，误差条反映模拟精度；CSV 另含二项区间。\n\n'
        md+='## Oracle 诊断\n\n以 W 为例比较 learned 与 oracle；这里只作对照，不能把 oracle 结果当作可行方法性能。\n\n'
        md+=table(['条件','learned RMSE','oracle RMSE','learned 覆盖','oracle 覆盖'],[[label(q['W'],stage),f'{number(q["W"],"metric"):.5f}',f'{number(oracle[k]["W"],"metric"):.5f}',f'{number(q["W"],"coverage"):.1%}',f'{number(oracle[k]["W"],"coverage"):.1%}'] for k,q in groups.items()])+'\n'
        if stage in BATCH and stage!='G0':
            pairs=read(target/'汇总表_CSV'/BATCH[stage]/'paired.csv')
            unique={}
            for r in pairs:
                if r['pipeline']=='learned' and r['method']=='W' and r['baseline'] in ['LS','H','HT']:
                    k=(key(r),r['baseline'])
                    if k in unique:assert abs(number(unique[k],'RMSE_ratio')-number(r,'RMSE_ratio'))<1e-10
                    unique[k]=r
            def pair_text(k,m):
                r=unique[(k,m)]
                return f'{number(r,"RMSE_ratio"):.3f} [{number(r,"ratio_lower"):.3f}, {number(r,"ratio_upper"):.3f}]'
            md+='## 配对比较\n\nRMSE 比及 2000 次配对 bootstrap 的 95% 区间；小于 1 表示 W 的 RMSE 较低，区间跨 1 时不判定明确优势。比较采用共同有效重复，完整分母、MSE 差异与区间长度差见 paired.csv；N 的对应比较也完整保留。多元比值使用整体 vector RMSE。\n\n'
            md+=table(['条件','W / LS','W / H','W / HT'],[[label(q['W'],stage)]+[pair_text(k,m) for m in ['LS','H','HT']] for k,q in groups.items()])+'\n'
        if stage in ['G3','G4']:
            baseline=grouped(allrows['G1']+allrows['G2']);small=grouped(allrows['G3']);large=grouped(allrows['G4'])
            by_n=[]
            for q in groups.values():
                r=q['W'];k=key(r);lo=(k[0],k[1],k[2],500.,0.);mid=(k[0],k[1],k[2],1000.,0.);hi=(k[0],k[1],k[2],2000.,0.)
                by_n.append([label(r,stage)]+[f'{number(a[kk]["W"],"metric"):.5f} / {number(a[kk]["W"],"coverage"):.1%}' for a,kk in [(small,lo),(baseline,mid),(large,hi)]])
            md+='## 正式样本量对照\n\n以下以 W 为例，每格为“RMSE / 覆盖率”。n=500、1000、2000 使用各自正式独立重复，不能按同次数据作配对比较；其他方法完整结果在对应 CSV。\n\n'+table(['条件','n=500','n=1000','n=2000'],by_n)+'\n'
        if stage=='G5':
            base=grouped(allrows['G1']+allrows['G2'])
            md+='## 与样条 Ridge 对照\n\n每格为 RF RMSE / Spline Ridge RMSE，超过 1 表示本次 RF 配置误差更大。数据与 outer folds 按方案复用，算法种子独立。此表是点估计比值，不将未计算的跨学习器配对区间虚构为显著性证据。\n\n'
            md+=table(['条件']+METHODS,[[label(q['W'],stage)]+[f'{number(q[m],"metric")/number(base[k][m],"metric"):.3f}' for m in METHODS] for k,q in groups.items()])+'\n'
        if stage=='G6B':md+='LS/H/HT 的估计、诊断与历史耗时复用；W/N 的二阶段耗时是本次缓存重算，不能解释为五方法全部重新拟合的总成本。\n\n'
        md+='## 重点图与完整数据\n\n'+f'[完整汇总 CSV]({rel(csv_path,report.parent)}) · [全部 PDF 图索引](../图/PDF图表索引.html) · [代码索引](../{stage}实验_代码/代码索引.md)\n\n'
        for i,p in enumerate(figs):
            preview=previews/f'{stage}_重点图{i+1}.png'
            md+=f'### 图 {i+1}\n\n[打开 PDF 矢量原图]({rel(p,report.parent)})\n\n![{stage} 重点图 {i+1}]({rel(preview,report.parent)})\n\n'
        md+='## 解释范围与复现\n\n固定四种分布、预设样本量和学习器构成本研究范围；偏态下的优势不是所有误差分布下的普遍优势。RMSE、覆盖率、区间长度和配对不确定性应结合解释；不删除不利结果、不以 observed coverage 调参。\n\n'
        md+=f'科学规范见代码目录“科学规范/rank_dml_guide.tex”；分组见同目录 README。统计数据源为本组 CSV，运行配置、种子和验收证据见代码索引。原始 G0–G3 的逐次缓存保留在 `{ARCHIVE}`；G4–G6 的 `_work` 已随本目录移至桌面。代码依赖现有 Python 环境和原始研究档案，代码压缩包不包含这些大型依赖。\n\n'
        md+='[返回总览](../../G0-G6_实验总览.md)\n'
        report.write_text(md,encoding='utf-8');write_html(report.with_suffix('.html'),stage+' 实验报告',md)
        # A searchable index exposes all figures without opening hundreds of tabs.
        entries=''.join(f'<tr><td>{i+1}</td><td><a href="{html.escape(rel(p,target/"图"),quote=True)}">{html.escape(p.stem)}</a></td><td>PDF</td></tr>' for i,p in enumerate(pdfs))
        page=f'<!doctype html><meta charset="utf-8"><title>{stage} PDF 图表索引</title><style>{CSS}</style><h1>{stage} PDF 图表索引</h1><p>共 {len(pdfs)} 张。搜索分布、learned/oracle、指标或坐标。<a href="../结果_MD/{stage}_实验报告.html">返回实验报告</a></p><input id="filter" placeholder="筛选图名，例如 sln100 learned coverage"><table><thead><tr><th>序号</th><th>图名（点击打开 PDF）</th><th>格式</th></tr></thead><tbody>{entries}</tbody></table><script>document.querySelector("#filter").addEventListener("input",e=>{{let words=e.target.value.toLowerCase().trim().split(/\\s+/);document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!words.every(w=>r.textContent.toLowerCase().includes(w)))}})</script>'
        (target/'图/PDF图表索引.html').write_text(page,encoding='utf-8')
        indexmd=f'# {stage} 全部 PDF 图\n\n共 {len(pdfs)} 张。[可搜索索引](PDF图表索引.html)\n\n'+'\n'.join(f'- [{p.stem}](<{rel(p,target/"图")}>)' for p in pdfs)+'\n'
        (target/'图/PDF图表索引.md').write_text(indexmd,encoding='utf-8')
        excluded={'_work','dependencies','__pycache__','_maintenance'}
        files=[p for p in code.rglob('*') if p.is_file() and not any(v in excluded for v in p.relative_to(code).parts) and p.suffix in ['.py','.ipynb','.md','.json','.tex','.txt','.ps1','.cmd'] and p.name not in ['代码索引.md','代码索引.html']]
        codemd=f'# {stage} 代码与复现索引\n\n'+('本组为历史源码快照，未另造 notebook；运行依赖原始研究档案的配置和缓存。' if stage in ['G0','G1','G2','G3'] else f'入口：[启动{stage}.ipynb](启动{stage}.ipynb)。默认最多 12 个单线程 worker。已有结果已完成；如只需重新汇总，使用 notebook 的 export 模式。')+'\n\n'
        codemd+='脚本与 notebook 同目录保存。`core.py` 为统计实现，`run.py` / `runner.py` 为调度入口，`reporting.py` 为指标汇总，交付/绘图脚本另列。原始工作缓存不纳入代码压缩包；目录中的 `_work` 仍完整保留。\n\n'
        codemd+='\n'.join(f'- [{p.relative_to(code).as_posix()}](<{p.relative_to(code).as_posix()}>)' for p in sorted(files))+'\n'
        (code/'代码索引.md').write_text(codemd,encoding='utf-8');write_html(code/'代码索引.html',stage+' 代码索引',codemd)
        for p in files:code_manifest.append(dict(stage=stage,path=rel(p,ROOT),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        overview.append([stage,TITLE[stage],COUNT[stage],len(pdfs),f'[报告]({rel(report,ROOT)})',f'[CSV]({rel(csv_path,ROOT)})',f'[代码]({stage}实验/{stage}实验_代码/代码索引.md)',f'[PDF 图]({stage}实验/图/PDF图表索引.html)'])
        print(stage,'report complete; PDFs',len(pdfs),flush=True)
    total=sum(r[3] for r in overview)
    md='# G0–G6 实验总览\n\n八组实验均已完成。报告提供 MD 和 HTML；图统一提供 PDF 矢量版本，原 SVG 保留。每组仍使用“代码、汇总表_CSV、结果_MD、图”四个子文件夹。\n\n'
    md+=table(['实验','内容','完成量','PDF 数','报告','数据','代码','图'],overview)+'\n'
    md+='## 研究结论\n\n- G0 完成九项实现核验，只是 smoke/pilot，不计入正式结论。\n- G1–G4 完成全部 60 个样条 Ridge 主条件，共 60000 次正式数据重复。W/N 在许多偏态条件下精度较好，但不能忽略覆盖率不足；与 Huber 的优势要看配对区间。\n- G5 复用 8000 份主数据执行 RF 流程，当前配置未整体优于样条 Ridge，强偏态标量下覆盖率下降较明显。\n- G6A 新增 2000 份更强平滑数据；W 的 RMSE 上升，覆盖率变化不一致。G6B 复用 4000 份主缓存检查两种额外步长，W 点估计不变，推断对步长存在小幅敏感性。\n\n'
    md+='**不要把 G5、G6B 的复用次数加成独立新数据。** 正式主实验、pilot 和敏感性各自分列。\n\n'
    md+=f'## 文件与使用\n\n- 共 {total} 张正式 PDF：1344 张由 G0–G3 的 SVG 转存，G4–G6 原有 PDF 保留。报告只展示每组 4 张重点图，完整图可在索引中搜索。\n- [全部代码压缩包](G0-G6_代码汇总.zip)：脚本、notebook、规范与配置；不包含大型缓存和安装依赖。\n- 每份报告的数据表均来自旁边 CSV。PNG 只供报告预览，下载正式图请选 PDF。\n- G4–G6 代码目录中的运行缓存已一起移至桌面；G0–G3 原始缓存仍位于 `{ARCHIVE}`。现有 settings 与脚本依赖该档案和本机 Python 环境，不能仅凭代码压缩包在另一台电脑直接复现。\n- CSV 关闭时保存的舍入版本已另存为 `G6B实验/G6B实验_代码/_maintenance/用户保存版_汇总表_20261117.csv`；报告使用验收哈希一致的全精度主表。\n\n本次只做结果整理、SVG 转 PDF 和文件迁移，没有启动新实验。\n'
    (ROOT/'G0-G6_实验总览.md').write_text(md,encoding='utf-8');write_html(ROOT/'G0-G6_实验总览.html','G0–G6 实验总览',md)
    (ROOT/'使用说明.md').write_text('# 实验交付\n\n请先打开 [G0–G6 实验总览](G0-G6_实验总览.html)。八份报告、全部 CSV、代码索引和 PDF 图入口均在其中。\n\n每组保留代码、CSV、结果和图四个目录；原 SVG 保留，PDF 为正式交付图。\n',encoding='utf-8')
    (HERE/'代码文件清单.json').write_text(json.dumps(code_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'G0-G6_代码汇总.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for item in code_manifest:z.write(ROOT/item['path'],item['path'])
        for stage in STAGES:
            code=ROOT/(stage+'实验')/(stage+'实验_代码')
            for name in ['代码索引.md','代码索引.html']:z.write(code/name,rel(code/name,ROOT))
    print('All eight reports and desktop index complete; total PDFs:',total,flush=True)


if __name__=='__main__':generate()

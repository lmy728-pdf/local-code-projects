"""Update both notebooks without executing them."""
import ast,json,pathlib,shutil
from datetime import datetime
ROOT=pathlib.Path(__file__).resolve().parent
targets=[ROOT.parent/'Huber_DML_pilot.ipynb',ROOT.parent/'毕设'/'Huber_DML_pilot.ipynb']
backup=ROOT/'backups'/('progress_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True,exist_ok=False)
source=(ROOT/'training_progress.py').read_text(encoding='utf-8')
for i,path in enumerate(targets):
    shutil.copy2(path,backup/f'notebook_{i}.ipynb')
    nb=json.loads(path.read_text(encoding='utf-8'))
    nb['cells'][6]['source']=['## 进度条、累计summary和覆盖率柱状图\n每完成一组数据（A/B/C全部完成）推进一次进度；每场景20、40……200次累计输出。桌面只更新两个文件，逐次数据保存在run目录。\n']
    nb['cells'][7]['source']=source.splitlines(keepends=True)
    setup=''.join(nb['cells'][9]['source'])
    setup=setup.replace('config.update(',"config.update(report_interval=20,\n              desktop_summary=str(ROOT.parent/'Huber_DML_summary.csv'),\n              desktop_figure=str(ROOT.parent/'Huber_DML_coverage.png'),\n              ",1)
    nb['cells'][9]['source']=setup.splitlines(keepends=True)
    nb['cells'][10]['source']=['## 执行实验\n总进度为600组数据；每组完成A/B/C后更新进度条。先检查每场景2次，再累计输出20、40……200次结果。桌面summary和柱状图在各节点更新；失败记录保留，不换seed。\n']
    execution='''if _experiment_started:
    raise RuntimeError("当前run已经启动过；如要开始新实验，请先重跑环境检查与新建run单元")
_experiment_started = True
summary = train_with_progress(config, output, manifest['legacy_run'],
                              run_replication, ROOT.parent, workers=args.workers)
print("桌面 summary:", ROOT.parent / "Huber_DML_summary.csv")
print("桌面柱状图:", ROOT.parent / "Huber_DML_coverage.png")
print("本轮详细记录:", output)
'''
    nb['cells'][11]['source']=execution.splitlines(keepends=True)
    nb['cells'][12]['source']=['## 查看最终整合结果\n只显示一个summary和一张柱状图；这里读取本轮目录，避免误读其他run的桌面最新结果。\n']
    nb['cells'][13]['source']='''from IPython.display import display, Image
final_summary = pd.read_csv(output / "summary.csv")
display(final_summary[["scenario", "method", "completed", "valid", "failed", "coverage",
                       "bias", "mean_T", "sd_T", "outcome_rmse", "outcome_mae"]].round(4))
display(Image(filename=str(output / "coverage_comparison.png")))
'''.splitlines(keepends=True)
    nb['cells'][0]['source']+=['\n\n### 进度和精简输出\n正式R=200时共600组数据（1800条方法记录）。进度条逐组更新；每场景累计20次更新一张汇总表，同时刷新桌面 `Huber_DML_summary.csv` 与 `Huber_DML_coverage.png`。桌面文件为最新视图，旧run完整记录保留；Excel占用CSV时会提示导出失败，但run目录仍有结果。只修改代码，尚未启动本轮训练。\n']
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']));cell['outputs']=[];cell['execution_count']=None
            cell['metadata'].pop('execution',None)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
runner=ROOT/'run_huber_pilot.py'
shutil.copy2(runner,backup/runner.name)
s=runner.read_text(encoding='utf-8').replace('from summarize_pilot import summarize','from training_progress import train_with_progress')
start=s.index('    rows=[]');end=s.index("if __name__=='__main__':main()",start)
s=s[:start]+'''    train_with_progress(config,output,manifest['legacy_run'],run_replication,ROOT.parent,workers=args.workers)
    print('Saved:',output)
'''+s[end:]
ast.parse(s);runner.write_text(s,encoding='utf-8')
with (ROOT/'README.md').open('a',encoding='utf-8') as f:
    f.write('\n\n2026-11-02 进度输出更新：600组数据逐组进度条（耗时/ETA），每场景累计20、40……200次输出。桌面只保留最新Huber_DML_summary.csv及Huber_DML_coverage.png；内部run目录保留summary、图、replications、config及任务/数组。新的运行入口不再自动生成多份分析CSV或Markdown报告。运行时需要已安装的tqdm 4.67.3；无需额外安装。\n')
print('Progress and compact output installed. No simulation started.')

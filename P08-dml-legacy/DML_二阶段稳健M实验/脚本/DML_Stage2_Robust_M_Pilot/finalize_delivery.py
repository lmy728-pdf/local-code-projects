"""Refresh the saved notebook figure without rerunning any computation."""
import base64, json, ast
import nbformat
from data_access import ROOT, dump, sha

path=ROOT.parent/'DML_Stage2_Robust_M_Pilot.ipynb'
nb=nbformat.read(path,as_version=4)
images=0; tables=0; errors=[]
for cell in nb.cells:
    if cell.cell_type=='code':
        ast.parse(cell.source)
        for out in cell.get('outputs',[]):
            if out.output_type=='error':errors.append(out)
            if 'image/png' in out.get('data',{}):
                out.data['image/png']=base64.b64encode((ROOT/'coverage_comparison.png').read_bytes()).decode('ascii')
                images+=1
            if 'text/html' in out.get('data',{}) and '<table' in out.data['text/html']:tables+=1
assert images==1 and tables==1 and not errors
nb.cells[7].source += '''

本轮已完成：600 组数据、2400 条结果，失败 0。Student-t 下 A→B coverage 为 27.5%→53.0%；lognormal 下 A→C 为 32.0%→40.0%。D 在两类重尾下分别为 25.0%、20.0%，没有实现本轮预期的保护。所有方法仍明显欠覆盖，不据此宣称渐近有效。

[中文审计报告](DML_Stage2_Robust_M_Pilot/pilot_report.md) · [完整汇总表](DML_Stage2_Robust_M_Pilot/summary.csv)
'''
nbformat.validate(nb)
nbformat.write(nb,path)
locations=[]
for p in ROOT.glob('*.py'):
    for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
        if isinstance(n,ast.FunctionDef):locations.append(dict(file=str(p),function=n.name,line=n.lineno))
dump(ROOT/'source_locations.json',locations)
dump(ROOT/'delivery_manifest.json',dict(notebook=str(path),notebook_sha256=sha(path),
     notebook_table_count=tables,notebook_figure_count=images,notebook_errors=0,
     files={str(p):sha(p) for p in [ROOT/'summary.csv',ROOT/'replication_results.csv',ROOT/'coverage_comparison.png',ROOT/'pilot_report.md']},
     default_recompute=False,note='Image refreshed after visual QA. Existing execution results retained; no refitting or pilot rerun.'))
print('Notebook finalized: one table, one figure, no error outputs.')

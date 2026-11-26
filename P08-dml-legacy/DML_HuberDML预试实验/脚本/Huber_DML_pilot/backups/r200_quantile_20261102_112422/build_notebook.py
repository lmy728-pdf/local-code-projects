"""Create notebook without executing any of its cells."""
import ast,json,pathlib,textwrap,uuid
ROOT=pathlib.Path(__file__).resolve().parent
TARGET=ROOT.parent/'Huber_DML_pilot.ipynb'
def cell(kind,source):
    result=dict(cell_type=kind,id=uuid.uuid4().hex[:8],metadata={},source=source.splitlines(keepends=True))
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result
def main():
    if TARGET.exists():raise FileExistsError(TARGET)
    core=(ROOT/'legacy_core.py').read_text(encoding='utf-8')
    methods=(ROOT/'huber_methods.py').read_text(encoding='utf-8')
    methods='\n'.join(line for line in methods.splitlines() if not line.startswith('from legacy_core import '))+'\n'
    reporting=(ROOT/'summarize_pilot.py').read_text(encoding='utf-8')
    runner=(ROOT/'run_huber_pilot.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(runner).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    body=textwrap.dedent('\n'.join(runner.splitlines()[node.body[0].lineno-1:node.end_lineno]))
    body=body[body.index("config=json.loads"):]
    setup,execute=body.split('rows=[]',1)
    execute='rows=[]'+execute
    setup=setup.replace("config.update(stage=args.stage", "config.update(notebook_sha256=sha(NOTEBOOK),stage=args.stage")
    cells=[
        cell('markdown','''# Huber-DML pilot（代码版，尚未运行）

只稳健化第一阶段 outcome nuisance；A=原HGBR平方损失，B=GBR平方损失，C=GBR Huber(alpha=0.90)。主要比较C−B。原DGP、treatment预测、fold、第二阶段及SE保持不变。

在VS Code中选择 `D:/Anoconda/envs/environment2/python.exe` 内核，然后按顺序运行。默认 `STAGE="smoke"`（每场景2次）；改为 `"pilot"` 才会每场景100次，并先检查其中前2次。拟合仅在最后的“执行实验”单元启动。

此notebook已包含方法与汇总代码。仍需保留桌面的 `Huber_DML_pilot` 文件夹（原配置及hash清单）和既有DML结果（用于核验A）。所有新输出进入该文件夹的outputs新run目录，不覆盖旧结果。重复运行执行单元前，须先重跑“环境检查与新建run”单元。
'''),
        cell('code','''# 用户参数：不提供alpha、DGP或R网格搜索
STAGE = "smoke"     # "smoke": 每场景2次；"pilot": 每场景100次
WORKERS = 4

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import pathlib, json, hashlib, platform, uuid
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import scipy, sklearn, joblib
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
ROOT = pathlib.Path(r"C:/Users/Lenovo/Desktop/Huber_DML_pilot")
NOTEBOOK = ROOT.parent / "Huber_DML_pilot.ipynb"
if STAGE not in ("smoke", "pilot") or not 1 <= WORKERS <= 16:
    raise ValueError("检查STAGE和WORKERS")
args = SimpleNamespace(stage=STAGE, workers=WORKERS)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
'''),
        cell('markdown','## 原DGP、A第一阶段与第二阶段SE\n以下13个原函数原样提取；仅定义函数，不训练模型。'),
        cell('code',core),
        cell('markdown','## 三组方法及逐次诊断\nB/C同训练集、同随机状态，仅loss不同。所有方法使用同一份m_hat；A与旧保存结果逐次核验。'),
        cell('code',methods),
        cell('markdown','## 汇总、配对比较与报告函数\n只定义函数。Huber不保证估计条件均值；偏态误差尤其需要检查位置偏移。跨场景gain仅作描述，不强加独立性假设。'),
        cell('code',reporting),
        cell('markdown','## 环境检查与新建run\n本单元检查版本、原源码hash和参数，创建新的输出目录，但不拟合模型。修改STAGE后，请从参数单元重新运行。'),
        cell('code',setup+'\n_experiment_started = False\nprint("Output:", output)\n'),
        cell('markdown','## 执行实验\n**运行本单元才会开始拟合。** smoke失败时保留记录并停止；pilot先验证前2次，再运行余下98次。失败不会换seed重试。'),
        cell('code','''if _experiment_started:
    raise RuntimeError("该run已启动过；请先重跑环境检查与新建run单元，避免覆盖")
_experiment_started = True
'''+execute),
        cell('markdown','## 查看结果\n报告只有运行实验后才会生成；本次交付没有任何模拟结果。'),
        cell('code','''from IPython.display import display, Markdown, Image
display(pd.read_csv(output / "summary.csv"))
display(pd.read_csv(output / "paired_comparisons.csv"))
display(pd.read_csv(output / "tail_deterioration.csv"))
display(Markdown((output / "pilot_report.md").read_text(encoding="utf-8")))
display(Image(filename=str(output / "coverage_comparison.png")))
display(Image(filename=str(output / "nuisance_rmse.png")))
''')]
    for c in cells:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python (environment2)',language='python',name='python3'),
                  language_info=dict(name='python',version='3.11.15')),nbformat=4,nbformat_minor=5)
    TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created:',TARGET,'; cells:',len(cells),'; executed: 0')
if __name__=='__main__':main()

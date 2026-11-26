"""Create only this project's new desktop notebook; refuse overwrite."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
PATH=ROOT.parent/'DML_Stage2_Robust_M_Pilot.ipynb'
if PATH.exists(): raise FileExistsError('Notebook already exists; not overwritten')
md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook()
nb.cells=[
md('''# 第二阶段分布适配与 Adaptive Huber-DML

独立实验，DGP2 / n=1000 / p=5 / 每场景 R=200 / 5-fold。

三场景四方法共用原来保存的 HGBR squared-error OOF 预测；不改第一阶段，不重新训练，不修改旧 notebook 或旧结果。

**A** 原始 DML；**B** 无截距对称 Huber；**C** 有截距不对称 Huber；**D** 有截距 Adaptive Huber。
这些编号只适用于本 notebook。

完成后只显示**一张汇总表、一张九面板图**。代码和审计底稿在桌面同名文件夹。运行内核：`D:/Anoconda/envs/environment2/python.exe`。'''),
md(r'''## 1. 固定配置与数据入口

MAD 基于初步无截距 OLS 残差：$s=1.4826\operatorname{median}|r-\operatorname{median}(r)|$。
B：$\tau=1.345s$；C：$\tau_-=2s,\tau_+=1.5s$；D：$\tau_n=s\sqrt{n/\log n}$，自然对数。

D 不读取误差标签或真实参数。阈值在求解过程中固定。配置不随 coverage 调整。

`RECOMPUTE=False`：已有完整 pilot 时直接读取；第一次执行则先核验和 smoke，再执行 pilot。主动设为 True 才在新子目录重跑，保留旧结果。'''),
code(r'''from pathlib import Path
import sys, json
ROOT = Path(r"C:\Users\Lenovo\Desktop\DML_Stage2_Robust_M_Pilot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RECOMPUTE = False
from run_pilot import load_or_run, DEFAULT_SETTINGS
print("R=200 per scenario; saved OOF predictions; parameters:", DEFAULT_SETTINGS)
'''),
md(r'''## 2. 数值验证 → smoke → pilot

先检查原函数 AST、配置、版本、数组 SHA256、原种子重建的数据及 folds；确认 A 的 θ 和 SE 与旧结果相同。

数值测试包括大阈值 OLS 极限、有限差分梯度/Hessian、独立 sandwich、固定数据回归测试、Oracle 斜率与 lognormal 位置偏移。

smoke 每场景 2 组通过后才继续到每场景 200 组；这 2 组计入 200，不额外重复。每 20 组输出累计 coverage，更新同一个 summary。失败单独记录，不算作 coverage=0。

新方法：$A=\overline{\psi'zz^T}$，$B=\overline{(z\psi-\overline{z\psi})(z\psi-\overline{z\psi})^T}$，$V=A^{-1}BA^{-T}/n$。带截距时使用完整联合矩阵的斜率方差。A 方法仍用原 IF-SE 的 ddof=1。'''),
code('''OUTPUT = load_or_run(recompute=RECOMPUTE)
print("结果目录：", OUTPUT)
'''),
md('''## 3. 一张汇总表

包含有效数、失败数、Bias、经验 SD、Mean SE、RMSE、coverage、T 两矩及两侧拒绝、区间长度、阈值及 clipping、截距、Δ_tail、相对 A 配对增益及 MCSE。
完整同一张表保存为 `summary.csv`。R=200 的少数百分点差异需结合 MCSE 判断。'''),
code('''import pandas as pd
from IPython.display import display, Image
summary = pd.read_csv(OUTPUT / "summary.csv")
display(summary.style.format(precision=4, na_rep="—"))
'''),
md('''## 4. 一张组合图与审计结论

图中同时展示 coverage、Gaussian→heavy-tail 恶化、Bias、SE 校准、区间长度、clipping 和三个场景的 T 分布。

工作 sandwich 把阈值视为固定，不能自动证明 nuisance orthogonality 或 DML 渐近有效。Oracle 验证与实际残差结果分开解释；按已知分布选择 B/C 是探索性策略，D 才是跨场景统一规则。'''),
code('''display(Image(filename=str(OUTPUT / "coverage_comparison.png")))
completion = json.loads((OUTPUT / "completion.json").read_text(encoding="utf-8"))
print("完成状态：", completion)
print("中文审计报告：", OUTPUT / "pilot_report.md")
print("逐次可追溯结果：", OUTPUT / "replication_results.csv")
print("已停止；不自动扩展 R 或 n。")
''')]
nb.metadata={'kernelspec':{'display_name':'Python (environment2)','language':'python','name':'environment2'},
             'language_info':{'name':'python','version':'3.11.15'},
             'vscode':{'interpreter':{'path':'D:/Anoconda/envs/environment2/python.exe'}}}
nbf.write(nb,PATH)
print(str(PATH))

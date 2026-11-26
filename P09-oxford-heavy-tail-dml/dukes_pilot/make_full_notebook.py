"""Package reviewed functions as a self-contained, unexecuted desktop notebook."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
DESTINATION = Path('C:/Users/Lenovo/Desktop/DML_Dukes_n1000_R1000.ipynb')


def definitions(filename, names=None):
    source = (ROOT / filename).read_text(encoding='utf-8')
    return '\n\n'.join(ast.get_source_segment(source, node) for node in ast.parse(source).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and
                       (names is None or node.name in names)) + '\n'


cells = []
def md(value):
    cells.append(nbformat.v4.new_markdown_cell(value))
def code(value):
    cells.append(nbformat.v4.new_code_cell(value, execution_count=None, outputs=[]))


md('''# 原 Mean-DML 与 Dukes DR-GCM：n=1000，R=1000

**每次数据集的样本量 n=1000；每场景 Monte Carlo 重复 R=1000。**
三种误差：Gaussian、Student-t(2.1)、总体中心化/标准化 lognormal(σ=2)。
DGP2、p=5、五折 HGBR，A/B/C 在同一数据与 folds 上比较，共 3000 个数据集、9000 条方法结果。

- **A**：原 Mean-DML，学习 Y|X 和 D|X，原 Wald 区间。
- **B**：直接学习 (Y−θD)|X 的未修正 score 检验。
- **C**：与 B 共用初始预测，加入 Dukes 辅助核回归与完整 score 修正。

增加 n 才是在检查样本量渐近趋势；增加 R 是减小覆盖率的 Monte Carlo 误差。
本 notebook 不自动扩大 n。B/C 只在 θ=1 检验真值是否属于反演置信集，不计算端点、区间长度或参数点估计。

**交付时所有单元均未执行，没有新实验结果。** 本文件自包含计算与汇总代码，不依赖旧项目的 .py 或 notebook。
在 Jupyter 选 `environment2` 内核后从上到下运行即可；“开始正式实验”单元才会训练模型。
默认4个进程、每进程1个计算线程。中断后重跑同样配置，自动跳过已完成任务。
输出写入桌面新目录 `DML_Dukes_full_results/`，不会修改旧实验，也不混用上次 R=200 的结果。
''')

code('''# 运行参数：n 与 R 是不同的量
N = 1000
R = 1000
P = 5
THETA_0 = 1.0
N_FOLDS = 5
WORKERS = 4                  # 日常使用电脑可改为2；不要盲目开满32进程
BASE_SEED = 20260826
NORMAL_BASELINE = "N(0,1)"
BOOSTING_PARAMS = dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
                       min_samples_leaf=20, l2_regularization=1.0)

from pathlib import Path
OUTPUT_ROOT = Path.home() / "Desktop" / "DML_Dukes_full_results"
assert P == 5 and N == 1000 and R == 1000 and N_FOLDS == 5
assert 1 <= WORKERS <= 16
''')

md('''## 环境与来源

已有环境：`D:/Anoconda/envs/environment2/python.exe`。需要 NumPy、SciPy、scikit-learn、joblib、threadpoolctl、pandas、matplotlib 与 IPython；此 notebook 不自动安装/升级依赖。
核回归为固定规则的 **Python 移植**，未与作者 R/np 优化软件逐项交叉核验；前一轮频繁命中带宽搜索上界，因此本轮保留相同预设并报告边界，不按 coverage 调参。

[Dukes, Vansteelandt & Whitney (2024), JMLR](https://jmlr.org/papers/v25/22-1233.html)；
[作者代码（锁定版本）](https://github.com/ordukes/DRInference/blob/e7dfb038117aea3d95ca1371e2e85557ff9f25d8/dri_functions.r)。
旧实验纯函数来自两份 p=5 主 notebook；只提取函数，没有调用旧 notebook 的顶层模拟或导出单元。
''')

code('''import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import ast, csv, hashlib, inspect, json, math, sys, time, warnings
from datetime import datetime, timezone
from statistics import NormalDist, mean, stdev
from contextlib import contextmanager
import numpy as np
import scipy, sklearn, joblib
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from IPython.display import display, Markdown, Image
import pandas as pd

print({"Python": sys.version.split()[0], "NumPy": np.__version__,
       "SciPy": scipy.__version__, "scikit-learn": sklearn.__version__,
       "joblib": joblib.__version__})
''')

md('## 1. 原实验的数据生成与 Mean-DML\n保留 X→V→U 的随机调用顺序、总体矩标准化、原 HGBR 参数和 A 的标准误公式。')
student = definitions('_legacy_student.py').replace('def generate_data(', 'def generate_data_student(')
code(student)
ln = definitions('_legacy_lognormal.py', {'ar1_covariance', 'nuisance_functions', 'standardized_centered_lognormal', 'generate_data'})
ln = ln.replace('ar1_covariance', 'lognormal_ar1_covariance').replace('nuisance_functions', 'lognormal_nuisance_functions').replace('def generate_data(', 'def generate_data_lognormal(')
code(ln + '''
def generate_data(n, error_case, dgp_name, seed, theta0=THETA_0, error_family="student_t"):
    if error_family == "lognormal":
        return generate_data_lognormal(n, error_case, dgp_name, seed, theta0)
    if error_family in ("student_t", "gaussian"):
        return generate_data_student(n, "N(0,1)" if error_family == "gaussian" else error_case,
                                     dgp_name, seed, theta0)
    raise ValueError(f"Unknown error family: {error_family}")
''')

md('''## 2. Dukes 辅助核回归、更新与完整 score

每个验证折内做一维 Gaussian NW；LOOCV 选带宽时排除自身，最终拟合包含自身。
固定 41 点对数网格 `[0.05s,20s]`，`s=SD(x)·n_fold^(-1/5)`；内部最优点局部细化，边界只记录、不自动扩展。
`G=NW(b|g_hat)`、`M=NW(a|m_hat)`；无截距更新后使用完整的
`psi=rD*rY-G*rD-M*rY`，合并各折再用 `ddof=1` 的 score 标准差标准化。
常数索引使用明示均值延拓；数值失败记录为失败，不记为接受真值。
''')
score_source = (ROOT / 'dukes_score.py').read_text(encoding='utf-8')
kernel_node = next(n for n in ast.parse(score_source).body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_KERNEL_CONFIG' for t in n.targets))
code(ast.get_source_segment(score_source, kernel_node) + '\n\n' + definitions('dukes_score.py'))

md('''## 3. 同一样本上的 A/B/C 与逐次保存

原编号固定为 DGP2=1；Gaussian=7；t(2.1) 与 lognormal(2)=0。replication 从0开始。
失败不换 seed。每份数据先保存完整数组，再原子写任务记录；完成记录用于续跑。
''')
driver_names = {'sha256','clean','dump_json','make_data','initial_a','compute_task','persist_task'}
driver = definitions('run_pilot.py', driver_names).replace('legacy.', '')
code('METHODS = ("original_mean_dml", "standard_score", "dukes_dr_gcm")\n\n' + driver)

md('## 4. 汇总与 Monte Carlo 不确定性\n保留有效数、失败数、待运行数、Wilson区间、配对差及其MCSE。B/C未计算的参数估计字段留空。')
summary_source = (ROOT / 'summarize_pilot.py').read_text(encoding='utf-8')
tree = ast.parse(summary_source)
assignments = '\n'.join(ast.get_source_segment(summary_source,n) for n in tree.body if isinstance(n,ast.Assign))
summary = definitions('summarize_pilot.py')
# Standalone notebook has no external audit report; do not imply the earlier
# R=200 validation certificate covers a future run of this notebook.
summary = summary.replace('run_dir.parents[1] / "audit"', 'run_dir / "audit"')
summary = summary.replace('R=200 时 95% 附近的 MCSE 约 1.54 个百分点，0.93 与 0.95 不足以作确定区分；区间包含 0.95 不是等价证明。',
                          'R=1000 时 95% 附近的 MCSE 约 0.69 个百分点；应以实际区间和配对误差判断，区间包含 0.95 不是等价证明。')
summary = summary.replace('汇总时尚未找到 audit/validation.md，原基准一致性与方法验证状态不能由本报告确认；请先完成该项。',
                          '本 notebook 的原型脚本已做过原函数一致性与公式校验；notebook 交付时仅做静态检查、未执行实验，当前实际完成数以上表为准。此汇总不能冒称与作者 R/np 完成软件对照。')
code(assignments + '\n\n' + summary)

md('''## 5. 配置锁定与断点续跑

配置哈希包括 n、R、原种子映射、核规则、有效包版本、HGBR完整默认值及当前计算函数的源码。
配置/代码变化会开新目录；只补同配置缺失任务，不将 R=200 的旧结果混入。
同一结果目录不能同时开两个运行。中断后若留有 `.running.lock`，确认旧内核任务已停止，再按提示删除该锁文件并重跑。
''')

function_names = []
for cell in cells:
    if cell.cell_type == 'code':
        function_names += [n.name for n in ast.parse(cell.source).body if isinstance(n, ast.FunctionDef)]
# Report-only functions do not affect simulation configuration.
computational_names = [n for n in function_names if n not in {x.name for x in tree.body if isinstance(x,ast.FunctionDef)}]
source_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('_legacy_student.py','_legacy_lognormal.py','dukes_score.py','run_pilot.py')}
config_code = '''
def current_code_fingerprint(names):
    # Jupyter retains function sources in its input cache. Normalize syntax,
    # ignoring cell numbers/line positions so reruns keep the same identity.
    sources = {}
    for name in names:
        function = globals()[name]
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError) as error:
            raise RuntimeError(f"无法取得{name}的源码用于续跑校验；请在Jupyter中从头执行定义单元") from error
        sources[name] = ast.dump(ast.parse(source), include_attributes=False)
    sources["NumericalFailure.__init__"] = ast.dump(ast.parse(
        __import__("textwrap").dedent(inspect.getsource(NumericalFailure.__init__))), include_attributes=False)
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def make_config():
    config = {
        "design": "dukes_full_notebook_v1", "dgp": "DGP2_nonlinear", "legacy_dgp_id": 1,
        "p": P, "n": N, "theta0": THETA_0, "replications": R, "n_folds": N_FOLDS,
        "base_seed": BASE_SEED, "fold_seed_offset": 500000,
        "scenarios": [
            {"name":"gaussian", "error_family":"gaussian", "error_parameter":None, "legacy_error_id":7, "error_case":"N(0,1)"},
            {"name":"t2.1", "error_family":"student_t", "error_parameter":2.1, "legacy_error_id":0, "error_case":2.1},
            {"name":"lognormal2", "error_family":"lognormal", "error_parameter":2.0, "legacy_error_id":0, "error_case":2.0}],
        "kernel": DEFAULT_KERNEL_CONFIG.copy(),
        "hgbr_params_seed_zero": make_nuisance_model("mean",0).get_params(),
        "environment": {"python":sys.version, "numpy":np.__version__, "scipy":scipy.__version__,
                        "sklearn":sklearn.__version__, "joblib":joblib.__version__},
        "source_script_hashes_at_packaging": SOURCE_PROVENANCE,
        "computational_code_hash": current_code_fingerprint(COMPUTATIONAL_FUNCTIONS),
        "a_critical_value":1.96, "bc_critical_probability":0.975, "bc_variance_ddof":1,
        "reuse_previous_pilot":False,
    }
    config["config_hash"] = hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return config


def run_experiment(config, workers=4):
    if not 1 <= workers <= 16:
        raise ValueError("workers须在1至16之间，建议4")
    folder = OUTPUT_ROOT / (f"n{config['n']}_R{config['replications']}_" + config["config_hash"][:12])
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".running.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"已有运行锁：{lock}。请先确认旧任务已停止，再移除此文件。") from error
    handle.write(json.dumps({"pid":os.getpid(),"utc":datetime.now(timezone.utc).isoformat()}))
    handle.close()
    try:
        for name in ("tasks","arrays"):
            (folder/name).mkdir(exist_ok=True)
        config_path = folder / "config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if saved != config:
                raise RuntimeError("配置不一致，禁止混写")
        else:
            dump_json(config_path,config)
        missing = []
        for scenario in config["scenarios"]:
            for rep in range(config["replications"]):
                task_path = folder/"tasks"/f"{scenario['name']}_{rep:04d}.json"
                if not task_path.exists():
                    missing.append((scenario,rep))
                    continue
                saved = json.loads(task_path.read_text(encoding="utf-8"))
                rows = saved["rows"]
                if len(rows)!=3 or {r["method"] for r in rows}!=set(METHODS) or not all(
                    r["config_hash"]==config["config_hash"] and r["replication"]==rep and
                    r["scenario"]==scenario["name"] for r in rows):
                    raise RuntimeError(f"检查点主键不匹配：{task_path}")
                if sha256(folder/saved["arrays_file"])!=saved["arrays_sha256"]:
                    raise RuntimeError(f"数组哈希不匹配：{task_path}")
        print(f"n={config['n']}, R={config['replications']}, 场景={len(config['scenarios'])}; 待运行数据集={len(missing)}")
        print(f"输出：{folder}", flush=True)
        print(f"并行：{workers}个进程，每进程1计算线程", flush=True)
        started = time.perf_counter()
        with parallel_config(backend="loky",inner_max_num_threads=1):
            stream = Parallel(n_jobs=workers,return_as="generator_unordered")(
                delayed(persist_task)(config,scenario,rep,str(folder)) for scenario,rep in missing)
            for count,(stem,seconds,failures) in enumerate(stream,1):
                if count<=3 or count%20==0 or count==len(missing):
                    elapsed=time.perf_counter()-started
                    eta=(len(missing)-count)*elapsed/count
                    print(f"{count}/{len(missing)} | {stem} | 方法失败{failures} | 已用{elapsed/60:.1f}分钟 | 剩余约{eta/60:.1f}分钟",flush=True)
        event={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"workers":workers,
               "inner_threads":1,"datasets_computed":len(missing),"wall_seconds":time.perf_counter()-started}
        with (folder/"execution_log.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(event)+"\\n")
        return folder
    finally:
        lock.unlink(missing_ok=True)
'''
code('COMPUTATIONAL_FUNCTIONS = ' + repr(computational_names + ['make_config','run_experiment','current_code_fingerprint']) + '\nSOURCE_PROVENANCE = ' + repr(source_hashes) + '\n' + config_code)

md('''## 6. 开始正式实验

**运行下面的单元才会开始训练。** 直接按 n=1000、R=1000 跑，不先另跑 R=200 的 pilot。
结果逐份保存；中断后从头执行定义单元，再执行本单元即可续跑。保持算法配置不变。
''')
code('''CONFIG = make_config()
RUN_DIR = run_experiment(CONFIG, workers=WORKERS)
print("正式实验完成：", RUN_DIR)
''')

md('## 7. 生成与查看报告\n全部结束后运行；若中途手动汇总，pending会单独列出，不计为算法失败。')
code('''summary, paired = summarize_run(RUN_DIR)
display(pd.DataFrame(summary)[["scenario","method","planned","attempted","valid","failed","pending",
                               "coverage_valid","mcse","wilson_low","wilson_high"]])
display(pd.DataFrame(paired)[["scenario","comparison","valid_pairs","mean_difference","mcse",
                              "count_minus1","count_zero","count_plus1"]])
display(Image(filename=str(RUN_DIR/"coverage_comparison.png")))
display(Markdown((RUN_DIR/"decision_note.md").read_text(encoding="utf-8")))
''')
md('''## 解释边界

R=1000 时，coverage≈0.95 的 Monte Carlo 标准误约0.0069；这不能替代增大n来检查渐近趋势。
R从200增到1000不会改变同一 n=1000 数据集的统计量分布，只会更精确地测量它。
本轮仍是固定DGP、固定HGBR、固定核搜索规则的有限样本实验；带宽边界与作者R软件差异需要继续披露。
高覆盖率不能代替区间宽度、功效或信息量的检查。请优先比较B→C识别修正效果，同时保留A作为原基准。
''')

notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec':{'display_name':'Python (environment2)','language':'python','name':'environment2'},
    'language_info':{'name':'python','version':'3.11.15'},
    'delivery':{'execution':'not_executed','validation':'JSON schema and Python syntax only',
                'n':1000,'replications_per_scenario':1000,'self_contained':True}})
nbformat.validate(notebook)
for index,cell in enumerate(notebook.cells):
    if cell.cell_type=='code':
        compile(cell.source,f'notebook-cell-{index}','exec')
assert all(c.execution_count is None and not c.outputs for c in notebook.cells if c.cell_type=='code')
if DESTINATION.exists():
    raise FileExistsError(f'Refusing to overwrite {DESTINATION}')
nbformat.write(notebook, DESTINATION)
print(json.dumps({'path':str(DESTINATION),'cells':len(cells),'code_cells':sum(c.cell_type=='code' for c in cells),
                  'bytes':DESTINATION.stat().st_size,'executed':False},ensure_ascii=False))

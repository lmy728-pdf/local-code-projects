# -*- coding: utf-8 -*-
r"""Huber-DML: original Wald versus null-score test (standalone).

中文说明：一次重复只训练一套五折 nuisance，点估计、阈值完全共享。
保持原 DGP、full 模式种子、HGBR 超参数和原版 Huber/Wald 计算。
仅在内存保存重复结果；磁盘只生成桌面两张 PNG，不写日志或中间数据。
运行：D:\Anoconda\envs\environment2\python.exe Huber_Score_Test.py
来源：DML_HeavyTail_Benchmark/DML_p5_student_t.ipynb
"""
import os
# 每个并行进程只用一个计算线程，避免线程争抢。
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')
import math
import time
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

N = 1000
P = 5
R = 200
K = 5
BASE_SEED = 20261103
N_JOBS = 4
CRITICAL = 1.96
# 真值只用于数据生成（原 generate_observed）与下方评价，未传入估计接口。
THETA_NULL = 1.0
ERROR_IDS = (0, 7)  # 沿用全量编号：Gaussian=0, Student-t(2.1)=7。
DGP_LABELS = ('DGP1 线性', 'DGP2 非线性')
ERROR_LABELS = {0: 'Gaussian', 7: 't(2.1)'}
METHOD_LABELS = ('Huber-Wald', 'Huber Score-test')

# 以下常量与函数直接复制自原 Notebook；未执行该 Notebook。
SOURCE_NOTEBOOK_SHA256 = '41ffadee98ab10aeb76b99c68f1cd308129a09c04dde29a80305ee4f0d6f4f55'
HGBR_PARAMS = {'categorical_features': 'from_dtype', 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 1, 'learning_rate': 0.05, 'loss': 'squared_error', 'max_bins': 255, 'max_depth': None, 'max_features': 1.0, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'quantile': None, 'random_state': 0, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
SOLVER = {'min_denominator': 1e-10, 'm3_condition_limit': 100000000.0, 'score_tol': 1e-08, 'kappa': 1.345, 'm3_steps': 1, 'm3_path': 'm(theta)=m(anchor)-(theta-anchor)*g; G,M,alpha fixed; beta reprojected analytically', 'm3_se': 'frozen_path_centered_sandwich_ddof0'}
DGP_NAMES = ('DGP1_linear', 'DGP2_nonlinear')
T_GRID = (20.0, 10.0, 5.0, 3.0, 2.6, 2.3, 2.1)
def sha(value):
    return hashlib.sha256(value).hexdigest()

def ar1_covariance(p: int, rho: float = 0.5) -> np.ndarray:
    """构造 Sigma[j,k] = rho**|j-k| 的 AR(1) 型协方差矩阵。"""
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])

def nuisance_functions(X: np.ndarray, dgp_name: str):
    """返回指定 DGP 的 m0(X) 与 g0(X)。"""
    x1, x2, x3, x4, x5 = X.T

    if dgp_name == "DGP1_linear":
        m0 = 0.80 * x1 - 0.50 * x2 + 0.30 * x3 + 0.20 * x4 - 0.10 * x5
        g0 = 1.00 * x1 + 0.50 * x2 - 0.50 * x3 + 0.25 * x4 + 0.10 * x5
    elif dgp_name == "DGP2_nonlinear":
        # x2**2 - 1 做中心化，因为标准正态变量满足 E[X^2]=1
        m0 = (
            0.60 * np.sin(x1)
            + 0.35 * (x2**2 - 1.0)
            + 0.25 * x3 * x4
            - 0.20 * np.tanh(x5)
        )
        g0 = (
            1.20 * np.sin(x1)
            + 0.70 * (x2**2 - 1.0)
            + 0.50 * x3 * x4
            + 0.30 * np.tanh(x5)
        )
    else:
        raise ValueError(f"未知 DGP: {dgp_name}")
    return m0, g0

def standardized_student_t(rng: np.random.Generator, df: float, size: int) -> np.ndarray:
    """生成均值 0、方差 1 的 Student-t 误差；要求 df > 2。"""
    if df <= 2:
        raise ValueError("Student-t 自由度必须大于 2，才能标准化为单位方差。")
    return rng.standard_t(df, size=size) * np.sqrt((df - 2.0) / df)

class NumericalFailure(ValueError):
    def __init__(self, failure_type, message, fold=None):
        super().__init__(message)
        self.failure_type = failure_type
        self.fold = fold

def validate_folds(folds, n):
    """Validate full complementary train/test partitions with equal test sizes."""
    clean, seen, sizes = [], np.zeros(n, dtype=int), []
    for k, (train, test) in enumerate(folds):
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        if (train.ndim != 1 or test.ndim != 1 or len(test) < 2 or
                len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test) or
                np.any(train < 0) or np.any(train >= n) or np.any(test < 0) or np.any(test >= n) or
                np.intersect1d(train, test).size or
                not np.array_equal(np.sort(np.r_[train, test]), np.arange(n))):
            raise NumericalFailure("invalid_folds", "Each training set must be the test complement", k)
        seen[test] += 1
        sizes.append(len(test))
        clean.append((train, test))
    if not clean or np.any(seen != 1) or len(set(sizes)) != 1:
        raise NumericalFailure("invalid_folds", "Equal-sized test folds must partition observations")
    return clean

def seed(config, dgp, error, rep, purpose, fold=0, learner=0):
    key = [config['base_seed'], {'smoke': 1, 'full': 2}[config['mode']],
           config['p'], dgp, config['n'], error, rep, purpose, fold, learner]
    return int(np.random.SeedSequence(key).generate_state(1)[0])

def generate_observed(config, dgp, error, rep):
    n = config['n']
    seeds = {name: seed(config, dgp, error, rep, purpose)
             for name, purpose in [('X', 1), ('v', 2), ('epsilon', 3), ('folds', 4)]}
    X = np.random.default_rng(seeds['X']).multivariate_normal(
        np.zeros(5), ar1_covariance(5), n)
    m, g = nuisance_functions(X, DGP_NAMES[dgp])
    v = np.random.default_rng(seeds['v']).normal(size=n)
    rng = np.random.default_rng(seeds['epsilon'])
    epsilon = rng.normal(size=n) if error == 0 else standardized_student_t(rng, T_GRID[error-1], n)
    D = m + v
    Y = D + g + epsilon  # theta_true=1 is confined to generation/evaluation.
    assert X.shape == (n, 5) and np.isfinite(np.r_[X.ravel(), D, Y]).all()
    folds = list(KFold(5, shuffle=True, random_state=seeds['folds']).split(X))
    validate_folds(folds, n)
    assert n % 5 == 0
    fold_ids = np.empty(n, dtype=np.int64)
    for k, (_, test) in enumerate(folds):
        fold_ids[test] = k
    audit = dict(seeds=seeds, data_hash=sha(X.tobytes()+D.tobytes()+Y.tobytes()),
                 fold_hash=sha(fold_ids.tobytes()),
                 epsilon_sample_variance=float(np.var(epsilon, ddof=1)))
    return (X, D, Y), folds, fold_ids, audit

def learner(config, random_state, median=False):
    params = config['spec']['learner'].copy()
    # sklearn 1.7 requires a float here; JSON conversions can erase .0.
    params['max_features'] = float(params['max_features'])
    params.update(random_state=random_state, loss='quantile' if median else 'squared_error',
                  quantile=.5 if median else None)
    return HistGradientBoostingRegressor(**params)

def crossfit(observed, folds, config, dgp, error, rep, median=False, candidate=None):
    X, D, Y = observed
    predictions = np.empty((2, len(D)))
    seen = np.zeros(len(D), dtype=int)
    for k, (train, test) in enumerate(folds):
        assert not np.intersect1d(train, test).size
        for j, target in enumerate((D, Y if candidate is None else Y-candidate*D)):
            s = seed(config, dgp, error, rep, 5 if candidate is None else 6,
                     k, j+2*int(median))
            model = learner(config, s, median)
            predictions[j, test] = model.fit(X[train], target[train]).predict(X[test])
        seen[test] += 1
    assert np.all(seen == 1) and np.isfinite(predictions).all()
    return predictions

def ols_point(w, u):
    J = float(np.mean(w*w))
    if J <= SOLVER['min_denominator']:
        raise ArithmeticError('degenerate_OLS_denominator')
    return float(np.mean(w*u)/J)

def m4(w, u):
    initial = ols_point(w, u)
    r0 = u-initial*w
    s = float(1.4826*np.median(abs(r0-np.median(r0))))
    if not np.isfinite(s) or s <= 1e-12:
        raise ArithmeticError('degenerate_Huber_MAD')
    tau = SOLVER['kappa']*s
    def equation(t):
        return float(np.mean(w*np.clip(u-t*w, -tau, tau)))
    width = 1.
    for _ in range(40):
        if equation(initial-width) >= 0 >= equation(initial+width):
            break
        width *= 2
    else:
        raise ArithmeticError('Huber_bracket_failed')
    t = float(brentq(equation, initial-width, initial+width, xtol=1e-12, rtol=1e-14))
    r = u-t*w
    q = w*np.clip(r, -tau, tau)
    A = float(np.mean(w*w*(abs(r) < tau)))
    if abs(q.mean()) > 1e-9:
        raise ArithmeticError('Huber_score_residual')
    out = dict(theta_hat=t, se_type='Huber_plugin_ddof1_fixed_tau', ci_type='Huber_plugin_Wald',
               diagnostics=dict(scale=s, tau=tau, A=A, score_error=float(abs(q.mean())),
                                clipping=float(np.mean(abs(r)>tau)),
                                boundary_count=int(np.isclose(abs(r), tau, atol=1e-10, rtol=0).sum()),
                                objective=float(np.mean(np.where(abs(r)<=tau, .5*r*r, tau*abs(r)-.5*tau*tau)))))
    out['se'] = float(np.std(q/A, ddof=1)/np.sqrt(len(w))) if A > 1e-12 else None
    return out


CONFIG = dict(n=N, p=P, K=K, R=R, mode='full', base_seed=BASE_SEED,
              spec=dict(learner=HGBR_PARAMS.copy()))


def desktop_path():
    """读取 Windows 的实际桌面路径，不依赖当前工作目录。"""
    if os.name == 'nt':
        import winreg
        key = r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as registry:
            value, _ = winreg.QueryValueEx(registry, 'Desktop')
        return Path(os.path.expandvars(value)).resolve()
    return Path.home() / 'Desktop'


def one_replication(dgp, error, rep):
    """真值与估计分离：拟合及阈值仅依赖观测数据。"""
    with threadpool_limits(limits=1):
        observed, folds, _, _ = generate_observed(CONFIG, dgp, error, rep)
        predictions = crossfit(observed, folds, CONFIG, dgp, error, rep)
        w = observed[1] - predictions[0]
        u = observed[2] - predictions[1]
        # m4 原函数仅调用一次；两个推断方法共享此估计、OLS-MAD 阈值。
        out = m4(w, u)
        theta_hat, se = out['theta_hat'], out['se']
        tau = out['diagnostics']['tau']
        if se is None or not np.isfinite(se) or se <= 0:
            raise ArithmeticError(f'原 Wald SE 无效: dgp={dgp}, error={error}, rep={rep}')
        lower, upper = theta_hat - CRITICAL*se, theta_hat + CRITICAL*se
        # 评价从这里开始：在原假设处计算逐观测 score，而非在 theta_hat 处。
        score = w * np.clip(u - THETA_NULL*w, -tau, tau)
        score_sd = float(np.std(score, ddof=1))
        if not np.isfinite(score_sd) or score_sd <= 0:
            raise ArithmeticError(f'原假设 score SD 无效: dgp={dgp}, error={error}, rep={rep}')
        score_t = math.sqrt(len(w))*float(np.mean(score))/score_sd
        wald_t = (theta_hat - THETA_NULL)/se
        # 同一个点估计进入两个结果位置，不另拟合、不用真值选根或改阈值。
        theta_pair = (theta_hat, theta_hat)
        wald_covered = lower <= THETA_NULL <= upper
        if wald_covered != (abs(wald_t) <= CRITICAL):
            raise ArithmeticError('Wald 区间与双侧检验不一致')
        return dict(dgp=dgp, error=error, rep=rep, theta_pair=theta_pair,
                    wald_t=wald_t, score_t=score_t, wald_covered=wald_covered,
                    score_covered=abs(score_t) <= CRITICAL, ci_length=upper-lower)


def summarize(results):
    rows = []
    for dgp in range(2):
        for error in ERROR_IDS:
            group = sorted((x for x in results if x['dgp']==dgp and x['error']==error),
                           key=lambda x: x['rep'])
            if len(group) != R or [x['rep'] for x in group] != list(range(R)):
                raise RuntimeError('重复次数或编号不完整；不输出不完整比较')
            theta = np.asarray([x['theta_pair'] for x in group], dtype=float)
            np.testing.assert_array_equal(theta[:, 0], theta[:, 1])
            errors = theta - THETA_NULL
            biases = np.mean(errors, axis=0)
            rmses = np.sqrt(np.mean(errors**2, axis=0))
            np.testing.assert_array_equal(biases[:1], biases[1:])
            np.testing.assert_array_equal(rmses[:1], rmses[1:])
            for method in range(2):
                prefix = 'wald' if method==0 else 'score'
                ts = np.asarray([x[prefix+'_t'] for x in group])
                covered = np.asarray([x[prefix+'_covered'] for x in group])
                left, right = float(np.mean(ts < -CRITICAL)), float(np.mean(ts > CRITICAL))
                cov = float(np.mean(covered))
                if not np.isclose(cov+left+right, 1.0, atol=1e-14):
                    raise RuntimeError('覆盖率与尾部拒绝率之和不为 1')
                rows.append(dict(dgp=dgp, error=error, method=method, n_valid=len(group),
                                 coverage=cov, mean_t=float(np.mean(ts)),
                                 sd_t=float(np.std(ts, ddof=1)), left=left, right=right,
                                 bias=float(biases[method]), rmse=float(rmses[method]),
                                 ci_length=float(np.mean([x['ci_length'] for x in group]))
                                 if method==0 else None,
                                 coverage_mcse=math.sqrt(cov*(1-cov)/len(group))))
    for row in rows:
        base = next(x for x in rows if x['dgp']==row['dgp'] and x['method']==row['method']
                    and x['error']==0)
        row['deterioration_pp'] = 100*(base['coverage']-row['coverage'])
    return rows


def print_summary(rows):
    print('\n逐配置汇总（T 分别为 Wald T 与原假设 score T；左右尾按 T 的符号）：')
    print('DGP/误差             推断方式             N   Coverage  Mean(T)  SD(T)  左拒绝   右拒绝    Bias       RMSE     Wald CI长度')
    for q in rows:
        name = f"DGP{q['dgp']+1}/{ERROR_LABELS[q['error']]}"
        length = f"{q['ci_length']:.6f}" if q['ci_length'] is not None else '不适用'
        print(f"{name:<21} {METHOD_LABELS[q['method']]:<20} {q['n_valid']:3d} "
              f"{q['coverage']:8.1%} {q['mean_t']:8.4f} {q['sd_t']:7.4f} "
              f"{q['left']:7.1%} {q['right']:7.1%} {q['bias']:+.6f} {q['rmse']:.6f} {length}")
    print('\nGaussian → t(2.1) coverage deterioration（百分点；正值表示覆盖率下降）：')
    for dgp in range(2):
        heavy = [q for q in rows if q['dgp']==dgp and q['error']==7]
        print(f"{DGP_LABELS[dgp]}：" + '；'.join(
            f"{METHOD_LABELS[q['method']]} {q['deterioration_pp']:+.1f} pp" for q in heavy))
    print('\n核对通过：四个配置中，两种推断逐次 theta_hat 完全相同，Bias/RMSE 完全一致。')
    print('Score-test 未反演区间，因此其 CI length 为不适用。')
    print('R=200：覆盖率步长为 0.5 个百分点；单个 coverage 的 MCSE 随结果约为 1.5–3.5 个百分点。')


def configure_plot():
    available = {x.name for x in font_manager.fontManager.ttflist}
    for font in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'SimSun'):
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#f5f7fb', 'axes.facecolor': 'white',
                         'savefig.facecolor': '#f5f7fb'})


def add_table(ax, columns, data, widths=None):
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center',
                     colWidths=widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (i,j), cell in table.get_celld().items():
        cell.set_edgecolor('#d8dfeb')
        if i==0:
            cell.set_facecolor('#203552')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#ffffff' if i%2 else '#eaf0f8')
    return table


def table_png(rows, path):
    fig = plt.figure(figsize=(19, 12.8))
    fig.suptitle('Huber-DML：原版 Wald 与原假设 Score-test 对照', fontsize=23, weight='bold', y=.97)
    fig.text(.5, .925, 'n=1000 · p=5 · R=200/配置 · 5-fold cross-fitting · theta0=1 · 临界值 ±1.96',
             ha='center', fontsize=13)
    grid = fig.add_gridspec(3, 1, left=.035, right=.965, bottom=.14, top=.89,
                           height_ratios=[2.15, 1.3, 1], hspace=.33)
    ax=fig.add_subplot(grid[0]); ax.set_title('推断校准：Score T 在 theta0 处计算；Wald T=(theta_hat−theta0)/SE', loc='left', pad=13)
    data=[]
    for q in rows:
        data.append([DGP_LABELS[q['dgp']], ERROR_LABELS[q['error']], METHOD_LABELS[q['method']],
                     str(q['n_valid']), f"{100*q['coverage']:.1f}%", f"{q['mean_t']:+.3f}",
                     f"{q['sd_t']:.3f}", f"{100*q['left']:.1f}%", f"{100*q['right']:.1f}%",
                     f"{100*q['coverage_mcse']:.2f} pp"])
    add_table(ax, ['DGP','误差','推断','有效 R','Coverage','Mean(T)','SD(T)','左尾拒绝','右尾拒绝','Cov. MCSE'], data,
              [.105,.09,.17,.065,.095,.09,.08,.095,.095,.115])
    ax=fig.add_subplot(grid[1]); ax.set_title('共享 Huber 点估计与原版 Wald 区间长度', loc='left', pad=13)
    data=[]
    for q in rows:
        if q['method']==0:
            data.append([DGP_LABELS[q['dgp']], ERROR_LABELS[q['error']], f"{q['bias']:+.6f}",
                         f"{q['rmse']:.6f}", f"{q['ci_length']:.6f}", '逐次相同 / Bias、RMSE 一致'])
    add_table(ax, ['DGP','误差','共享 Bias','共享 RMSE','Wald 平均 CI length','Wald / Score 核对'],data,
              [.13,.12,.14,.14,.19,.28])
    ax=fig.add_subplot(grid[2]); ax.set_title('Gaussian → t(2.1)：Coverage deterioration（百分点）',loc='left',pad=13)
    data=[]
    for dgp in range(2):
        h=[next(q for q in rows if q['dgp']==dgp and q['error']==7 and q['method']==m) for m in range(2)]
        data.append([DGP_LABELS[dgp], f"{h[0]['deterioration_pp']:+.1f} pp",
                     f"{h[1]['deterioration_pp']:+.1f} pp"])
    add_table(ax,['DGP','Huber-Wald','Huber Score-test'],data,[.3,.35,.35])
    fig.text(.04,.085,'Score-test 使用 w_i·clip(u_i−theta0w_i, −τ, τ)，以样本标准差（ddof=1）标准化；不反演区间，无 CI length。',fontsize=11)
    fig.text(.04,.057,'两种推断共享数据、OOF 预测、OLS-MAD 阈值与同一个无截距 Huber 点估计；保留原始 Wald SE 和临界值。',fontsize=11)
    fig.text(.04,.029,'R=200 的结果用于本次小规模诊断；同一配置两种推断配对，Gaussian 与 t 使用原有独立种子。',fontsize=11,color='#52617a')
    fig.savefig(path,dpi=180)
    plt.close(fig)


def comparison_png(rows,path):
    colors=('#2b62a6','#d87832')
    fig,axes=plt.subplots(2,3,figsize=(17.5,10.7))
    fig.suptitle('Huber-DML：换用 Score-test 能否改善覆盖率？',fontsize=21,weight='bold',y=.975)
    fig.text(.5,.927,'相同点估计、相同第一阶段与阈值；n=1000，p=5，每配置 R=200',ha='center',fontsize=12)
    for dgp in range(2):
        ax=axes[0,dgp]; x=np.arange(2)
        for m in range(2):
            qs=[next(q for q in rows if q['dgp']==dgp and q['error']==e and q['method']==m) for e in ERROR_IDS]
            heights=[100*q['coverage'] for q in qs]
            bars=ax.bar(x+(m-.5)*.32, heights,.32,color=colors[m],label=METHOD_LABELS[m])
            ax.bar_label(bars,labels=[f'{h:.1f}%' for h in heights],padding=4,fontsize=10)
        ax.axhline(95,color='#bb4141',ls='--',lw=1.2,label='名义 95%')
        ax.set(xticks=x,xticklabels=['Gaussian','t(2.1)'],ylim=(0,108),ylabel='Coverage (%)',title=DGP_LABELS[dgp])
        ax.legend(fontsize=8,loc='lower left'); ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    ax=axes[0,2];x=np.arange(2)
    for m in range(2):
        values=[next(q['deterioration_pp'] for q in rows if q['dgp']==d and q['error']==7 and q['method']==m) for d in range(2)]
        bars=ax.bar(x+(m-.5)*.32,values,.32,color=colors[m],label=METHOD_LABELS[m])
        ax.bar_label(bars,labels=[f'{v:+.1f}' for v in values],padding=4)
    ax.axhline(0,color='#8994a6',lw=1);ax.set(xticks=x,xticklabels=DGP_LABELS,ylabel='下降幅度（百分点）',title='Gaussian → t(2.1) 覆盖率下降')
    ax.margins(y=.25);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    labels=['线性\nGaussian','线性\nt(2.1)','非线性\nGaussian','非线性\nt(2.1)'];x=np.arange(4)
    for j,metric in enumerate(('mean_t','sd_t')):
        ax=axes[1,j]
        for m in range(2):
            qs=[q for q in rows if q['method']==m]
            ax.plot(x,[q[metric] for q in qs],marker='o',color=colors[m],label=METHOD_LABELS[m],lw=2)
        ax.axhline(0 if metric=='mean_t' else 1,color='#bb4141',ls='--',lw=1.2)
        ax.set(xticks=x,xticklabels=labels,title='Mean(T)：参考值 0' if metric=='mean_t' else 'SD(T)：参考值 1')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.16)
    ax=axes[1,2]
    for m in range(2):
        qs=[q for q in rows if q['method']==m]
        ax.plot(x,[100*q['left'] for q in qs],marker='v',color=colors[m],ls='--',label=METHOD_LABELS[m]+' 左尾')
        ax.plot(x,[100*q['right'] for q in qs],marker='^',color=colors[m],label=METHOD_LABELS[m]+' 右尾')
    ax.axhline(2.5,color='#bb4141',ls=':',lw=1.2)
    ax.set(xticks=x,xticklabels=labels,ylabel='拒绝率 (%)',title='左右尾拒绝率：各 2.5% 为参考')
    ax.legend(fontsize=7);ax.grid(axis='y',alpha=.16)
    fig.subplots_adjust(left=.055,right=.98,bottom=.12,top=.865,wspace=.28,hspace=.33)
    fig.text(.055,.047,'T 分别表示 Wald T 与在 theta0 处计算的 score T。Score-test 不反演区间；共享点估计的 Bias/RMSE 及 Wald 区间长度见对照表。',fontsize=10)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    started=time.perf_counter()
    print(f'Huber-Wald / Score-test：2 DGP × 2 误差 × {R} 重复 = {4*R} 次数据任务。',flush=True)
    print(f'保留 full 随机种子，base_seed={BASE_SEED}；{N_JOBS} 个进程，每进程 1 线程。',flush=True)
    print('每任务只训练一次 5-fold nuisance（Y、D 各 5 次拟合），共享原版 Huber 点估计。',flush=True)
    tasks=((d,e,r) for d in range(2) for e in ERROR_IDS for r in range(R))
    # 所有逐次结果仅驻留内存，不产生 checkpoint、CSV、JSON 或额外数据文件。
    with parallel_config(backend='loky',inner_max_num_threads=1):
        stream=Parallel(n_jobs=N_JOBS,return_as='generator_unordered',pre_dispatch=N_JOBS,batch_size=1)(
            delayed(one_replication)(d,e,r) for d,e,r in tasks)
        results=list(tqdm(stream,total=4*R,desc='Huber 配对模拟',unit='次',mininterval=2))
    rows=summarize(results)
    print_summary(rows)
    configure_plot()
    output=desktop_path()
    table_path=output/'Huber_Score_Test_对照表.png'
    chart_path=output/'Huber_Score_Test_对照图.png'
    table_png(rows,table_path)
    comparison_png(rows,chart_path)
    print(f'\n对照表：{table_path}\n对照图：{chart_path}')
    print(f'总耗时：{(time.perf_counter()-started)/60:.2f} 分钟；失败 0/{len(results)}。')


if __name__=='__main__':
    main()

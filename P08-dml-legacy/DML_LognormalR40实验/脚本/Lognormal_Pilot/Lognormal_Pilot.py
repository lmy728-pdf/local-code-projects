# -*- coding: utf-8 -*-
"""五方法 Lognormal pilot，统计函数直接复制自主实验。
使用 environment2 Python 运行。结果、记录和图写在本脚本旁边。
"""
import ctypes
from ctypes import wintypes
import os
import sys
# 在加载任何科学计算库之前设置 Windows Low / Idle 优先级。
if os.name != 'nt':
    raise RuntimeError('本实验要求 Windows 低优先级运行')
_kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
_kernel32.GetCurrentProcess.restype=wintypes.HANDLE
_kernel32.SetPriorityClass.argtypes=(wintypes.HANDLE,wintypes.DWORD)
_kernel32.SetPriorityClass.restype=wintypes.BOOL
_kernel32.GetPriorityClass.argtypes=(wintypes.HANDLE,)
_kernel32.GetPriorityClass.restype=wintypes.DWORD
_handle=_kernel32.GetCurrentProcess()
if not _kernel32.SetPriorityClass(_handle,0x40) or _kernel32.GetPriorityClass(_handle)!=0x40:
    raise RuntimeError('未能确认 Low 优先级，停止 Pilot')
for _stream in (sys.stdout,sys.stderr):
    if hasattr(_stream,'reconfigure'):
        _stream.reconfigure(encoding='utf-8',errors='replace')


# 依赖安装由用户负责；本本不自动安装，也不会读取或执行旧 Notebook。
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
from pathlib import Path
import platform
import time
import traceback
import uuid
import warnings
from collections import Counter
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar, minimize
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


from threadpoolctl import threadpool_info
import ast
import copy
import html


# 只改 MODE 切换规模。修改后重启内核，从头运行，避免旧对象残留。
BASE_SEED = 20261103
P = 5
K = 5
CRITICAL = 1.96
THETA_TRUE = 1.0  # 仅生成/最终评价。
THETA_NULL = 1.0  # 仅原假设检验，绝不传入点估计或调参函数。
VERSION = 'p5_5methods_score_v3_fast_moment_20261105'
METHODS = ('M1','M2','M3','M4','M5')
METHOD_LABELS = {'M1':'Standard DML','M2':'Median-DML','M3':'Dukes (verified Python correction)',
                 'M4':'Huber-DML','M5':'改进 Huber-DML'}
DGP_NAMES = ('DGP1_linear','DGP2_nonlinear')
T_GRID = (20.,10.,5.,3.,2.6,2.3,2.1)
DESIGNS = {'smoke':{'n':1000,'R':3,'errors':[0,1,7]},
           'full':{'n':1000,'R':1000,'errors':list(range(8))}}
SOLVER = {'min_denominator':1e-10,'kappa':1.345}
# 仅 M3 点估计求根；不是置信区间搜索。每个候选都重新训练 transformed outcome。
M3_SOLVER = {'max_evaluations':80,'secant_steps':6,'initial_step':0.02,
 'bracket_width':0.05,'bracket_expansions':6,'max_brackets':3,'max_anchor_distance':2.,
 'bracket_maxiter':40,'interior_margin':0.05,'moment_T_power':0.25,'score_tol':1e-6}
REQUIRED_VERSIONS = {'numpy':'2.3.5','scipy':'1.17.1','scikit-learn':'1.7.2','joblib':'1.5.3'}
ALLOW_VERSION_MISMATCH = False
HGBR_PARAMS = {'categorical_features': 'from_dtype',
 'early_stopping': 'auto',
 'interaction_cst': None,
 'l2_regularization': 1,
 'learning_rate': 0.05,
 'loss': 'squared_error',
 'max_bins': 255,
 'max_depth': None,
 'max_features': 1.0,
 'max_iter': 100,
 'max_leaf_nodes': 15,
 'min_samples_leaf': 20,
 'monotonic_cst': None,
 'n_iter_no_change': 10,
 'quantile': None,
 'random_state': 0,
 'scoring': 'loss',
 'tol': 1e-07,
 'validation_fraction': 0.1,
 'verbose': 0,
 'warm_start': False}
DEFAULT_KERNEL_CONFIG = {
    "implementation": "python_gaussian_nw_loocv_v1",
    "kernel": "gaussian_order2", "regtype": "local_constant", "bwtype": "fixed",
    "cv": "leave_one_out_squared_error", "scale": "sample_sd_times_fold_n_power_minus_0.2",
    "lower_multiplier": 0.05, "upper_multiplier": 20.0, "grid_size": 41,
    "refine": True, "log_bandwidth_xatol": 1e-6, "refine_maxiter": 80,
    "constant_index": "mean_response", "final_include_self": True,
}
INNER_K=3
LAMBDA_RATIOS=np.logspace(0,-3,13)
ADAPTIVE_GAMMA=1.0
WEIGHT_EPS=1e-3
RIDGE_ALPHA=1.0
PILOT_HUBER_EPS=1.35
NUISANCE_HUBER_KAPPA=1.345
KKT_TOL=2e-6


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

def _finite_vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise NumericalFailure("nonfinite_or_nonvector", f"{name} must be a finite vector")
    return value

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

def _nw_from_distances(squared_distances, y, bandwidth, leave_one_out):
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise NumericalFailure("invalid_bandwidth", "Bandwidth must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        logs = -0.5 * (squared_distances / bandwidth / bandwidth)
        if leave_one_out:
            np.fill_diagonal(logs, -np.inf)
        row_max = np.max(logs, axis=1)
        if not np.all(np.isfinite(row_max)):
            raise NumericalFailure("kernel_denominator", "No finite kernel weights")
        weights = np.exp(logs - row_max[:, None])
        predictions = (weights @ y) / weights.sum(axis=1)
    return _finite_vector(predictions, "NW predictions")

def nw_predict(x, y, bandwidth, leave_one_out=False):
    """NW fit evaluated at training positions (LOO only when explicitly requested)."""
    x, y = _finite_vector(x, "index"), _finite_vector(y, "response")
    if len(x) != len(y) or len(x) < 2:
        raise NumericalFailure("invalid_kernel_input", "Kernel vectors must have equal length >=2")
    return _nw_from_distances((x[:, None] - x[None, :]) ** 2, y, bandwidth, leave_one_out)

def select_nw(x, y, kernel_config=None):
    """Deterministic finite-grid LOOCV followed by local log-bandwidth refinement."""
    cfg = dict(DEFAULT_KERNEL_CONFIG)
    if kernel_config is not None:
        unknown = set(kernel_config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown kernel settings: {unknown}")
        cfg.update(kernel_config)
    for key in ("implementation", "kernel", "regtype", "bwtype", "cv", "scale", "constant_index", "final_include_self"):
        if cfg[key] != DEFAULT_KERNEL_CONFIG[key]:
            raise ValueError(f"Unsupported kernel choice {key}={cfg[key]}")
    x, y = _finite_vector(x, "index"), _finite_vector(y, "response")
    n = len(x)
    if n != len(y) or n < 2:
        raise NumericalFailure("invalid_kernel_input", "Kernel vectors must have equal length >=2")
    if np.all(x == x[0]):
        pred = np.full(n, np.mean(y))
        loo = (np.sum(y) - y) / (n - 1)
        return pred, {"bandwidth": None, "cv_loss": float(np.mean((y - loo)**2)),
                      "boundary": False, "boundary_side": "", "constant_index": True,
                      "bandwidth_low": None, "bandwidth_high": None, "optimizer_success": True}
    scale = float(np.std(x, ddof=1) * n ** (-0.2))
    lo, hi, count = float(cfg["lower_multiplier"]), float(cfg["upper_multiplier"]), int(cfg["grid_size"])
    if not (0 < lo < hi) or count < 3 or not np.isfinite(scale) or scale <= 0:
        raise NumericalFailure("invalid_bandwidth_grid", "Invalid bandwidth scale or grid")
    logs = np.linspace(np.log(scale * lo), np.log(scale * hi), count)
    distances = (x[:, None] - x[None, :]) ** 2

    def objective(log_h):
        try:
            pred = _nw_from_distances(distances, y, np.exp(log_h), True)
            loss = float(np.mean((y - pred) ** 2))
            return loss if np.isfinite(loss) else np.inf
        except NumericalFailure:
            return np.inf

    losses = np.array([objective(t) for t in logs])
    if not np.any(np.isfinite(losses)):
        raise NumericalFailure("no_valid_bandwidth", "All LOOCV candidates failed")
    best = int(np.argmin(losses))  # exact ties choose smaller h, deterministic
    best_log, best_loss, success = logs[best], losses[best], True
    if cfg["refine"] and 0 < best < count - 1:
        opt = minimize_scalar(objective, bounds=(logs[best-1], logs[best+1]), method="bounded",
                              options={"xatol": cfg["log_bandwidth_xatol"], "maxiter": cfg["refine_maxiter"]})
        success = bool(opt.success)
        if opt.success and opt.fun < best_loss:
            best_log, best_loss = float(opt.x), float(opt.fun)
    h = float(np.exp(best_log))
    return _nw_from_distances(distances, y, h, False), {
        "bandwidth": h, "cv_loss": float(best_loss), "boundary": best in (0, count-1),
        "boundary_side": "lower" if best == 0 else "upper" if best == count-1 else "",
        "constant_index": False, "bandwidth_low": float(np.exp(logs[0])),
        "bandwidth_high": float(np.exp(logs[-1])), "optimizer_success": success,
    }

def stable_projection(direction, target):
    """No-intercept 1D projection, scaling before squaring to avoid underflow."""
    q, target = _finite_vector(direction, "direction"), _finite_vector(target, "target")
    if len(q) != len(target):
        raise NumericalFailure("invalid_projection", "Projection lengths differ")
    scale = float(np.max(np.abs(q)))
    if scale == 0:
        return np.zeros_like(q), {"coefficient": 0.0, "direction_sum_squares": 0.0,
                                  "direction_norm": 0.0, "update_norm": 0.0, "zero_direction": True}
    normalized = q / scale
    denominator = float(normalized @ normalized)
    amount = float((normalized @ target) / denominator)
    update = normalized * amount
    with np.errstate(over="ignore", under="ignore"):
        coefficient = float(np.divide(amount, scale))
        raw_denominator = float(np.dot(q, q))
    if not np.isfinite(coefficient):
        raise NumericalFailure("nonfinite_coefficient", "Projection coefficient cannot be represented")
    _finite_vector(update, "update")
    return update, {"coefficient": coefficient, "direction_sum_squares": raw_denominator,
                    "direction_norm": float(scale * np.sqrt(denominator)),
                    "update_norm": float(np.linalg.norm(update)), "zero_direction": False}

def corrected_score(a, b, G, M):
    """Full pointwise score; correction terms must remain in its variance."""
    a, b, G, M = [_finite_vector(x, name) for x, name in ((a,"a"),(b,"b"),(G,"G"),(M,"M"))]
    if len({len(a), len(b), len(G), len(M)}) != 1:
        raise NumericalFailure("invalid_input", "Score array lengths differ")
    update_D, alpha = stable_projection(G, a)
    update_Y, beta = stable_projection(M, b)
    rD, rY = a - update_D, b - update_Y
    scores = _finite_vector(rD * rY - G * rD - M * rY, "corrected score")
    diagnostics = {"alpha": alpha["coefficient"], "beta": beta["coefficient"],
                   "G_direction_sum_squares": alpha["direction_sum_squares"],
                   "M_direction_sum_squares": beta["direction_sum_squares"],
                   "G_direction_norm": alpha["direction_norm"], "M_direction_norm": beta["direction_norm"],
                   "D_update_norm": alpha["update_norm"], "Y_update_norm": beta["update_norm"],
                   "zero_G": alpha["zero_direction"], "zero_M": beta["zero_direction"],
                   "mean_G_rD": float(np.mean(G*rD)), "mean_M_rY": float(np.mean(M*rY)),
                   "full_score_variance": float(np.var(scores, ddof=1)),
                   "product_only_variance": float(np.var(rD*rY, ddof=1))}
    return {"scores": scores, "rD": rD, "rY": rY, "diagnostics": diagnostics}

def dukes_from_initial(D, Y, theta, g_hat, m_hat, folds, kernel_config=None):
    D, Y, g_hat, m_hat = [_finite_vector(x, name) for x, name in
                         ((D,"D"),(Y,"Y"),(g_hat,"g_hat"),(m_hat,"m_hat"))]
    if len({len(D), len(Y), len(g_hat), len(m_hat)}) != 1 or not np.isfinite(theta):
        raise NumericalFailure("invalid_input", "Input dimensions/theta invalid")
    folds = validate_folds(folds, len(D))
    a, b = D-g_hat, Y-theta*D-m_hat
    arrays = {key: np.empty(len(D)) for key in ("G", "M", "rD", "rY")}
    scores, diagnostics = np.empty(len(D)), []
    for k, (_, test) in enumerate(folds):
        try:
            G, diag_G = select_nw(g_hat[test], b[test], kernel_config)
            M, diag_M = select_nw(m_hat[test], a[test], kernel_config)
            out = corrected_score(a[test], b[test], G, M)
            scores[test] = out["scores"]
            for key, value in (("G",G),("M",M),("rD",out["rD"]),("rY",out["rY"])):
                arrays[key][test] = value
            diag = {"fold": k, "n_fold": len(test), **out["diagnostics"]}
            for suffix, part in (("G", diag_G), ("M", diag_M)):
                diag.update({f"{name}_{suffix}": value for name, value in part.items()})
            diagnostics.append(diag)
        except NumericalFailure as exc:
            exc.fold = k
            raise
    return {"scores": scores, "standard_scores": a*b,
            "fold_diagnostics": diagnostics, "arrays": {**arrays, "a": a, "b": b}}


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


def feature_dictionary(X):
    """固定共同字典：原始变量、全部平方、全部两两交互；不选择有效变量。"""
    if X.shape[1]!=P:
        raise ValueError('本实验只允许 p=5')
    values=[X,X**2]
    values.extend((X[:,j]*X[:,k])[:,None] for j in range(P) for k in range(j+1,P))
    result=np.column_stack(values)
    if result.shape!=(len(X),20) or not np.isfinite(result).all():
        raise ArithmeticError('字典非有限或维度错误')
    return result

def huber_loss(residual,tau):
    a=np.abs(residual)
    return np.where(a<=tau,.5*residual**2,tau*a-.5*tau*tau)

def prepare_training(F,y,robust):
    """每个内层/外层训练集单独拟合尺度、pilot 和自适应权重。"""
    scaler=StandardScaler().fit(F)
    Z=np.ascontiguousarray(scaler.transform(F))
    location=float(np.median(y) if robust else np.mean(y))
    scale=float(1.4826*np.median(np.abs(y-location)) if robust else np.std(y,ddof=0))
    if not np.isfinite(scale) or scale<=1e-10:
        raise ArithmeticError('training_target_scale_degenerate')
    target=(y-location)/scale
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if robust:
            # pilot 为成熟的 Huber+轻微 L2，用于训练数据驱动的初始权重。
            # 最终目标单独求解 Huber+加权 L1，不保留这个 L2 惩罚。
            pilot=HuberRegressor(epsilon=PILOT_HUBER_EPS,alpha=1e-4,
                                 fit_intercept=True,max_iter=3000,tol=1e-9).fit(Z,target)
        else:
            pilot=Ridge(alpha=RIDGE_ALPHA,fit_intercept=True,solver='cholesky').fit(Z,target)
    weights=1.0/(np.abs(pilot.coef_)+WEIGHT_EPS)**ADAPTIVE_GAMMA
    if not np.isfinite(weights).all() or np.any(weights<=0):
        raise ArithmeticError('adaptive_weights_invalid')
    state=dict(scaler=scaler,Z=Z,y=target,location=location,scale=scale,
               weights=weights,robust=robust,max_kkt=0.)
    if robust:
        pilot_res=target-pilot.predict(Z)
        residual_scale=float(1.4826*np.median(np.abs(pilot_res-np.median(pilot_res))))
        if residual_scale<=1e-10 or not np.isfinite(residual_scale):
            raise ArithmeticError('robust_pilot_MAD_degenerate')
        tau=NUISANCE_HUBER_KAPPA*residual_scale
        # 截距不正则化。lambda_max 对应只有稳健截距、系数全零的解。
        intercept=float(brentq(lambda c:np.mean(np.clip(target-c,-tau,tau)),
                              float(target.min()-tau),float(target.max()+tau),xtol=1e-12))
        derivative=np.clip(target-intercept,-tau,tau)
        state.update(tau=tau,warm=np.r_[intercept,np.zeros(2*Z.shape[1])])
    else:
        derivative=target-target.mean()
        state['lasso']=Lasso(fit_intercept=True,warm_start=True,selection='cyclic',
                              max_iter=100000,tol=1e-9,precompute=True)
        state['weighted_Z']=np.asfortranarray(Z/weights)
    lam_max=float(np.max(np.abs(Z.T@derivative)/len(y)/weights))
    if lam_max<=1e-14 or not np.isfinite(lam_max):
        raise ArithmeticError('lambda_max_degenerate')
    state['lambda_max']=lam_max
    return state

def fit_penalty(state,ratio):
    """精确加权 L1 目标；逐个候选都检查一阶最优性，不把失败改成 OLS。"""
    Z,y,w=state['Z'],state['y'],state['weights']
    lam=float(ratio*state['lambda_max'])
    n,q=Z.shape
    if not state['robust']:
        model=state['lasso']
        model.set_params(alpha=lam)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(state['weighted_Z'],y)
        # gamma=w*beta：普通 Lasso 的 L1 恰好等于原坐标的加权 L1。
        beta=model.coef_/w
        intercept=float(model.intercept_)
        residual=y-intercept-Z@beta
        grad=-Z.T@residual/n
        active=np.abs(beta)>1e-8
        violation=np.where(active,np.abs(grad+lam*w*np.sign(beta)),np.maximum(np.abs(grad)-lam*w,0.))
        kkt=float(max(abs(residual.mean()),violation.max()))
        if not np.isfinite(kkt) or kkt>KKT_TOL:
            raise ArithmeticError(f'adaptive_lasso_KKT={kkt:.3g}')
    else:
        tau=state['tau']
        def objective(z):
            beta=z[1:1+q]-z[1+q:]
            residual=y-z[0]-Z@beta
            psi=np.clip(residual,-tau,tau)
            gradient=-Z.T@psi/n
            value=float(np.mean(huber_loss(residual,tau))+lam*np.dot(w,z[1:1+q]+z[1+q:]))
            jac=np.r_[-psi.mean(),gradient+lam*w,-gradient+lam*w]
            return value,jac
        # beta=beta_plus-beta_minus, beta_plus,beta_minus>=0。
        # 这是凸的 Huber+加权 L1 目标，截距没有约束和惩罚。
        solved=minimize(objective,state['warm'],jac=True,method='L-BFGS-B',
                        bounds=[(None,None)]+[(0.,None)]*(2*q),
                        options=dict(maxiter=3000,maxls=50,gtol=1e-8,ftol=1e-14,maxcor=20))
        _,gradient=objective(solved.x)
        projected=gradient.copy()
        projected[1:]=np.where(solved.x[1:]<=1e-9,np.minimum(gradient[1:],0.),gradient[1:])
        kkt=float(np.max(np.abs(projected)))
        if not solved.success or not np.isfinite(kkt) or kkt>KKT_TOL:
            raise ArithmeticError(f'robust_adaptive_lasso_not_converged: {solved.message}; KKT={kkt:.3g}')
        state['warm']=solved.x.copy()
        beta=solved.x[1:1+q]-solved.x[1+q:]
        intercept=float(solved.x[0])
    state['max_kkt']=max(state['max_kkt'],kkt)
    if not np.isfinite(np.r_[intercept,beta]).all():
        raise ArithmeticError('nonfinite_first_stage_coefficients')
    return beta.copy(),intercept

def predict_state(state,fit,F):
    beta,intercept=fit
    return state['location']+state['scale']*(state['scaler'].transform(F)@beta+intercept)

def adaptive_prediction(Ftrain,ytrain,Ftest,inner_splits,robust=False):
    """内层选 lambda 比例，然后只在完整外层训练集重拟合选定的模型。"""
    losses=np.zeros(len(LAMBDA_RATIOS))
    max_kkt=0.
    for train,validation in inner_splits:
        if np.intersect1d(train,validation).size or len(train)+len(validation)!=len(Ftrain):
            raise ArithmeticError('inner_fold_leakage')
        state=prepare_training(Ftrain[train],ytrain[train],robust)
        for j,ratio in enumerate(LAMBDA_RATIOS):
            fitted=fit_penalty(state,ratio)
            residual=ytrain[validation]-predict_state(state,fitted,Ftrain[validation])
            # 稳健路线用训练集决定阈值的 Huber 验证损失；普通路线用 MSE。
            loss=huber_loss(residual,state['tau']*state['scale']) if robust else residual**2
            losses[j]+=float(np.sum(loss))
        max_kkt=max(max_kkt,state['max_kkt'])
    if not np.isfinite(losses).all():
        raise ArithmeticError('nonfinite_inner_cv_loss')
    best=int(np.argmin(losses))  # 平局取较大的 lambda，不按真值选模型。
    state=prepare_training(Ftrain,ytrain,robust)
    fitted=None
    for ratio in LAMBDA_RATIOS[:best+1]:
        fitted=fit_penalty(state,ratio)
    predictions=predict_state(state,fitted,Ftest)
    max_kkt=max(max_kkt,state['max_kkt'])
    if not np.isfinite(predictions).all():
        raise ArithmeticError('nonfinite_outer_predictions')
    return predictions,dict(ratio=float(LAMBDA_RATIOS[best]),lambda_value=float(state['lambda_max']*LAMBDA_RATIOS[best]),
                            max_kkt=max_kkt,grid_edge=best in (0,len(LAMBDA_RATIOS)-1))


def m5_crossfit(observed, folds, config, dgp, error, rep):
    """复用原稳健分支的 D-then-Y 顺序与全部数值核心；没有普通 Y 路线。"""
    X,D,Y=observed
    F=feature_dictionary(X)
    predictions=np.empty((2,len(D)))
    diagnostics=[]
    for target_id,target,robust in ((0,D,False),(1,Y,True)):
        seen=np.zeros(len(D),dtype=int)
        for k,(train,test) in enumerate(folds):
            if np.intersect1d(train,test).size:
                raise ArithmeticError('outer_fold_leakage')
            splits=list(KFold(INNER_K,shuffle=True,
                random_state=seed(config,dgp,error,rep,50,k,0)).split(train))
            try:
                predictions[target_id,test],diag=adaptive_prediction(
                    F[train],target[train],F[test],splits,robust)
            except Exception as exc:
                raise RuntimeError(f'outer_fold={k}; target={"D" if target_id==0 else "Y"}; '
                                   f'{type(exc).__name__}: {exc}') from exc
            diagnostics.append(dict(target=target_id,fold=k,**diag))
            seen[test]+=1
        if not np.all(seen==1):
            raise ArithmeticError('OOF_predictions_incomplete')
    if not np.isfinite(predictions).all():
        raise ArithmeticError('nonfinite_outer_predictions')
    return predictions,diagnostics


def lad_point(w,u):
    # 原无截距 LAD 求解与次梯度检查，删除此前仅供区间使用的密度/SE 部分。
    keep=abs(w)>1e-10
    if not np.any(keep):
        raise ArithmeticError('degenerate_LAD_design')
    values,weights=u[keep]/w[keep],abs(w[keep])
    order=np.argsort(values,kind='stable')
    theta=float(values[order[np.searchsorted(np.cumsum(weights[order]),weights.sum()/2,side='left')]])
    residual=u-theta*w
    zero=abs(residual)<=1e-9*np.maximum(1.,abs(u)+abs(theta*w))
    signed=float(np.sum(w[~zero]*np.sign(residual[~zero])))
    radius=float(abs(w[zero]).sum())
    if abs(signed)>radius+1e-7:
        raise ArithmeticError('LAD_subgradient_not_zero')
    return theta,dict(subgradient_signed=signed,subgradient_radius=radius,
                      tie_rule='lower_weighted_median')


def huber_threshold(w,u):
    initial=ols_point(w,u)
    residual=u-initial*w
    scale=float(1.4826*np.median(abs(residual-np.median(residual))))
    if not np.isfinite(scale) or scale<=1e-12:
        raise ArithmeticError('degenerate_Huber_MAD')
    return initial,SOLVER['kappa']*scale,scale


def huber_point(w,u,initial,tau):
    # 原 m4 的同一估计方程、括区规则、求解器与精度；只移除区间计算。
    def equation(theta):
        return float(np.mean(w*np.clip(u-theta*w,-tau,tau)))
    width=1.
    for _ in range(40):
        if equation(initial-width)>=0>=equation(initial+width):
            break
        width*=2
    else:
        raise ArithmeticError('Huber_bracket_failed')
    theta=float(brentq(equation,initial-width,initial+width,xtol=1e-12,rtol=1e-14))
    residual=u-theta*w
    score=w*np.clip(residual,-tau,tau)
    if abs(score.mean())>1e-9:
        raise ArithmeticError('Huber_score_residual')
    return theta,dict(score_error=float(abs(score.mean())),
                      clipping=float(np.mean(abs(residual)>tau)))


def standardize_score(scores,critical):
    scores=_finite_vector(scores,'scores')
    if len(scores)<2:
        raise NumericalFailure('insufficient_scores','At least two scores required')
    mean=float(np.mean(scores));sd=float(np.std(scores,ddof=1))
    if not np.isfinite(sd) or sd<=0:
        raise NumericalFailure('invalid_score_variance','Score SD must be positive')
    statistic=float(np.sqrt(len(scores))*mean/sd)
    if not np.isfinite(statistic):
        raise NumericalFailure('nonfinite_statistic','Score T must be finite')
    return dict(score_mean=mean,score_sd=sd,score_variance=sd*sd,T=statistic,
                score_sum=float(np.sum(scores)),score_centered_sum_squares=float(np.sum((scores-mean)**2)),
                p_value=float(2*norm.sf(abs(statistic))),
                accepted=bool(abs(statistic)<=critical),critical_value=float(critical),score_ddof=1)


def residual_scores(method,w,u,hypothesis,tau=None):
    residual=u-hypothesis*w
    if method=='M1':
        return w*residual
    if method=='M2':
        return .5*w*np.sign(residual)
    if method in ('M4','M5'):
        if tau is None or not np.isfinite(tau) or tau<=0:
            raise ArithmeticError('Huber_threshold_unavailable')
        return w*np.clip(residual,-tau,tau)
    raise ValueError('Residual score only applies to M1/M2/M4/M5')


def dukes_at_candidate(observed,folds,config,dgp,error,rep,g_hat,candidate):
    """给定任意候选，真正拟合 transformed outcome；不读取模拟真值。"""
    X,D,Y=observed
    m_hat=np.empty(len(D))
    response=Y-candidate*D
    for k,(train,test) in enumerate(folds):
        model=learner(config,seed(config,dgp,error,rep,6,k,1))
        m_hat[test]=model.fit(X[train],response[train]).predict(X[test])
    if not np.isfinite(m_hat).all():
        raise ArithmeticError('M3_nonfinite_transformed_outcome')
    out=dukes_from_initial(D,Y,candidate,g_hat,m_hat,folds,config['spec']['kernel'])
    return out,m_hat


class DukesPointObjective:
    """点估计专属缓存；不从检验步骤读取原假设、真值或接受/拒绝信息。"""
    def __init__(self,observed,folds,config,dgp,error,rep,g_hat):
        self.args=(observed,folds,config,dgp,error,rep,g_hat)
        self.options=config['spec']['m3_solver']
        self.cache={}
        self.trace=[]
        self.calls=0

    def evaluate(self,candidate):
        candidate=float(candidate)
        if not np.isfinite(candidate):
            raise ArithmeticError('M3_nonfinite_point_candidate')
        key=candidate.hex()
        if key not in self.cache:
            if self.calls>=self.options['max_evaluations']:
                raise ArithmeticError('M3_point_evaluation_budget')
            self.calls+=1
            try:
                out,_=dukes_at_candidate(*self.args,candidate)
            except Exception as exc:
                self.trace.append(dict(theta=candidate,error=f'{type(exc).__name__}: {exc}'))
                raise
            scores=out['scores']
            item=dict(theta=candidate,mean=float(np.mean(scores)),sd=float(np.std(scores,ddof=1)),
                      fold_diagnostics=out['fold_diagnostics'])
            if item['sd']<=0:
                raise ArithmeticError('M3_degenerate_point_score_variance')
            if not np.isfinite([item['mean'],item['sd']]).all():
                raise ArithmeticError('M3_nonfinite_point_score')
            self.cache[key]=item
            self.trace.append(dict(theta=candidate,score_mean=item['mean'],score_sd=item['sd']))
        return self.cache[key]

    def residual_T(self,item):
        return float(np.sqrt(len(self.args[0][1]))*item['mean']/item['sd'])

    def root_ok(self,item):
        # 按样本规模递减的 moment 残差停止，不额外追逐坐标区间精度。
        n=len(self.args[0][1])
        return abs(self.residual_T(item))<=n**(-self.options['moment_T_power'])

    def result(self,item,anchor):
        n=len(self.args[0][1])
        exact=abs(item['mean'])<=self.options['score_tol']*max(1.,item['sd'])
        kind='exact_score_root' if exact else 'approximate_score_root'
        return item['theta'],dict(anchor=anchor,root_theta=item['theta'],solution_type=kind,
            root_score_mean=item['mean'],root_score_sd=item['sd'],point_equation_T=self.residual_T(item),
            point_equation_T_limit=float(n**(-self.options['moment_T_power'])),
            evaluations=self.calls,candidate_trace=self.trace,fold_diagnostics=item['fold_diagnostics'],
            solver='adaptive_secant_Illinois_moment_stopping')








def dukes_solution_valid(theta,diag,n,options):
    """核对近似 Z 解证据，不使用原假设检验或 Coverage。"""
    try:
        mean,sd=diag['root_score_mean'],diag['root_score_sd']
        if not np.isfinite([theta,mean,sd]).all() or sd<=0:return False
        if theta!=diag['root_theta']:return False
        residual=float(np.sqrt(n)*mean/sd)
        limit=n**(-options['moment_T_power'])
        if abs(residual)>limit:return False
        if not np.isclose(residual,diag['point_equation_T'],rtol=1e-12,atol=1e-12):return False
        if not np.isclose(limit,diag['point_equation_T_limit'],rtol=1e-12,atol=1e-12):return False
        exact=abs(mean)<=options['score_tol']*max(1.,sd)
        expected='exact_score_root' if exact else 'approximate_score_root'
        return diag['solution_type']==expected
    except (KeyError,TypeError,ValueError):return False


def dukes_point(objective,anchor):
    """残差达标即停；必要时用有保护的 secant/Illinois 收缩变号段。"""
    options=objective.options
    previous=objective.evaluate(anchor)
    if objective.root_ok(previous):return objective.result(previous,anchor)
    current=objective.evaluate(anchor+options['initial_step'])
    if objective.root_ok(current):return objective.result(current,anchor)
    # 局部快速更新。每个新候选仍完整重拟合原 nuisance 和 correction。
    for _ in range(options['secant_steps']):
        difference=current['mean']-previous['mean']
        if abs(difference)<1e-14:break
        trial=current['theta']-current['mean']*(current['theta']-previous['theta'])/difference
        if not np.isfinite(trial) or abs(trial-anchor)>options['max_anchor_distance']:break
        if trial.hex() in objective.cache:break
        previous,current=current,objective.evaluate(trial)
        if objective.root_ok(current):return objective.result(current,anchor)
    # 只有残差未达标才寻找变号段；区间只服务点求解，不是区间反演。
    brackets=[]
    for level in range(options['bracket_expansions']+1):
        items=sorted(objective.cache.values(),key=lambda z:z['theta'])
        brackets=[(a,b) for a,b in zip(items[:-1],items[1:]) if a['mean']*b['mean']<0]
        if brackets:break
        width=options['bracket_width']*2**level
        for candidate in (anchor-width,anchor+width):
            item=objective.evaluate(candidate)
            if objective.root_ok(item):return objective.result(item,anchor)
    if not brackets:
        items=sorted(objective.cache.values(),key=lambda z:z['theta'])
        brackets=[(a,b) for a,b in zip(items[:-1],items[1:]) if a['mean']*b['mean']<0]
    brackets.sort(key=lambda ab:(abs((ab[0]['theta']+ab[1]['theta'])/2-anchor),ab[0]['theta']))
    for left,right in brackets[:options['max_brackets']]:
        # Illinois 权重仅用于提出下一候选；真实 score 从不被缩放或替换。
        weighted_left,weighted_right=left['mean'],right['mean']
        last_replaced=None
        for _ in range(options['bracket_maxiter']):
            width=right['theta']-left['theta']
            denominator=weighted_right-weighted_left
            trial=(left['theta']*weighted_right-right['theta']*weighted_left)/denominator if denominator else float('nan')
            margin=options['interior_margin']*width
            if not np.isfinite(trial) or not left['theta']+margin<trial<right['theta']-margin:
                trial=(left['theta']+right['theta'])/2
            if trial in (left['theta'],right['theta']):break
            item=objective.evaluate(trial)
            if objective.root_ok(item):return objective.result(item,anchor)
            if left['mean']*item['mean']<0:
                right=item;weighted_right=item['mean']
                weighted_left=weighted_left*.5 if last_replaced=='right' else left['mean']
                last_replaced='right'
            else:
                left=item;weighted_left=item['mean']
                weighted_right=weighted_right*.5 if last_replaced=='left' else right['mean']
                last_replaced='left'
    closest=min(objective.cache.values(),key=lambda z:abs(objective.residual_T(z)))
    raise ArithmeticError(f'M3_point_moment_unresolved: calls={objective.calls}; '
                         f'best_standardized_residual={objective.residual_T(closest):.6g}')


def nuisance_diagnostics(observed,predictions,dgp):
    # 只在拟合之后评价；返回值不回流到模型、lambda、阈值或点估计。
    X,D,Y=observed
    m_true,g_true=nuisance_functions(X,DGP_NAMES[dgp])
    ell_true=THETA_TRUE*m_true+g_true
    return dict(y_nuisance_mse=float(np.mean((predictions[1]-ell_true)**2)),
                d_nuisance_mse=float(np.mean((predictions[0]-m_true)**2)),
                y_observed_mse=float(np.mean((predictions[1]-Y)**2)),
                d_observed_mse=float(np.mean((predictions[0]-D)**2)))


def get_predictions(method,observed,folds,config,dgp,error,rep,cache):
    target='median' if method=='M2' else 'adaptive' if method=='M5' else 'mean'
    if target not in cache:
        if target=='adaptive':
            predictions,diagnostics=m5_crossfit(observed,folds,config,dgp,error,rep)
        else:
            predictions=crossfit(observed,folds,config,dgp,error,rep,median=target=='median')
            diagnostics=[]
        cache[target]=(predictions,diagnostics)
    return cache[target]


def failure_details(exc,stage):
    return dict(stage=stage,type=getattr(exc,'failure_type',type(exc).__name__),
                fold=getattr(exc,'fold',None),message=str(exc),traceback=traceback.format_exc())


def estimate_record(method,observed,folds,config,dgp,error,rep,cache):
    record=dict(method=method,point_status='failed',test_status='failed',nuisance_status='failed',
                theta_hat=None,T=None,accepted=None,failures=[],warnings=[],diagnostics={})
    total_start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            start=time.perf_counter()
            predictions,diagnostics=get_predictions(method,observed,folds,config,dgp,error,rep,cache)
            record['nuisance_seconds']=time.perf_counter()-start
            record['nuisance_status']='ok'
            record['prediction_hash']=sha(predictions.tobytes())
            record['diagnostics']['first_stage']=diagnostics
        except Exception as exc:
            record['failures'].append(failure_details(exc,'first_stage'))
            record.update(point_seconds=0.,test_seconds=0.)
        else:
            w,u=observed[1]-predictions[0],observed[2]-predictions[1]
            tau=initial=scale=None
            threshold_error=None
            objective=None
            if method in ('M4','M5'):
                try:
                    initial,tau,scale=huber_threshold(w,u)
                    record['diagnostics'].update(tau=tau,scale=scale,OLS_initial=initial)
                except Exception as exc:
                    threshold_error=exc
            start=time.perf_counter()
            try:
                if method=='M1':
                    theta=ols_point(w,u)
                    score_error=float(abs(np.mean(w*(u-theta*w))))
                    if score_error>1e-9:raise ArithmeticError('OLS_score_residual')
                    point_diag=dict(score_error=score_error)
                elif method=='M2':
                    theta,point_diag=lad_point(w,u)
                elif method in ('M4','M5'):
                    if threshold_error is not None:raise threshold_error
                    theta,point_diag=huber_point(w,u,initial,tau)
                elif method=='M3':
                    # 点估计接口不接收原假设或真值。检验尚未计算。
                    anchor=ols_point(w,u)
                    objective=DukesPointObjective(observed,folds,config,dgp,error,rep,predictions[0])
                    theta,point_diag=dukes_point(objective,anchor)
                else:raise ValueError('Unknown method')
                if not np.isfinite(theta):raise ArithmeticError('nonfinite_point_estimate')
                record.update(theta_hat=float(theta),point_status='ok')
                record['diagnostics']['point']=point_diag
                record['point_solution_type']=point_diag.get('solution_type','original_estimator')
            except Exception as exc:
                record['failures'].append(failure_details(exc,'point'))
                if objective is not None:
                    record['diagnostics']['point_candidate_trace']=objective.trace
            record['point_seconds']=time.perf_counter()-start
            # 此步骤与点估计成功与否无关，绝不把点估计处接近零的 score 当作检验。
            start=time.perf_counter()
            try:
                hypothesis=config['theta_null']
                if method=='M3':
                    out,m_hat=dukes_at_candidate(observed,folds,config,dgp,error,rep,predictions[0],hypothesis)
                    arrays=out['arrays']
                    alternate=arrays['rD']*arrays['rY']-arrays['G']*arrays['rD']-arrays['M']*arrays['rY']
                    if not np.allclose(out['scores'],alternate,rtol=1e-12,atol=1e-12):
                        raise ArithmeticError('M3_full_score_identity')
                    scores=out['scores']
                    record['diagnostics']['test_folds']=out['fold_diagnostics']
                    record['anchor_prediction_hash']=record['prediction_hash']
                    record['prediction_hash']=sha(predictions[0].tobytes()+m_hat.tobytes())
                    record['diagnostics']['full_score_identity_checked']=True
                    # 恢复对 E[Y|X] 的预测仅作诊断，不用于任何 M3 训练或选根。
                    evaluation_predictions=np.stack((predictions[0],m_hat+hypothesis*predictions[0]))
                else:
                    if threshold_error is not None:raise threshold_error
                    scores=residual_scores(method,w,u,hypothesis,tau)
                    evaluation_predictions=predictions
                record.update(standardize_score(scores,config['critical']))
                record.update(test_status='ok',theta_tested=hypothesis)
                record['diagnostics']['test_score_variance_checked']=bool(
                    np.isclose(record['score_variance'],np.var(scores,ddof=1),rtol=1e-12,atol=1e-12))
                record.update(nuisance_diagnostics(observed,evaluation_predictions,dgp))
            except Exception as exc:
                record['failures'].append(failure_details(exc,'test'))
            record['test_seconds']=time.perf_counter()-start
        record['warnings']=[f'{x.category.__name__}: {x.message}' for x in caught]
    # 纯评价在点估计和检验结束之后；不会改变其输入或候选。
    if record['point_status']=='ok':record['estimation_error']=record['theta_hat']-THETA_TRUE
    record['status']='ok' if record['point_status']==record['test_status']=='ok' else 'partial' if (
        record['point_status']=='ok' or record['test_status']=='ok') else 'failed'
    record['elapsed_seconds']=time.perf_counter()-total_start
    return record


def digest_object(obj):
    return sha(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8'))

def clean_json(value):
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(x) for x in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist())
    if isinstance(value,np.generic):return clean_json(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(clean_json(obj),ensure_ascii=False,indent=1,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)

def code_payload(value):
    # 忽略 Jupyter 临时单元文件名/行号；保留真实字节码、常量与嵌套函数。
    if inspect.iscode(value):
        return dict(code=value.co_code.hex(),constants=[code_payload(x) for x in value.co_consts],
                    names=list(value.co_names),varnames=list(value.co_varnames),
                    freevars=list(value.co_freevars),cellvars=list(value.co_cellvars),
                    args=value.co_argcount,posonly=value.co_posonlyargcount,
                    kwonly=value.co_kwonlyargcount,flags=value.co_flags,
                    exception_table=value.co_exceptiontable.hex())
    if isinstance(value,bytes):return dict(bytes=value.hex())
    if isinstance(value,(tuple,list)):return [code_payload(x) for x in value]
    if isinstance(value,dict):return {str(k):code_payload(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return repr(value)

def implementation_fingerprint():
    result={}
    for name in STATISTICAL_DEFINITIONS:
        obj=globals()[name]
        functions={name:obj} if inspect.isfunction(obj) else {
            name+'.'+key:val for key,val in vars(obj).items() if inspect.isfunction(val)}
        for key,fun in functions.items():
            result[key]=dict(body=code_payload(fun.__code__),defaults=code_payload(fun.__defaults__),
                             kwdefaults=code_payload(fun.__kwdefaults__))
    return digest_object(result)

def sample_mcse(values):
    values=np.asarray(values,dtype=float)
    return float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else np.nan

def summarize_records(records,config):
    frame=pd.DataFrame(records)
    rows=[]
    for d in range(2):
        for error in config['errors']:
            for method in METHODS:
                if frame.empty:group=pd.DataFrame()
                else:group=frame[(frame.dgp_id==d)&(frame.error_id==error)&(frame.method==method)]
                points=group[group.point_status=='ok'] if len(group) else group
                tests=group[group.test_status=='ok'] if len(group) else group
                both=group[(group.point_status=='ok')&(group.test_status=='ok')] if len(group) else group
                errors=np.array([r-THETA_TRUE for r in points.theta_hat],dtype=float) if len(points) else np.array([])
                ts=tests['T'].astype(float).to_numpy() if len(tests) else np.array([])
                covered=tests.accepted.astype(float).to_numpy() if len(tests) else np.array([])
                n,nt=len(points),len(tests)
                coverage=float(covered.mean()) if nt else np.nan
                rmse=float(np.sqrt(np.mean(errors**2))) if n else np.nan
                ncovered=int(covered.sum()) if nt else 0
                row=dict(mode=config['mode'],p=config['p'],n=config['n'],K=config['K'],dgp=DGP_NAMES[d],
                    dgp_id=d,error_id=error,error_label=ERROR_LABELS[error],
                    method=method,method_label=METHOD_LABELS[method],estimator_signature=config['method_signature'],
                    R_planned=config['R'],R_attempted=len(group),N_point_valid=n,N_test_valid=nt,
                    N_point_test_common=len(both),N_covered=ncovered,
                    joint_success_rate=len(both)/len(group) if len(group) else np.nan,
                    completion_rate=len(group)/config['R'],point_success_rate=n/len(group) if len(group) else np.nan,
                    test_success_rate=nt/len(group) if len(group) else np.nan,
                    point_failure_rate=(len(group)-n)/len(group) if len(group) else np.nan,
                    test_failure_rate=(len(group)-nt)/len(group) if len(group) else np.nan,
                    bias=float(errors.mean()) if n else np.nan,rmse=rmse,
                    empirical_sd=float(np.std(errors,ddof=1)) if n>1 else np.nan,
                    bias_mcse=sample_mcse(errors),rmse_mcse=sample_mcse(errors**2)/(2*rmse) if n>1 and rmse>0 else np.nan,
                    coverage=coverage,coverage_mcse=math.sqrt(coverage*(1-coverage)/nt) if nt else np.nan,
                    coverage_missing_lower=ncovered/config['R'],
                    coverage_missing_upper=(ncovered+config['R']-nt)/config['R'],
                    mean_T=float(np.mean(ts)) if nt else np.nan,sd_T=float(np.std(ts,ddof=1)) if nt>1 else np.nan,
                    left_reject_rate=float(np.mean(ts<-config['critical'])) if nt else np.nan,
                    right_reject_rate=float(np.mean(ts>config['critical'])) if nt else np.nan,
                    mean_score_sd=float(tests.score_sd.astype(float).mean()) if nt else np.nan,
                    quality_status='PENDING_VALIDATION')
                for key in ('y_nuisance_mse','d_nuisance_mse','y_observed_mse','d_observed_mse',
                            'nuisance_seconds','point_seconds','test_seconds','elapsed_seconds'):
                    values=pd.to_numeric(group[key],errors='coerce') if len(group) and key in group else pd.Series(dtype=float)
                    row['mean_'+key]=float(values.mean()) if values.notna().any() else np.nan
                    if key in ('y_nuisance_mse','d_nuisance_mse'):row['N_'+key]=int(values.notna().sum())
                rows.append(row)
    summary=pd.DataFrame(rows)
    for idx,row in summary.iterrows():
        baseline=summary[(summary.dgp_id==row.dgp_id)&(summary.error_id==0)&(summary.method==row.method)].iloc[0]
        summary.loc[idx,'coverage_deterioration_pp']=100*(baseline.coverage-row.coverage)
        summary.loc[idx,'deterioration_mcse_pp']=0. if row.error_id==0 else 100*np.sqrt(
            baseline.coverage_mcse**2+row.coverage_mcse**2)
        summary.loc[idx,'rmse_ratio']=row.rmse/baseline.rmse if baseline.rmse>0 else np.nan
    return frame,summary

def paired_comparisons(frame,config):
    rows=[]
    if frame.empty:return pd.DataFrame()
    for d in range(2):
        for error in config['errors']:
            group=frame[(frame.dgp_id==d)&(frame.error_id==error)]
            for i,a in enumerate(METHODS):
                for b in METHODS[i+1:]:
                    aa=group[group.method==a].set_index('replication_id')
                    bb=group[group.method==b].set_index('replication_id')
                    ids=sorted(set(aa.index)&set(bb.index))
                    pid=[r for r in ids if aa.loc[r,'point_status']==bb.loc[r,'point_status']=='ok']
                    tid=[r for r in ids if aa.loc[r,'test_status']==bb.loc[r,'test_status']=='ok']
                    row=dict(dgp=DGP_NAMES[d],dgp_id=d,error_id=error,A=a,B=b,N_point_pair=len(pid),N_test_pair=len(tid))
                    if tid:
                        diff=np.array([float(aa.loc[r,'accepted'])-float(bb.loc[r,'accepted']) for r in tid])
                        row.update(coverage_diff_A_minus_B_pp=100*float(diff.mean()),
                                   coverage_diff_mcse_pp=100*sample_mcse(diff))
                    if pid:
                        ea=aa.loc[pid,'theta_hat'].to_numpy(dtype=float)-THETA_TRUE
                        eb=bb.loc[pid,'theta_hat'].to_numpy(dtype=float)-THETA_TRUE
                        ra=float(np.sqrt(np.mean(ea**2)));rb=float(np.sqrt(np.mean(eb**2)))
                        influence=(ea**2-ra**2)/(2*ra)-(eb**2-rb**2)/(2*rb) if ra>0 and rb>0 else np.full(len(pid),np.nan)
                        row.update(bias_diff_A_minus_B=float(np.mean(ea-eb)),bias_diff_mcse=sample_mcse(ea-eb),
                                   rmse_diff_A_minus_B=ra-rb,rmse_diff_mcse=sample_mcse(influence),
                                   rmse_ratio_A_over_B=ra/rb if rb>0 else np.nan)
                    rows.append(row)
    paired=pd.DataFrame(rows)
    if 'coverage_diff_A_minus_B_pp' in paired:
        for idx,row in paired.iterrows():
            base=paired[(paired.dgp_id==row.dgp_id)&(paired.error_id==0)&(paired.A==row.A)&(paired.B==row.B)]
            if len(base):
                base=base.iloc[0]
                # Delta_B - Delta_A = (Cov_A-Cov_B)_error - (Cov_A-Cov_B)_Gaussian。
                paired.loc[idx,'deterioration_advantage_A_pp']=row.get('coverage_diff_A_minus_B_pp',np.nan)-base.get('coverage_diff_A_minus_B_pp',np.nan)
                paired.loc[idx,'deterioration_advantage_mcse_pp']=0. if row.error_id==0 else np.hypot(
                    row.get('coverage_diff_mcse_pp',np.nan),base.get('coverage_diff_mcse_pp',np.nan))
    return paired

def validate_run(records,packets,summary,config):
    planned=2*len(config['errors'])*config['R']*len(METHODS)
    keys=[(r['dgp_id'],r['error_id'],r['replication_id'],r['method']) for r in records]
    expected={(d,e,r,m) for d in range(2) for e in config['errors'] for r in range(config['R']) for m in METHODS}
    checks=dict(unique_keys=len(keys)==len(set(keys)),only_planned_keys=set(keys)<=expected,
                correct_dimensions=all(r['n']==config['n'] and r['p']==5 and r['K']==5 for r in records),
                correct_signature=all(r['config_hash']==config['config_hash'] for r in records),
                exactly_five_methods=tuple(METHODS)==('M1','M2','M3','M4','M5'))
    shared=True;shared_mean=True
    for packet in packets:
        results=packet['results']
        shared &= len({x['data_hash'] for x in results.values()})<=1 and len({x['fold_hash'] for x in results.values()})<=1
        a,b=results.get('M1'),results.get('M4')
        if a and b and a['nuisance_status']==b['nuisance_status']=='ok':
            shared_mean &= a.get('prediction_hash')==b.get('prediction_hash')
    checks.update(shared_data_and_folds=shared,shared_M1_M4_predictions=shared_mean)
    tests=[r for r in records if r['test_status']=='ok']
    points=[r for r in records if r['point_status']=='ok']
    checks['score_ddof_1']=all(r['score_ddof']==1 for r in tests)
    checks['null_only_tested']=all(r['theta_tested']==config['theta_null'] for r in tests)
    checks['test_definition']=all(r['accepted']==(abs(r['T'])<=config['critical']) for r in tests)
    checks['finite_score_statistics']=all(np.isfinite([r['T'],r['score_mean'],r['score_sd']]).all() and r['score_sd']>0 for r in tests)
    checks['variance_checks']=all(r['diagnostics'].get('test_score_variance_checked',False) for r in tests)
    checks['T_from_score_moments']=all(np.isclose(r['T'],r['score_sum']/np.sqrt(
        r['n']*r['score_centered_sum_squares']/(r['n']-1)),rtol=1e-10,atol=1e-10) for r in tests)
    checks['finite_points']=all(np.isfinite(r['theta_hat']) for r in points)
    checks['huber_fixed_tau']=all(np.isclose(r['diagnostics']['tau'],SOLVER['kappa']*r['diagnostics']['scale'],
        rtol=1e-14,atol=1e-14) for r in records if r['method'] in ('M4','M5') and 'tau' in r['diagnostics'])
    checks['point_equation_checks']=all(r['diagnostics']['point'].get('score_error',0.)<=1e-9 for r in points if r['method'] in ('M1','M4','M5'))
    checks['lad_subgradient_checks']=all(abs(r['diagnostics']['point']['subgradient_signed'])<=
        r['diagnostics']['point']['subgradient_radius']+1e-7 for r in points if r['method']=='M2')
    checks['M3_full_corrected_score']=all(r['diagnostics'].get('full_score_identity_checked',False) for r in tests if r['method']=='M3')
    checks['M3_fold_calibration']=all(
        abs(diag['mean_G_rD'])<=1e-8*max(1.,diag['G_direction_norm']) and
        abs(diag['mean_M_rY'])<=1e-8*max(1.,diag['M_direction_norm'])
        for r in tests if r['method']=='M3' for diag in r['diagnostics']['test_folds'])
    checks['M3_point_solution_evidence']=all(dukes_solution_valid(r['theta_hat'],r['diagnostics']['point'],
        r['n'],config['spec']['m3_solver']) for r in points if r['method']=='M3')
    checks['M5_KKT']=all(diag['max_kkt']<=KKT_TOL for r in records if r['method']=='M5' and r['nuisance_status']=='ok'
                        for diag in r['diagnostics']['first_stage'])
    checks['M5_outer_models']=all(len(r['diagnostics']['first_stage'])==10 for r in records
                                if r['method']=='M5' and r['nuisance_status']=='ok')
    checks['coverage_tail_identity']=all(np.isclose(row.coverage+row.left_reject_rate+row.right_reject_rate,1.)
        for _,row in summary.iterrows() if row.N_test_valid>0)
    complete=set(keys)==expected
    point_failures=sum(r['point_status']!='ok' for r in records)
    test_failures=sum(r['test_status']!='ok' for r in records)
    if not all(checks.values()):status='VALIDATION_FAILED'
    elif not complete:status='INCOMPLETE'
    elif config['mode']=='smoke':status='SMOKE_PASSED' if point_failures==test_failures==0 else 'SMOKE_FAILED'
    else:status='FULL_COMPLETE_REVIEW' if point_failures==test_failures==0 else 'FULL_COMPLETE_WITH_FAILURES_REVIEW'
    return dict(status=status,checks=checks,complete=complete,planned_records=planned,actual_records=len(records),
        N_point_valid=len(points),N_test_valid=len(tests),point_failures=point_failures,test_failures=test_failures,
        method_signature=config['method_signature'],config_hash=config['config_hash'],dependencies=config['dependencies'],
        method_counts=dict(Counter(r['method'] for r in records)),score_ddof=1,
        M3_point_solution_counts=dict(Counter(r.get('point_solution_type','failed') for r in records if r['method']=='M3')),
        coverage_definition='Non-rejection of the true null; no intervals constructed',
        execution_claim='Generated by standalone Lognormal pilot',
        updated_utc=datetime.now(timezone.utc).isoformat())

def configure_plot_style():
    fonts={font.name for font in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun'):
        if name in fonts:
            plt.rcParams['font.sans-serif']=[name,'DejaVu Sans'];break
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'#f6f8fc','axes.facecolor':'white'})

STATISTICAL_DEFINITIONS = (
    'ar1_covariance','nuisance_functions','standardized_student_t','seed','generate_observed',
    'learner','crossfit','ols_point','NumericalFailure','_finite_vector','validate_folds',
    '_nw_from_distances','nw_predict','select_nw','stable_projection','corrected_score','dukes_from_initial',
    'feature_dictionary','huber_loss','prepare_training','fit_penalty','predict_state','adaptive_prediction',
    'm5_crossfit','lad_point','huber_threshold','huber_point','standardize_score','residual_scores',
    'dukes_at_candidate','DukesPointObjective','dukes_point','dukes_solution_valid','nuisance_diagnostics','get_predictions','estimate_record')

SOURCE_CONFIG = {'mode': 'full', 'n': 1000, 'R': 1000, 'errors': [0, 1, 2, 3, 4, 5, 6, 7], 'p': 5, 'K': 5, 'base_seed': 20261103, 'theta_true': 1.0, 'theta_null': 1.0, 'critical': 1.96, 'dependencies': {'numpy': '2.3.5', 'scipy': '1.17.1', 'scikit-learn': '1.7.2', 'joblib': '1.5.3', 'pandas': '2.3.3', 'matplotlib': '3.10.9', 'threadpoolctl': '3.6.0', 'tqdm': '4.67.3'}, 'python': '3.11.15', 'method_signature': 'fb0022c542dbb2c8c191cebd1848e0ecc2e37a2a0a422469a75700499074c6be', 'spec': {'version': 'p5_5methods_score_v3_fast_moment_20261105', 'code_hash': '0b9e5fcc8d54b917a93ba43f746d574db9f3f9013806a3cf314627a134dc5c72', 'designs': {'smoke': {'n': 1000, 'R': 3, 'errors': [0, 1, 7]}, 'full': {'n': 1000, 'R': 1000, 'errors': [0, 1, 2, 3, 4, 5, 6, 7]}}, 'methods': ['M1', 'M2', 'M3', 'M4', 'M5'], 'learner': {'categorical_features': 'from_dtype', 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 1, 'learning_rate': 0.05, 'loss': 'squared_error', 'max_bins': 255, 'max_depth': None, 'max_features': 1.0, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'quantile': None, 'random_state': 0, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}, 'kernel': {'implementation': 'python_gaussian_nw_loocv_v1', 'kernel': 'gaussian_order2', 'regtype': 'local_constant', 'bwtype': 'fixed', 'cv': 'leave_one_out_squared_error', 'scale': 'sample_sd_times_fold_n_power_minus_0.2', 'lower_multiplier': 0.05, 'upper_multiplier': 20.0, 'grid_size': 41, 'refine': True, 'log_bandwidth_xatol': 1e-06, 'refine_maxiter': 80, 'constant_index': 'mean_response', 'final_include_self': True}, 'solvers': {'min_denominator': 1e-10, 'kappa': 1.345}, 'm3_solver': {'max_evaluations': 80, 'secant_steps': 6, 'initial_step': 0.02, 'bracket_width': 0.05, 'bracket_expansions': 6, 'max_brackets': 3, 'max_anchor_distance': 2.0, 'bracket_maxiter': 40, 'interior_margin': 0.05, 'moment_T_power': 0.25, 'score_tol': 1e-06}, 'm5': {'inner_K': 3, 'lambda_ratios': [1.0, 0.5623413251903491, 0.31622776601683794, 0.1778279410038923, 0.1, 0.05623413251903491, 0.03162277660168379, 0.01778279410038923, 0.01, 0.005623413251903491, 0.0031622776601683794, 0.0017782794100389228, 0.001], 'gamma': 1.0, 'weight_eps': 0.001, 'ridge_alpha': 1.0, 'pilot_epsilon': 1.35, 'nuisance_kappa': 1.345, 'kkt_tol': 2e-06}, 'critical': 1.96, 'score_ddof': 1, 'p': 5, 'K': 5, 'base_seed': 20261103, 'sources': [{'role': 'base_notebook', 'path': 'C:\\Users\\Lenovo\\Desktop\\DML_HeavyTail_Benchmark\\DML_p5_student_t.ipynb', 'sha256': '41ffadee98ab10aeb76b99c68f1cd308129a09c04dde29a80305ee4f0d6f4f55'}, {'role': 'adaptive_source', 'path': 'C:\\Users\\Lenovo\\Desktop\\Adaptive_Lasso_DML.py', 'sha256': 'e843649024ecca7bc9577f18586de620e8e7827374bf7d50db14f319860c7665'}, {'role': 'verified_dukes_source', 'path': 'C:\\Users\\Lenovo\\Desktop\\毕设\\DML_Dukes_n1000_R1000.ipynb', 'sha256': 'e80fbf528039c1c554b63b4a9cfbf96af5cab70f85f5da491f3934049c8c8394'}, {'role': 'dukes_validation_report', 'path': 'C:\\Users\\Lenovo\\Desktop\\DML_Dukes_full_results\\R_cross_validation\\cross_validation_report.md', 'sha256': 'fa2731456c54ff46401987a40262c0b078f76d7dc788d8ca841a8d8bb13cc3d1'}], 'copied_definitions': {'feature_dictionary': '109abfdd08de5b7e779a533a34c3280df8a3dea6a21bd253d071ae3a07bec514', 'huber_loss': 'bd496b3f1997cd64039c2b3a97cd54096a7313584e685e9973e76b15966acebf', 'prepare_training': 'f57a30f01eae59a9d36e869dca4404a26af60844da91a2ecddd3f8cffc0c42f4', 'fit_penalty': '40bba0058303b5f4dd0d1f32229376746d58bfed5709c2538f301956b1626f44', 'predict_state': 'bb1c316e8da4a3968cc83a04865b72fe81d92f7fe5c7c34fdb9058845d514b11', 'adaptive_prediction': 'e0a55b16a77575d521320eb320e9295a0a758d3cd08e4ebdca7ccd3e138837ee', 'NumericalFailure': '9feb3bdc92cd01c9b25cbcd4213562c0d57242d2d5a67ff43d410a9f5ce2951d', '_finite_vector': '9f615f5e93d8448ed5cdbbbaa7a52e04378376397794def5407cbcfe5c9d7a8b', 'validate_folds': '5bcb3778d9e8c5fa13e7c9d3376c1e42c858ac7d98b5e79e1f91ec61d77369d2', '_nw_from_distances': '243006c3a936ddf7c7da92690ad474cf1125b58b4309285860402a306b66617f', 'nw_predict': '6cd6adea70cdef76531b6dd3e26057ffdcc8220f7778d941621dc43e0b107818', 'select_nw': '4993e962e72b35a5dc69294f8a5acdb8abaca367f38107f57ae4e8d7dcbb3425', 'stable_projection': '7aac6903cbc94ce4ee6649bf5505ca92c39c265b5c5bedb3927b7793cf34e5fe', 'corrected_score': 'befea92bf10b6431b2e099852ae1d2b37e0e53af2c2fd4656bcdd1f680337e7d', 'dukes_from_initial': '256c0e04fb8bb6a573b38c3f5b8d928efe0ec6834aa002634167e66d492d2a72'}}, 'code_hash': '0b9e5fcc8d54b917a93ba43f746d574db9f3f9013806a3cf314627a134dc5c72', 'config_hash': 'f7d73b2c1cee1e1aec0cf3968699fbe83c98b67fa9215e2fe9b2952c62d9245e', 'output_root': 'C:\\Users\\Lenovo\\Desktop\\DML_5Methods_ScoreTest_Results', 'run_id': 'full_f7d73b2c1cee1e1a'}

SOURCE_NOTEBOOK_SHA256 = '804f7148c0c88e3cd5ad8b7f5406ccf027c63153f64d6a1e7cb7131b4d2f5e6a'

SOURCE_MAIN_FOLDER = Path('C:\\Users\\Lenovo\\Desktop\\DML_5Methods_ScoreTest_Results\\full\\full_f7d73b2c1cee1e1a')

def generate_pilot_observed(config, dgp, error, rep):
    n = config['n']
    seeds = {name: seed(config, dgp, error, rep, purpose)
             for name, purpose in [('X', 1), ('v', 2), ('epsilon', 3), ('folds', 4)]}
    X = np.random.default_rng(seeds['X']).multivariate_normal(
        np.zeros(5), ar1_covariance(5), n)
    m, g = nuisance_functions(X, DGP_NAMES[dgp])
    v = np.random.default_rng(seeds['v']).normal(size=n)
    rng = np.random.default_rng(seeds['epsilon'])
    epsilon = rng.normal(size=n) if error == 0 else standardized_lognormal(rng, LOGNORMAL_SIGMAS[error], n)
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


LOGNORMAL_SIGMAS={9:.5,11:1.,13:1.5,14:2.}
ERROR_LABELS={0:'Gaussian',**{e:f'LN({sigma:g})' for e,sigma in LOGNORMAL_SIGMAS.items()}}
PILOT_ERRORS=tuple(ERROR_LABELS)
PILOT_R=40
ROOT=Path(__file__).resolve().parent

def standardized_lognormal(rng,sigma,n):
    """总体均值中心化、总体方差标准化。"""
    mu=math.exp(sigma*sigma/2)
    return (np.exp(sigma*rng.normal(size=n))-mu)/(mu*math.sqrt(math.expm1(sigma*sigma)))

def notebook_style_fingerprint():
    # Notebook 逐条编译顶层定义；单文件编译的常量池可产生不同字节码索引。
    # 对脚本中的相同定义逐条编译后，使用原指纹规则核对主实验。
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    entries={}
    for name in STATISTICAL_DEFINITIONS:
        node=nodes[name]
        code=compile(ast.Module(body=[node],type_ignores=[]),'<source-audit>','exec',dont_inherit=True,optimize=0)
        body=next(x for x in code.co_consts if inspect.iscode(x))
        obj=globals()[name]
        if isinstance(node,ast.ClassDef):
            items=[(name+'.'+x.co_name,x,vars(obj)[x.co_name]) for x in body.co_consts if inspect.iscode(x)]
        else:
            items=[(name,body,obj)]
        for key,subcode,fn in items:
            entries[key]=dict(body=code_payload(subcode),defaults=code_payload(fn.__defaults__),
                              kwdefaults=code_payload(fn.__kwdefaults__))
    return digest_object(entries)

def pilot_config():
    found=notebook_style_fingerprint()
    if found!=SOURCE_CONFIG['code_hash']:
        raise RuntimeError(f'Student-t source fingerprint mismatch: {found} vs {SOURCE_CONFIG["code_hash"]}')
    versions={k:metadata.version(k) for k in SOURCE_CONFIG['dependencies']}
    if versions!=SOURCE_CONFIG['dependencies'] or platform.python_version()!=SOURCE_CONFIG['python']:
        raise RuntimeError(f'Dependency mismatch: {versions}')
    config=copy.deepcopy(SOURCE_CONFIG)
    config.update(R=PILOT_R,errors=list(PILOT_ERRORS),run_id='lognormal_pilot_r40',output_root=str(ROOT))
    config['pilot_design']=dict(sigmas=LOGNORMAL_SIGMAS,processes=1,threads=1,priority='Windows Idle',
        source_signature=SOURCE_CONFIG['method_signature'],notebook_sha256=SOURCE_NOTEBOOK_SHA256,
        verified_notebook_style_code_hash=found,script_runtime_code_hash=implementation_fingerprint(),
        generator_hash=digest_object(code_payload(generate_pilot_observed.__code__)),
        transform_hash=digest_object(code_payload(standardized_lognormal.__code__)))
    config['method_signature']=digest_object(config['pilot_design'])
    config.pop('config_hash')
    config['config_hash']=digest_object(config)
    return config

def pilot_task(config,task):
    dgp,error,rep=task
    path=ROOT/'checkpoints'/f'd{dgp}_e{error}_r{rep:02d}.json'
    if path.exists():
        packet=json.loads(path.read_text(encoding='utf-8'))
        if packet['config_hash']!=config['config_hash'] or packet['task']!=list(task) or set(packet['results'])!=set(METHODS):
            raise RuntimeError(f'Checkpoint 配置错误: {path}')
        return packet
    observed,folds,_,audit=generate_pilot_observed(config,dgp,error,rep)
    common=dict(run_id=config['run_id'],config_hash=config['config_hash'],n=config['n'],p=config['p'],K=config['K'],
                dgp_id=dgp,dgp=DGP_NAMES[dgp],error_id=error,error_label=ERROR_LABELS[error],replication_id=rep,
                data_hash=audit['data_hash'],fold_hash=audit['fold_hash'],seeds=audit['seeds'],
                estimator_signature=config['method_signature'])
    results={};cache={}
    for method in METHODS:
        row=dict(common,**estimate_record(method,observed,folds,config,dgp,error,rep,cache))
        results[method]=row
        for failure in row['failures']:
            tqdm.write(f'{DGP_NAMES[dgp]} {ERROR_LABELS[error]} rep={rep} {method} {failure["stage"]}: {failure["message"]}')
    packet=dict(task=list(task),config_hash=config['config_hash'],data_hash=audit['data_hash'],
                fold_hash=audit['fold_hash'],results=results)
    atomic_json(path,packet)
    return packet

def run_selfcheck(config):
    # 使用四份计划内数据。通过后复用这四份结果。
    tasks=[(d,e,0) for d in range(2) for e in (0,14)]
    packets=[pilot_task(config,task) for task in tqdm(tasks,desc='自检',unit='份')]
    records=[r for x in packets for r in x['results'].values()]
    tiny=dict(config,R=1,errors=[0,14])
    _,summary=summarize_records(records,tiny)
    report=validate_run(records,packets,summary,tiny)
    replay=[]
    for packet in packets:
        d,e,rep=packet['task']
        if e!=0:continue
        prior=json.loads((SOURCE_MAIN_FOLDER/'checkpoints'/f'd{d}_e0_r{rep:04d}.json').read_text(encoding='utf-8'))
        if prior['config_hash']!=SOURCE_CONFIG['config_hash'] or any(packet[k]!=prior[k] for k in ('data_hash','fold_hash')):
            raise RuntimeError('Gaussian 主实验数据、折或配置指纹不匹配')
        for m in METHODS:
            a,b=packet['results'][m],prior['results'][m]
            matched=all(a[k]==b[k] for k in ('point_status','test_status'))
            for k in ('theta_hat','T'):
                matched &= (a[k]==b[k]) if a[k] is None or b[k] is None else bool(np.isclose(a[k],b[k],atol=1e-10,rtol=1e-10))
            replay.append(dict(dgp=d,method=m,matched=matched))
    report['gaussian_replay']=replay
    report['passed']=report['complete'] and all(report['checks'].values()) and all(x['matched'] for x in replay)
    report['passed'] &= all(any(r['method']==m and r['point_status']==r['test_status']=='ok' for r in records) for m in METHODS)
    atomic_json(ROOT/'selfcheck.json',report)
    if not report['passed']:raise RuntimeError('自检失败，查看 selfcheck.json')
    print(f'SELFCHECK PASSED | 20 records | Gaussian replay 10/10 | point failures={report["point_failures"]}, test failures={report["test_failures"]}',flush=True)
    return {tuple(x['task']):x for x in packets}

def export_table(summary,paired,report):
    table=summary.copy()
    for key,count in [('点失败','N_point_valid'),('检验失败','N_test_valid'),('至少一项失败','N_point_test_common')]:
        table[key]=table['R_attempted']-table[count]
    labels={'dgp':'DGP','error_label':'误差','method':'方法','R_attempted':'尝试 N','N_point_valid':'点有效 N',
        '点失败':'点失败','N_test_valid':'检验有效 N','检验失败':'检验失败','N_point_test_common':'均成功 N',
        '至少一项失败':'至少一项失败','N_covered':'不拒绝 N','coverage':'Coverage','coverage_mcse':'Coverage MCSE',
        'bias':'Bias','rmse':'RMSE','coverage_deterioration_pp':'下降（百分点）','mean_T':'Mean(T)','sd_T':'SD(T)',
        'left_reject_rate':'左拒绝','right_reject_rate':'右拒绝','mean_y_nuisance_mse':'Y OOF MSE',
        'N_y_nuisance_mse':'Y MSE N','mean_d_nuisance_mse':'D OOF MSE','N_d_nuisance_mse':'D MSE N'}
    formatters={labels[k]:lambda v:f'{v:.1%}' for k in ('coverage','coverage_mcse','left_reject_rate','right_reject_rate')}
    body=table[list(labels)].rename(columns=labels).to_html(index=False,formatters=formatters,float_format=lambda v:f'{v:.5f}',na_rep='—')
    relevant=paired[(paired.error_id==14)&(((paired.dgp_id==0)&(paired.A=='M3')&(paired.B=='M5'))|((paired.dgp_id==1)&(paired.A=='M2')&(paired.B=='M5')))]
    pair_body=relevant.to_html(index=False,float_format=lambda v:f'{v:.5f}',na_rep='—')
    content=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Lognormal Pilot 五方法对照表</title>
<style>body{{font:15px "Microsoft YaHei",sans-serif;margin:30px;color:#18304d;background:#f6f8fc}}table{{border-collapse:collapse;background:white;margin:18px 0}}th,td{{padding:7px 10px;border:1px solid #d2ddec;white-space:nowrap}}th{{background:#203552;color:white}}tr:nth-child(even){{background:#edf3fa}}p{{max-width:1100px}}</style>
<h1>Lognormal Pilot 五方法对照表</h1><p>n=1000，p=5，外层五折，每场景 R=40。状态 {html.escape(report['status'])}。</p>
<p>Coverage 为真值处不拒绝比例。下降为同一 DGP 和方法的 Gaussian Coverage 减当前 Coverage。Y/D OOF MSE 以真实条件均值为参照，仅作评价。M3 的 Y 预测由原假设处 transformed-outcome 模型重建。</p>
<p>Lognormal 按总体均值中心化、总体方差标准化。M1 Standard DML，M2 Median-DML，M3 Dukes，M4 Huber-DML，M5 改进 Huber-DML。</p>
{body}<h2>σ=2 配对比较</h2><p>差值方向为 A − B，使用共同有效重复。</p>{pair_body}</html>"""
    (ROOT/'Lognormal_Pilot_对照表.html').write_text(content,encoding='utf-8')

def export_plot(summary):
    configure_plot_style()
    colors={'M1':'#4e79a7','M2':'#59a14f','M3':'#b07aa1','M4':'#e15759','M5':'#f28e2b'}
    metrics=(('coverage','Coverage (%)',100.,95.),('rmse','RMSE',1.,None),
             ('coverage_deterioration_pp','Coverage 下降（百分点）',1.,0.))
    fig,axes=plt.subplots(3,2,figsize=(15,11.5))
    x=np.arange(len(PILOT_ERRORS));labels=list(ERROR_LABELS.values())
    for i,(metric,title,scale,reference) in enumerate(metrics):
        for d in range(2):
            ax=axes[i,d]
            for m in METHODS:
                rows=summary[(summary.dgp_id==d)&(summary.method==m)].set_index('error_id').reindex(PILOT_ERRORS)
                ax.plot(x,rows[metric]*scale,marker='o',color=colors[m],label=m,lw=2)
            if reference is not None:ax.axhline(reference,color='#77869a',ls='--',lw=1)
            if metric=='coverage':ax.set_ylim(-2,103)
            ax.set(title=('DGP1 线性' if d==0 else 'DGP2 非线性')+' · '+title,xticks=x,xticklabels=labels)
            ax.grid(axis='y',alpha=.2);ax.legend(ncol=5,fontsize=9)
    fig.suptitle('Lognormal Pilot · 五方法 Score-test 对照',fontsize=20,weight='bold')
    fig.text(.05,.015,'n=1000，p=5，每场景 R=40。Coverage 与点估计各自使用有效分母，详见完整对照表。',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.96))
    fig.savefig(ROOT/'Lognormal_Pilot_对照图.png',dpi=180);plt.close(fig)

def main():
    lock=ROOT/'.pilot.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        with threadpool_limits(limits=1):
            pools=threadpool_info()
            if any(lib['num_threads']!=1 for lib in pools):raise RuntimeError(f'计算库非单线程: {pools}')
            config=pilot_config()
            atomic_json(ROOT/'pilot_config.json',config)
            print(f'PID={os.getpid()} | 单进程、单线程、Low | 线程池: {[x["num_threads"] for x in pools]}',flush=True)
            print('主实验统计函数指纹与依赖一致。',flush=True)
            started=time.perf_counter()
            checked=run_selfcheck(config)
            tasks=[(d,e,r) for d in range(2) for e in PILOT_ERRORS for r in range(PILOT_R)]
            packets=[checked[task] if task in checked else pilot_task(config,task)
                     for task in tqdm(tasks,desc='五方法 Pilot R=40',unit='份',mininterval=5)]
            records=[r for x in packets for r in x['results'].values()]
            frame,summary=summarize_records(records,config)
            report=validate_run(records,packets,summary,config)
            report['elapsed_seconds']=time.perf_counter()-started
            paired=paired_comparisons(frame,config)
            atomic_json(ROOT/'validation_report.json',report)
            atomic_json(ROOT/'summary.json',summary.to_dict('records'))
            atomic_json(ROOT/'paired_comparisons.json',paired.to_dict('records'))
            with (ROOT/'failures.jsonl').open('w',encoding='utf-8') as f:
                for r in records:
                    for problem in r['failures']:
                        f.write(json.dumps(clean_json(dict(dgp=r['dgp'],error=r['error_label'],rep=r['replication_id'],method=r['method'],**problem)),ensure_ascii=False)+'\n')
            export_table(summary,paired,report);export_plot(summary)
            print(summary[['dgp','error_label','method','R_attempted','N_point_valid','N_test_valid','coverage','bias','rmse',
                 'coverage_deterioration_pp','mean_T','sd_T','left_reject_rate','right_reject_rate','mean_y_nuisance_mse','mean_d_nuisance_mse']].to_string(index=False),flush=True)
            print(f'PILOT COMPLETE | {report["status"]} | records={len(records)} | point failures={report["point_failures"]} | test failures={report["test_failures"]} | {report["elapsed_seconds"]/60:.1f} min',flush=True)
            if not report['complete'] or not all(report['checks'].values()):raise RuntimeError('最终验证未通过')
    finally:lock.unlink()

if __name__=='__main__':
    main()

# -*- coding: utf-8 -*-
"""Windows spawn Worker、断点存储和报表。正式实验入口是桌面 Notebook。"""
import os
import sys
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'
for _stream in (sys.stdout,sys.stderr):
    if hasattr(_stream,'reconfigure'):
        _stream.reconfigure(encoding='utf-8',errors='replace')

import hashlib
import html
import importlib.metadata as metadata
import json
import math
import multiprocessing
import platform
import sqlite3
import time
import traceback
import uuid
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

SOURCE_HASHES = {'Lognormal_Pilot_R200.py': '8b8ac81045ffd125c941e77c1f1c872e51bcbbf1112833382f368b59dff26a91', 'DML_p5_StudentT_5Methods_ScoreTest.ipynb': '804f7148c0c88e3cd5ad8b7f5406ccf027c63153f64d6a1e7cb7131b4d2f5e6a', 'Spline_Ridge_DML_Pilot.py': '380c60a172f0a1a8acc32311a2a7fc5a82ba08b0999160dc0ec42c7ff424d6d9', 'Adaptive_Centering_DML_Pilot.py': 'b55c218eb46e8b895f4a020031eb27561f56c95606bf96848be5425977eb4157', 'Lognormal_Pilot.py': '0c842dc57a53d7997804e60aa4e9a3596b9ac44ec8177350178da5184f916655'}
ORIGINAL_DEPENDENCIES = {'numpy': '2.3.5', 'scipy': '1.17.1', 'scikit-learn': '1.7.2', 'joblib': '1.5.3', 'pandas': '2.3.3', 'matplotlib': '3.10.9', 'threadpoolctl': '3.6.0', 'tqdm': '4.67.3'}
ORIGINAL_PYTHON = '3.11.15'

BASE_SEED = 20261103

P = 5

K = 5

CRITICAL = 1.96

THETA_TRUE = 1.0

THETA_NULL = 1.0

DGP_NAMES = ('DGP1_linear','DGP2_nonlinear')

T_GRID = (20.,10.,5.,3.,2.6,2.3,2.1)

SOLVER = {'min_denominator':1e-10,'kappa':1.345}

M3_SOLVER = {'max_evaluations':80,'secant_steps':6,'initial_step':0.02,
 'bracket_width':0.05,'bracket_expansions':6,'max_brackets':3,'max_anchor_distance':2.,
 'bracket_maxiter':40,'interior_margin':0.05,'moment_T_power':0.25,'score_tol':1e-6}

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

WORKERS=8

INNER_K=3

ALPHAS = np.logspace(-4,4,17)

KNOTS = 5

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

def failure_details(exc,stage):
    return dict(stage=stage,type=getattr(exc,'failure_type',type(exc).__name__),
                fold=getattr(exc,'fold',None),message=str(exc),traceback=traceback.format_exc())

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

def configure_plot_style():
    fonts={font.name for font in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun'):
        if name in fonts:
            plt.rcParams['font.sans-serif']=[name,'DejaVu Sans'];break
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'#f6f8fc','axes.facecolor':'white'})


def init_worker(barrier):
    global worker_limits
    worker_limits=threadpool_limits(limits=1)
    if any(lib['num_threads']!=1 for lib in threadpool_info()):
        raise RuntimeError('Worker 数值计算库线程数不为 1')
    if barrier is not None:
        barrier.wait()

def position(method, residual, tau):
    if method == 'A':
        return float(np.mean(residual))
    if method == 'B':
        return float(np.median(residual))
    if method != 'C':
        raise ValueError(f'未知位置方法 {method}')
    low, high = float(np.min(residual)), float(np.max(residual))
    def moment(a):
        return float(np.mean(np.clip(residual-a, -tau, tau)))
    result = brentq(moment, low, high, xtol=1e-12, rtol=1e-14)
    if abs(moment(result)) > 1e-10:
        raise ArithmeticError('Huber_location_score_residual')
    return result

def centered_point(method, w, u, initial, tau):
    # 以仅依赖数据的 OLS 为锚，从近到远寻找连续 profile 矩的变号区间。
    def moment(theta):
        e = u - theta*w
        a = position(method, e, tau)
        return float(np.mean(w*np.clip(e-a, -tau, tau)))
    calls = 1
    origin = moment(initial)
    left = right = (initial, origin)
    for expansion in range(27):
        if abs(origin) <= 1e-10:
            theta = initial
            break
        radius = .01 * 2**expansion
        endpoints = [(initial-radius, moment(initial-radius)),
                     (initial+radius, moment(initial+radius))]
        calls += 2
        brackets = [(endpoints[0], left), (right, endpoints[1])]
        # 只接受局部下降方向的交叉。等距时优先与锚的矩符号相符的一侧。
        if origin > 0:
            brackets.reverse()
        for low, high in brackets:
            if low[1] >= 0 >= high[1]:
                theta, details = brentq(moment, low[0], high[0], xtol=1e-12,
                                        rtol=1e-14, full_output=True)
                calls += details.function_calls
                break
        else:
            left, right = endpoints
            continue
        break
    else:
        raise ArithmeticError(f'profile_bracket_failed: evaluations={calls}')
    e = u-theta*w
    a = position(method, e, tau)
    error = abs(moment(theta))
    if not np.isfinite(theta) or error > 1e-9:
        raise ArithmeticError(f'profile_score_residual={error:.9g}')
    return dict(theta=float(theta), point_a=a, point_score_residual=error,
                point_evaluations=calls+1)

def centered_test(method, w, e, tau):
    a = position(method, e, tau)
    centered = e-a
    psi = np.clip(centered, -tau, tau)
    s = w*psi
    derivative_s = -float(np.mean(w*(abs(centered) < tau)))
    bandwidth = density = None
    if method == 'A':
        q, derivative_q = centered, -1.
    elif method == 'B':
        q = np.sign(centered)
        # 稳健 Silverman 带宽及高斯核。使用同一份原假设残差。
        iqr = float(np.subtract(*np.percentile(e, [75,25])))
        bandwidth = .9*min(float(np.std(e, ddof=1)), iqr/1.349)*len(e)**(-.2)
        if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
            raise ArithmeticError('median_density_bandwidth_invalid')
        density = float(np.mean(norm.pdf(centered/bandwidth))/bandwidth)
        derivative_q = -2*density
    else:
        q = psi
        derivative_q = -float(np.mean(abs(centered) < tau))
    if not np.isfinite(derivative_q) or abs(derivative_q) <= 1e-12:
        raise ArithmeticError('location_equation_derivative_invalid')
    coefficient = derivative_s/derivative_q
    adjusted = standardize_score(s-coefficient*q, CRITICAL)
    return dict(a=a, psi_mean=float(psi.mean()), q_mean=float(q.mean()),
                location_lambda=coefficient, derivative_a_s=derivative_s,
                derivative_a_q=derivative_q, median_density=density,
                median_bandwidth=bandwidth, T=adjusted['T'],
                score_mean=adjusted['score_mean'], score_sd=adjusted['score_sd'])

def naive_centered_test(method, w, e, tau):
    a = position(method,e,tau)
    naive = standardize_score(w*np.clip(e-a,-tau,tau),CRITICAL)
    return dict(naive_T=naive['T'], naive_score_mean=naive['score_mean'],
                naive_score_sd=naive['score_sd'])

def oracle_replicate(method, observed, dgp, tau):
    X,D,Y = observed
    m,g = nuisance_functions(X,DGP_NAMES[dgp])
    w = D-m
    residual = Y-THETA_NULL*D-g
    return {f'oracle_{key}':value for key,value in
            centered_test(method,w,residual,tau).items()
            if key in ('T','score_mean','score_sd','a','psi_mean')}

def record_stage(row, stage, operation):
    # 点估计、主检验、naive 检验和诊断有独立有效分母。
    started = time.perf_counter()
    try:
        row.update(operation())
    except Exception as exc:
        evidence = failure_details(exc, stage)
        evidence.pop('traceback')
        row['failures'].append(evidence)
    row[f'{stage}_seconds'] = time.perf_counter()-started

def spline_features(train, test):
    """每次只由传入的训练子折确定分位数结点和标准化系数。"""
    spline = SplineTransformer(n_knots=KNOTS,degree=3,knots='quantile',
                               include_bias=False,extrapolation='constant')
    spline.fit(train)
    left,right = np.triu_indices(train.shape[1],k=1)
    def feature_matrix(X):
        return np.column_stack((spline.transform(X),X,X[:,left]*X[:,right]))
    train_features = feature_matrix(train)
    scaler = StandardScaler().fit(train_features)
    return scaler.transform(train_features),scaler.transform(feature_matrix(test))

def ridge_cv_alphas(X, targets, splits, verify):
    # RidgeCV 的每个显式 CV split 只读取自己的预处理块。复制原始观测的索引，
    # 让各训练子折可以有不同的样条结点与尺度。自动全块 refit 不用于预测。
    blocks, responses, cv = [], [], []
    offset = 0
    for train,val in splits:
        X_train,X_val = spline_features(X[train],X[val])
        blocks.extend((X_train,X_val))
        responses.extend((targets[train],targets[val]))
        cv.append((np.arange(offset,offset+len(train)),
                   np.arange(offset+len(train),offset+len(train)+len(val))))
        offset += len(train)+len(val)
    features = np.vstack(blocks)
    y = np.vstack(responses)
    alpha = [float(RidgeCV(alphas=ALPHAS,cv=cv,scoring='neg_mean_squared_error')
                   .fit(features,y[:,side]).alpha_) for side in range(y.shape[1])]
    if verify:
        # 自检独立展开折和 alpha，确认 RidgeCV 选择与真正的逐折 MSE 相同。
        for side,value in enumerate(alpha):
            errors = [np.mean([np.mean((Ridge(alpha=a).fit(features[tr],y[tr,side])
                         .predict(features[va])-y[va,side])**2) for tr,va in cv]) for a in ALPHAS]
            assert value==ALPHAS[np.argmin(errors)], 'RidgeCV 与逐折平方损失不一致'
    return alpha

def spline_oof(observed,folds,config,task):
    X,D,Y = observed
    targets = np.column_stack((D,Y))
    prediction = np.full((2,len(D)),np.nan)
    selected = []
    for k,(train,test) in enumerate(folds):
        splits = list(KFold(INNER_K,shuffle=True,random_state=seed(config,*task,50,k,0)).split(train))
        try:
            alpha = ridge_cv_alphas(X[train],targets[train],splits,config['mode']=='smoke' and k==0)
            X_train,X_test = spline_features(X[train],X[test])
            for side,a in enumerate(alpha):
                prediction[side,test] = Ridge(alpha=a).fit(X_train,targets[train,side]).predict(X_test)
        except Exception as exc:
            failure = NumericalFailure('spline_ridge_fit_failed',f'fold={k}: {type(exc).__name__}: {exc}')
            failure.fold = k
            raise failure from exc
        selected.append(alpha)
    if not np.isfinite(prediction).all():
        raise ArithmeticError('Spline-Ridge OOF 预测不完整')
    return prediction,selected

def prediction_errors(observed,dgp,m,ell,transformed=False):
    X,D,Y = observed
    m0,g0 = nuisance_functions(X,DGP_NAMES[dgp])
    dm = None if m is None else m-m0
    if transformed:
        ell0 = g0
        outcome = Y-THETA_NULL*D
    else:
        ell0 = THETA_TRUE*m0+g0
        outcome = Y
    dy = None if ell is None else ell-ell0
    result = dict(y_target='E[Y−θ₀D|X]' if transformed else 'E[Y|X]')
    if dm is not None:
        result.update(d_nuisance_mse=float(np.mean(dm**2)),d_prediction_bias=float(dm.mean()),
                      d_observed_mse=float(np.mean((m-D)**2)),delta_m2=float(np.mean(dm**2)))
    if dy is not None:
        result.update(y_nuisance_mse=float(np.mean(dy**2)),y_prediction_bias=float(dy.mean()),
                      y_observed_mse=float(np.mean((ell-outcome)**2)))
    if dm is not None and dy is not None and not transformed:
        result.update(delta_m_delta_ell=float(np.mean(dm*dy)),
                      delta_product_net=float(np.mean(dm*dy-THETA_TRUE*dm**2)))
    return result

def mean_huber(row,observed,task,m,ell):
    _,D,Y = observed
    w,u = D-m,Y-ell
    def threshold():
        initial,tau,_ = huber_threshold(w,u)
        return dict(initial=initial,tau=tau)
    record_stage(row,'threshold',threshold)
    tau = row.get('tau')
    if tau is not None:
        e = u-THETA_NULL*w
        record_stage(row,'point',lambda:centered_point('A',w,u,row['initial'],tau))
        record_stage(row,'test',lambda:centered_test('A',w,e,tau))
        record_stage(row,'naive',lambda:naive_centered_test('A',w,e,tau))
        record_stage(row,'oracle_fixed_tau',lambda:{'oracle_fixed_tau_T':oracle_replicate('A',observed,task[0],tau)['oracle_T']})

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
            objective=None
            start=time.perf_counter()
            try:
                if method=='M1':
                    theta=ols_point(w,u)
                    score_error=float(abs(np.mean(w*(u-theta*w))))
                    if score_error>1e-9:raise ArithmeticError('OLS_score_residual')
                    point_diag=dict(score_error=score_error)
                elif method=='M2':
                    theta,point_diag=lad_point(w,u)
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
                else:
                    scores=residual_scores(method,w,u,hypothesis)
                record.update(standardize_score(scores,config['critical']))
                record.update(test_status='ok',theta_tested=hypothesis)
                record['diagnostics']['test_score_variance_checked']=bool(
                    np.isclose(record['score_variance'],np.var(scores,ddof=1),rtol=1e-12,atol=1e-12))
                # M3 的 Y 诊断以实际 transformed-outcome nuisance 为准。
                record.update(prediction_errors(observed,dgp,predictions[0],
                    m_hat if method=='M3' else predictions[1],transformed=method=='M3'))
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


ROOT = Path(__file__).resolve().parent
METHODS = ('M1','M2','M3','S')
LABELS = dict(M1='M1 Standard DML',M2='M2 Median-DML',M3='M3 Dukes',S='S Spline-Ridge-Huber')
COLORS = dict(M1='#536d89',M2='#c57829',M3='#8662a7',S='#007d75')
FAMILIES = {'Student-t':(0,*range(1,8))}
ERROR_LABELS = {0:'Gaussian',**{i:f't({df:g})' for i,df in enumerate(T_GRID,1)}}
FLUSH_EVERY = 32
ORACLE_EVERY = 100
FULL_R = 1000
CORE_DGP = ('DGP1 线性','DGP2 非线性')


def config_for(mode):
    if mode not in ('smoke','full'):
        raise ValueError(f'未知模式 {mode}')
    current = {k:metadata.version(k) for k in ORIGINAL_DEPENDENCIES}
    if current!=ORIGINAL_DEPENDENCIES or platform.python_version()!=ORIGINAL_PYTHON:
        raise RuntimeError(f'需使用原实验 environment2 内核。当前依赖 {current}')
    cfg = dict(mode=mode,n=1000,R=FULL_R if mode=='full' else 1,
               p=P,K=K,base_seed=BASE_SEED,critical=CRITICAL,
               theta_true=THETA_TRUE,theta_null=THETA_NULL,
               errors=list(ERROR_LABELS) if mode=='full' else [0,1,5,7],
               spec=dict(learner=HGBR_PARAMS,kernel=DEFAULT_KERNEL_CONFIG,m3_solver=M3_SOLVER),
               source_hashes=SOURCE_HASHES, worker_sha256=sha(Path(__file__).read_bytes()),
               workers=WORKERS,threads=1,python=ORIGINAL_PYTHON,dependencies=current)
    cfg['config_hash'] = sha(json.dumps(cfg,sort_keys=True).encode())
    return cfg


def get_predictions(method,observed,folds,config,dgp,error,rep,cache):
    # 原 M1/M3 共享 HGBR squared 的 OOF。M2 的 quantile 预测独立。
    target = 'median' if method=='M2' else 'mean'
    if target not in cache:
        cache[target] = crossfit(observed,folds,config,dgp,error,rep,median=target=='median')
    return cache[target],[]


def generate(config, task):
    if task[1] not in config['errors']:
        raise ValueError(f'误差水平不属于本次范围 {task[1]}')
    return generate_observed(config,*task)


def attach_compact_baseline(record):
    diagnostics = record.pop('diagnostics')
    point = diagnostics.get('point',{})
    record['dukes_evaluations'] = point.get('evaluations')
    record['dukes_point_equation_T'] = point.get('point_equation_T')
    record['point_score_residual'] = point.get('root_score_mean',point.get('score_error'))
    record['corrected_identity_checked'] = diagnostics.get('full_score_identity_checked')
    for fail in record['failures']:
        fail.pop('traceback',None)
    return record


def fit_s(config,task,observed,folds):
    row = dict(method='S',failures=[],warnings=[])
    started = time.perf_counter()
    def first_stage():
        prediction,alpha = spline_oof(observed,folds,config,task)
        return dict(predictions=prediction,alpha_d=[v[0] for v in alpha],alpha_y=[v[1] for v in alpha])
    record_stage(row,'nuisance',first_stage)
    prediction = row.pop('predictions',None)
    if prediction is not None:
        mean_huber(row,observed,task,*prediction)
        row.update(prediction_errors(observed,task[0],*prediction))
        row['prediction_hash'] = sha(prediction.tobytes())
        if row.get('T') is not None:
            if not np.isclose(row['score_mean'],row.get('naive_score_mean',np.nan),rtol=1e-9,atol=1e-11):
                raise ArithmeticError('S 位置调整后的分子与原始平均 score 不符')
    row['theta_hat'] = row.pop('theta',None)
    row['point_status'] = 'ok' if row['theta_hat'] is not None else 'failed'
    row['test_status'] = 'ok' if row.get('T') is not None else 'failed'
    row['nuisance_status'] = 'ok' if prediction is not None else 'failed'
    row['accepted'] = bool(abs(row['T'])<=CRITICAL) if row.get('T') is not None else None
    row['score_ddof'] = 1 if row.get('T') is not None else None
    if task[2]%ORACLE_EVERY!=0:
        row.pop('oracle_fixed_tau_T',None)
    row['elapsed_seconds'] = time.perf_counter()-started
    return row


def worker_packet(config,task,methods=METHODS):
    started = time.perf_counter()
    observed,folds,_,audit = generate(config,task)
    cache, rows = {},{}
    for method in methods:
        if method=='S':
            row = fit_s(config,task,observed,folds)
        else:
            row = attach_compact_baseline(estimate_record(method,observed,folds,config,*task,cache))
            if row['nuisance_status']=='ok' and method in ('M1','M2') and 'y_nuisance_mse' not in row:
                prediction,_ = get_predictions(method,observed,folds,config,*task,cache)
                row.update(prediction_errors(observed,task[0],*prediction))
            if method=='M3':
                row.setdefault('y_target','E[Y−θ₀D|X]')
                if row['nuisance_status']=='ok' and 'd_nuisance_mse' not in row:
                    prediction,_ = get_predictions(method,observed,folds,config,*task,cache)
                    row.update(prediction_errors(observed,task[0],prediction[0],None,transformed=True))
            if method=='M1' and task[2]%ORACLE_EVERY==0:
                def oracle():
                    X,D,Y = observed
                    m,g = nuisance_functions(X,DGP_NAMES[task[0]])
                    w,u = D-m,Y-(THETA_TRUE*m+g)
                    return dict(oracle_fixed_tau_T=standardize_score(residual_scores('M1',w,u,THETA_NULL),CRITICAL)['T'])
                record_stage(row,'oracle',oracle)
        rows[method] = row
    return dict(task=list(task),config_hash=config['config_hash'],data_hash=audit['data_hash'],
                fold_hash=audit['fold_hash'],seeds=audit['seeds'],
                epsilon_sample_variance=audit['epsilon_sample_variance'],
                worker_pid=os.getpid(),seconds=time.perf_counter()-started,rows=rows)


def validate_packet(packet,config,task,methods=METHODS):
    assert packet['task']==list(task) and packet['config_hash']==config['config_hash']
    assert set(packet['rows'])==set(methods)
    assert len(packet['data_hash'])==64 and len(packet['fold_hash'])==64
    assert packet['seeds']=={name:seed(config,*task,index) for index,name in enumerate(('X','v','epsilon','folds'),1)}
    for method,row in packet['rows'].items():
        assert row['method']==method
        assert (row['theta_hat'] is not None)==(row['point_status']=='ok')
        if row['theta_hat'] is not None:
            assert np.isfinite(row['theta_hat'])
        assert (row.get('T') is not None)==(row['test_status']=='ok')
        assert (row['point_status']==row['test_status']=='ok') or row['failures']
        if row.get('T') is not None:
            assert row['score_ddof']==1 and row['score_sd']>0
            assert np.isclose(row['T'],np.sqrt(config['n'])*row['score_mean']/row['score_sd'],
                              rtol=1e-12,atol=1e-12)
            assert row['accepted']==bool(abs(row['T'])<=config['critical'])
            if method=='M3':
                assert row['corrected_identity_checked'] and row['y_target']=='E[Y−θ₀D|X]'
        if method=='M3' and row['point_status']=='ok':
            assert abs(row['dukes_point_equation_T'])<=config['n']**(-M3_SOLVER['moment_T_power'])


def failed_methods(packet):
    return tuple(m for m,row in packet['rows'].items()
                 if row['point_status']!='ok' or row['test_status']!='ok')


def merged_packet(previous,current):
    for key in ('task','config_hash','data_hash','fold_hash','seeds'):
        assert previous[key]==current[key],key
    result = {**current,'rows':dict(previous['rows'])}
    for m,row in current['rows'].items():
        old = previous['rows'][m]
        # 保留已有有效点估计/检验，成功分量不得因一次重试而丢失。
        if any(old[key]=='ok' and row[key]!='ok' for key in ('point_status','test_status')):
            continue
        result['rows'][m] = row
    return result


@contextmanager
def run_lock(destination):
    import msvcrt
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'run.lock').open('a+b') as handle:
        if handle.tell()==0:
            handle.write(b'1')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:
            raise RuntimeError(f'此运行目录已由另一个 Notebook 占用 {destination}') from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


@contextmanager
def checkpoint_db(destination,config):
    path = destination/'checkpoints'
    path.mkdir(exist_ok=True)
    db = sqlite3.connect(path/'replications.sqlite3')
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS manifest (config_hash TEXT PRIMARY KEY, config TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS packets (dgp INTEGER, error INTEGER, rep INTEGER,
            attempt INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(dgp,error,rep));
        CREATE TABLE IF NOT EXISTS attempts (dgp INTEGER, error INTEGER, rep INTEGER,
            attempt INTEGER, payload TEXT NOT NULL, PRIMARY KEY(dgp,error,rep,attempt));
        ''')
        manifest = db.execute('SELECT config_hash FROM manifest').fetchall()
        if manifest and manifest!=[(config['config_hash'],)]:
            raise RuntimeError('断点与当前配置或代码不一致，需要独立运行目录')
        with db:
            db.execute('INSERT OR IGNORE INTO manifest VALUES (?,?)',
                       (config['config_hash'],json.dumps(config,ensure_ascii=False)))
        yield db
    finally:
        db.close()


def record_batch(db,batch):
    # 只有 Notebook 父进程持久化。每批事务完整提交，保留所有历史尝试。
    with db:
        for task,attempt,raw,chosen in batch:
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?)',
                       (*task,attempt,json.dumps(clean_json(raw),ensure_ascii=False,allow_nan=False)))
            db.execute('INSERT OR REPLACE INTO packets VALUES (?,?,?,?,?)',
                       (*task,attempt,json.dumps(clean_json(chosen),ensure_ascii=False,allow_nan=False)))


def load_packets(db,config):
    loaded = {}
    for d,e,r,attempt,payload in db.execute('SELECT dgp,error,rep,attempt,payload FROM packets'):
        task = (d,e,r)
        packet = json.loads(payload)
        validate_packet(packet,config,task)
        loaded[task] = (attempt,packet)
    return loaded


def append_log(destination,message):
    line = f'{datetime.now().astimezone().isoformat(timespec="seconds")} {message}'
    with (destination/'run.log').open('a',encoding='utf-8') as stream:
        print(line,file=stream,flush=True)


def save_progress(destination,config,loaded,completed,total,elapsed,started_pending,status):
    counts = Counter((d,e) for d,e,_ in loaded)
    rate = completed/elapsed if elapsed>0 and completed else None
    scenes = {f'DGP{d+1} {ERROR_LABELS[e]}':counts[d,e]
              for d in range(2) for e in config['errors']}
    progress = dict(status=status,mode=config['mode'],completed=len(loaded),total=total,
        session_completed=completed,session_pending=started_pending,session_seconds=elapsed,
        datasets_per_second=rate,eta_seconds=(started_pending-completed)/rate if rate else None,
        updated=datetime.now().astimezone().isoformat(timespec='seconds'),
        per_scenario=scenes,config_hash=config['config_hash'])
    atomic_json(destination/'progress.json',progress)
    return progress


def collect(destination,db,config,retry_failed=False):
    tasks = [(d,e,r) for r in range(config['R']) for d in range(2) for e in config['errors']]
    loaded = load_packets(db,config)
    if not loaded.keys()<=set(tasks):
        raise RuntimeError('断点存在不属于本次设计的任务')
    todo = {}
    for task in tasks:
        if task not in loaded:
            todo[task] = METHODS
        elif retry_failed:
            methods = failed_methods(loaded[task][1])
            if methods:
                todo[task] = methods
    append_log(destination,f'断点已完成 {len(loaded)}/{len(tasks)}，本次待处理 {len(todo)}')
    save_progress(destination,config,loaded,0,len(tasks),0,len(todo),'running')
    if not todo:
        return loaded
    ctx = multiprocessing.get_context('spawn')
    is_fresh_smoke = config['mode']=='smoke' and len(todo)==WORKERS and not loaded
    barrier = ctx.Barrier(WORKERS,timeout=180) if is_fresh_smoke else None
    started = time.perf_counter()
    batch,completed = [],0
    source = iter(todo.items())
    futures = {}
    def submit_one(pool):
        next_task = next(source,None)
        if next_task is not None:
            task,methods = next_task
            futures[pool.submit(worker_packet,config,task,methods)] = task
    def flush():
        if batch:
            record_batch(db,batch)
            batch.clear()
            p = save_progress(destination,config,loaded,completed,len(tasks),
                              time.perf_counter()-started,len(todo),'running')
            append_log(destination,f'{p["completed"]}/{p["total"]} 已持久化，'
                       f'本次 {completed}/{len(todo)}，{p["datasets_per_second"]:.2f} 组/s')
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=ctx,initializer=init_worker,
                             initargs=(barrier,)) as pool:
        for _ in range(3*WORKERS):
            submit_one(pool)
        retry_n = sum(task in loaded for task in todo)
        with tqdm(total=len(tasks)+retry_n,initial=len(loaded),
                  desc='正式 R1000' if config['mode']=='full' else 'Smoke 八进程',unit='组',mininterval=12) as bar:
            try:
                while futures:
                    ready,_ = wait(futures,return_when=FIRST_COMPLETED)
                    for future in ready:
                        task = futures.pop(future)
                        raw = future.result()
                        validate_packet(raw,config,task,todo[task])
                        previous_attempt,previous = loaded.get(task,(0,None))
                        selected = raw if previous is None else merged_packet(previous,raw)
                        batch.append((task,previous_attempt+1,raw,selected))
                        loaded[task] = (previous_attempt+1,selected)
                        completed += 1
                        bar.update()
                        if completed%FLUSH_EVERY==0 or completed==len(todo):
                            seen = Counter((d,e) for d,e,_ in loaded)
                            bar.set_postfix_str(f'场景 {min(seen.values(),default=0)}–{max(seen.values(),default=0)}/{config["R"]}')
                            flush()
                        submit_one(pool)
            finally:
                flush()
                for future in futures:
                    future.cancel()
    return loaded


def atomic_csv(destination,frame):
    path = Path(destination)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    frame.to_csv(temp,index=False,encoding='utf-8-sig')
    os.replace(temp,path)


def atomic_text(path,contents):
    path = Path(path)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(contents,encoding='utf-8')
    os.replace(tmp,path)


def flatten_results(db,loaded,config):
    rows,failures = [],[]
    for task,(attempt,p) in sorted(loaded.items()):
        d,e,r = task
        common = dict(dgp_id=d,dgp=CORE_DGP[d],error_id=e,error=ERROR_LABELS[e],
                      replication_id=r,data_hash=p['data_hash'],fold_hash=p['fold_hash'],
                      epsilon_sample_variance=p['epsilon_sample_variance'],**{f'seed_{k}':v for k,v in p['seeds'].items()})
        for method,row in p['rows'].items():
            flat = dict(common,attempt_count=attempt,**{k:v for k,v in row.items() if k not in ('failures','warnings')})
            flat['failures_n'] = len(row['failures'])
            flat['warnings'] = ' | '.join(row['warnings'])
            if method=='M2':
                flat['y_target']='E[Y|X] 共同诊断参照，学习器拟合条件中位数'
            if row['theta_hat'] is not None:
                flat['estimation_error'] = row['theta_hat']-THETA_TRUE
            rows.append(flat)
    for d,e,r,attempt,raw in db.execute('SELECT dgp,error,rep,attempt,payload FROM attempts ORDER BY dgp,error,rep,attempt'):
        packet = json.loads(raw)
        for m,row in packet['rows'].items():
            for failure in row['failures']:
                failures.append(dict(dgp_id=d,error_id=e,replication_id=r,method=m,attempt=attempt,
                                     point_status=row['point_status'],test_status=row['test_status'],**failure))
    result = pd.DataFrame(rows)
    assert len(result)==len(loaded)*4
    assert not result.duplicated(['dgp_id','error_id','replication_id','method']).any()
    assert np.array_equal(result.point_status.eq('ok').to_numpy(),result.theta_hat.notna().to_numpy())
    assert np.array_equal(result.test_status.eq('ok').to_numpy(),result['T'].notna().to_numpy())
    return result,pd.DataFrame(failures,columns=['dgp_id','error_id','replication_id','method','attempt',
                                               'point_status','test_status','stage','type','fold','message'])


def summarize_results(raw):
    rows,nuisance = [],[]
    for (d,e,m),group in raw.groupby(['dgp_id','error_id','method'],sort=True):
        T = group['T'].dropna()
        err = group.estimation_error.dropna()
        naive = group.naive_T.dropna()
        info = dict(dgp_id=d,dgp=CORE_DGP[d],error_id=e,error=ERROR_LABELS[e],method=m)
        targets = group.y_target.dropna()
        prediction = {key:group[key].mean() for key in
                      ('y_nuisance_mse','d_nuisance_mse','y_prediction_bias','d_prediction_bias',
                       'y_observed_mse','d_observed_mse')}
        rows.append(dict(info,planned_n=len(group),point_n=len(err),point_fail=len(group)-len(err),
            test_n=len(T),test_fail=len(group)-len(T),accept_n=int((T.abs()<=CRITICAL).sum()),
            coverage_pct=100*(T.abs()<=CRITICAL).mean(),bias=err.mean(),rmse=np.sqrt((err**2).mean()),
            mean_T=T.mean(),sd_T=T.std(ddof=1),left_reject_pct=100*(T< -CRITICAL).mean(),
            right_reject_pct=100*(T>CRITICAL).mean(),naive_test_n=len(naive),
            naive_coverage_pct=100*(naive.abs()<=CRITICAL).mean(),
            y_oof_n=int(group.y_nuisance_mse.notna().sum()),d_oof_n=int(group.d_nuisance_mse.notna().sum()),
            y_target=targets.iloc[0] if len(targets) else '',**prediction))
        diag = dict(info,**prediction)
        for k in ('delta_m_delta_ell','delta_m2','delta_product_net'):
            series = group[k].dropna()
            diag.update({f'{k}_n':len(series),f'{k}_mean':series.mean(),f'{k}_sd':series.std(ddof=1),
                         **{f'{k}_q{q}':series.quantile(q/100) for q in (10,25,50,75,90)}})
        bnet = group.delta_product_net
        for name in ('T','estimation_error'):
            common = group[['delta_product_net',name]].dropna()
            diag[f'bnet_{name}_pairs'] = len(common)
            diag[f'bnet_{name}_corr'] = common.delta_product_net.corr(common[name]) if len(common)>2 else np.nan
        pair = group[['T','oracle_fixed_tau_T']].dropna()
        diag.update(oracle_pairs=len(pair),estimated_mean_T_paired=pair['T'].mean(),
                    oracle_mean_T=pair.oracle_fixed_tau_T.mean(),
                    estimated_minus_oracle_mean_T=(pair['T']-pair.oracle_fixed_tau_T).mean())
        nuisance.append(diag)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary.error_id==0,['dgp_id','method','coverage_pct']].rename(
        columns={'coverage_pct':'gaussian_coverage_pct'})
    summary = summary.merge(baseline,on=['dgp_id','method'],validate='many_to_one')
    summary['deterioration_pp'] = summary.gaussian_coverage_pct-summary.coverage_pct
    return summary,pd.DataFrame(nuisance)


def pair_results(raw,summary):
    pair = []
    for (d,e),group in raw.groupby(['dgp_id','error_id']):
        sub = group.set_index(['method','replication_id'])
        rows = summary[(summary.dgp_id==d)&(summary.error_id==e)&summary.method.ne('S')]
        best = rows.sort_values(['coverage_pct','rmse','method'],ascending=[False,True,True]).iloc[0].method
        nominal = rows.assign(distance=(rows.coverage_pct-95).abs()).sort_values(
            ['distance','rmse','method']).iloc[0].method
        rmse_best = rows.sort_values(['rmse','method']).iloc[0].method
        s = sub.loc['S']
        for ref in METHODS[:-1]:
            b = sub.loc[ref]
            combined = s[['T','estimation_error','data_hash','fold_hash']].join(
                b[['T','estimation_error','data_hash','fold_hash']],lsuffix='_S',rsuffix='_ref',
                how='inner',validate='one_to_one')
            assert len(combined)==len(s)==len(b)
            for key in ('data_hash','fold_hash'):
                assert combined[f'{key}_S'].equals(combined[f'{key}_ref'])
            T = combined[['T_S','T_ref']].dropna()
            point = combined[['estimation_error_S','estimation_error_ref']].dropna()
            yes = T.abs()<=CRITICAL
            difference = 100*(yes.T_S.astype(float)-yes.T_ref.astype(float))
            rmse_s,rmse_b = np.sqrt((point**2).mean())
            pair.append(dict(dgp_id=d,dgp=CORE_DGP[d],error_id=e,error=ERROR_LABELS[e],method='S',
                reference=ref,is_highest_coverage_benchmark=ref==best,
                is_closest_nominal_benchmark=ref==nominal,is_lowest_rmse_benchmark=ref==rmse_best,
                test_pairs=len(T),point_pairs=len(point),
                coverage_S_pct=100*yes.T_S.mean(),coverage_ref_pct=100*yes.T_ref.mean(),
                coverage_diff_pp=difference.mean(),
                coverage_diff_MCSE_pp=difference.std(ddof=1)/np.sqrt(len(T)) if len(T)>1 else np.nan,
                bias_S=point.estimation_error_S.mean(),bias_ref=point.estimation_error_ref.mean(),
                bias_diff=(point.estimation_error_S-point.estimation_error_ref).mean(),
                rmse_S=rmse_s,rmse_ref=rmse_b,rmse_diff=rmse_s-rmse_b,
                rmse_ratio=rmse_s/rmse_b if rmse_b>0 else np.nan))
    return pd.DataFrame(pair)


def figure_panels(destination, table, filename, title, panels):
    configure_plot_style()
    fig,axes = plt.subplots(len(panels),2,squeeze=False,figsize=(13.2,3.15*len(panels)+.75),
                            layout='constrained')
    handles = {}
    for row,(family,metric,label,reference) in enumerate(panels):
        available = set(table.error_id)
        errors = [e for e in FAMILIES[family] if e in available]
        for d in range(2):
            ax = axes[row,d]
            for method in METHODS:
                part = table[(table.dgp_id==d)&table.method.eq(method)].set_index('error_id').reindex(errors)
                values = part[metric].to_numpy(dtype=float)
                if not np.isfinite(values).any():
                    continue
                is_transformed = metric=='y_nuisance_mse' and method=='M3'
                line, = ax.plot(range(len(errors)),values,marker='o',markersize=4 if method=='S' else 3,
                                lw=2.15 if method=='S' else 1.35,color=COLORS[method],
                                ls='--' if is_transformed else '-')
                handles.setdefault(method,line)
                if method=='S' and metric in ('coverage_pct','deterioration_pp'):
                    for x,y in enumerate(values):
                        if np.isfinite(y):
                            ax.annotate(f'{y:.1f}',(x,y),xytext=(0,6),textcoords='offset points',
                                        ha='center',fontsize=8,color=COLORS[method])
                if method=='S' and metric=='delta_product_net_mean':
                    ax.fill_between(range(len(errors)),part.delta_product_net_q10,part.delta_product_net_q90,
                                    color=COLORS['S'],alpha=.13)
            if reference is not None:
                ax.axhline(reference,ls=':',color='#636f81',lw=1)
            ax.set_xticks(range(len(errors)),[ERROR_LABELS[e] for e in errors],fontsize=8.5)
            ax.set_title(f'{CORE_DGP[d]} · {family}',fontsize=10.5)
            ax.set_ylabel(label)
            ax.grid(axis='y',alpha=.2)
            ax.margins(x=.05,y=.2 if metric in ('coverage_pct','deterioration_pp') else .12)
    fig.suptitle(title,fontsize=17)
    legend = '虚线 M3 Y MSE 使用变换结果；S 净交叉项阴影为 10%–90% 分位数' if filename=='nuisance_diagnostics.png' else None
    fig.legend([handles[m] for m in METHODS if m in handles],[LABELS[m] for m in METHODS if m in handles],
               loc='outside lower center',ncol=4,frameon=False,title=legend)
    fig.savefig(destination/filename,dpi=220)
    plt.close(fig)


def export_figures(destination,summary,nuisance):
    for family,suffix in [('Student-t','student_t')]:
        figure_panels(destination,summary,f'coverage_{suffix}.png',
                      f'{family} · Score-test 覆盖率',[(family,'coverage_pct','覆盖率 (%)',95)])
        figure_panels(destination,summary,f'deterioration_{suffix}.png',
                      f'{family} · 相对 Gaussian 的覆盖下降',
                      [(family,'deterioration_pp','覆盖下降 (百分点)',0)])
        figure_panels(destination,summary,f'bias_rmse_{suffix}.png',
                      f'{family} · 点估计 Bias 与 RMSE',[(family,'bias','Bias',0),(family,'rmse','RMSE',None)])
    figure_panels(destination,summary,'score_diagnostics.png','检验统计量位置与离散程度',
        [(f,k,label,target) for f in FAMILIES for k,label,target in [('mean_T','Mean(T)',0),('sd_T','SD(T)',1)]])
    figure_panels(destination,nuisance,'nuisance_diagnostics.png','第一阶段预测误差与净交叉项',
        [(f,k,label,target) for f in FAMILIES for k,label,target in
         [('y_nuisance_mse','Y OOF MSE',None),('d_nuisance_mse','D OOF MSE',None),
          ('delta_product_net_mean','净交叉项均值',0)]])


def findings(summary,paired,nuisance,mode):
    lines = []
    top = paired[paired.is_highest_coverage_benchmark]
    lowest = paired[paired.is_lowest_rmse_benchmark]
    for family,errors in FAMILIES.items():
        for d in range(2):
            s = summary[(summary.dgp_id==d)&(summary.error_id.isin(errors[1:]))&summary.method.eq('S')]
            if s.empty:
                continue
            benchmark = top[(top.dgp_id==d)&top.error_id.isin(s.error_id)]
            rmse_comparison = lowest[(lowest.dgp_id==d)&lowest.error_id.isin(s.error_id)]
            weak = s.loc[s.coverage_pct.idxmin()]
            higher = int((benchmark.coverage_diff_pp>=0).sum())
            lower_rmse = int((rmse_comparison.rmse_ratio<1).sum())
            lines.append(f'**{CORE_DGP[d]} · {family}**。S 的覆盖率范围 '
                         f'{s.coverage_pct.min():.1f}%–{s.coverage_pct.max():.1f}%，最低出现在 {weak.error}。'
                         f'对最高覆盖的原始基准，S 在 {higher}/{len(s)} 个场景的配对覆盖差为非负，'
                         f'差值范围 {benchmark.coverage_diff_pp.min():+.1f} 至 {benchmark.coverage_diff_pp.max():+.1f} 个百分点。'
                         f'对最低 RMSE 的原始基准，S 在 {lower_rmse}/{len(s)} 个场景的配对 RMSE 更小。'
                         f'S 的 Gaussian 至重尾覆盖下降范围为 {s.deterioration_pp.min():+.1f} 至 '
                         f'{s.deterioration_pp.max():+.1f} 个百分点。')
            for ref in METHODS[:-1]:
                comp = paired[(paired.dgp_id==d)&(paired.error_id.isin(s.error_id))&paired.reference.eq(ref)]
                merged = summary[(summary.dgp_id==d)&(summary.error_id.isin(s.error_id))&summary.method.eq(ref)]
                delta = s.set_index('error_id').deterioration_pp-merged.set_index('error_id').deterioration_pp
                lines.append(f'{CORE_DGP[d]} · {family}，相对 {ref}，配对覆盖差为 '
                             f'{comp.coverage_diff_pp.min():+.1f} 至 {comp.coverage_diff_pp.max():+.1f} 个百分点，'
                             f'RMSE 比值为 {comp.rmse_ratio.min():.3f}–{comp.rmse_ratio.max():.3f}，'
                             f'覆盖下降差为 {delta.min():+.1f} 至 {delta.max():+.1f} 个百分点。')
    for d in range(2):
        g = summary[(summary.dgp_id==d)&summary.error_id.eq(0)&summary.method.eq('S')].iloc[0]
        lines.append(f'**{CORE_DGP[d]} · Gaussian**。S 的覆盖率 {g.coverage_pct:.1f}%，Bias {g.bias:+.5f}，'
                     f'RMSE {g.rmse:.5f}，Mean(T) {g.mean_T:+.3f}，SD(T) {g.sd_T:.3f}。')
    for ref in ('M1','M2'):
        metrics = ['dgp_id','error_id','delta_product_net_mean','bnet_T_corr']
        s = nuisance.loc[nuisance.method=='S',metrics]
        b = nuisance.loc[nuisance.method==ref,metrics]
        g = s.merge(b,on=metrics[:2],suffixes=('_S','_ref'),validate='one_to_one')
        smaller = g.delta_product_net_mean_S.abs()<g.delta_product_net_mean_ref.abs()
        a = summary[summary.method.eq('S')].merge(summary[summary.method.eq(ref)],on=metrics[:2],
                    suffixes=('_S','_ref'),validate='one_to_one')
        cov_and_rmse = a.coverage_pct_S.ge(a.coverage_pct_ref)&a.rmse_S.lt(a.rmse_ref)
        larger_mse = a.y_nuisance_mse_S.gt(a.y_nuisance_mse_ref)|a.d_nuisance_mse_S.gt(a.d_nuisance_mse_ref)
        lines.append(f'**误差机制 · 相对 {ref}**。S 在 {int(smaller.sum())}/{len(g)} 个场景的净交叉项均值更接近零。'
                     f'有 {int((cov_and_rmse&larger_mse).sum())} 个场景至少一侧 OOF MSE 更高，'
                     '但覆盖率不低且 RMSE 更小。')
    if mode=='full':
        direction = summary[['dgp_id','error_id','method','mean_T','bias']].merge(
            nuisance[['dgp_id','error_id','method','delta_product_net_mean','bnet_T_corr','oracle_pairs',
                       'estimated_minus_oracle_mean_T']],on=['dgp_id','error_id','method'],validate='one_to_one')
        s = direction[direction.method.eq('S')]
        for family,errors in FAMILIES.items():
            g = s[s.error_id.isin(errors[1:])]
            lines.append(f'**{family} 诊断**。S 的场景内净交叉项与 T 的相关系数中位数为 '
                         f'{g.bnet_T_corr.median():+.3f}，小样本 Oracle 对照下，估计 nuisance 与 Oracle 的 '
                         f'Mean(T) 差的场景中位数为 {g.estimated_minus_oracle_mean_T.median():+.3f}。'
                         '未截断误差乘积中的净交叉项可用于解释部分偏移。Huber 截断、位置调整和标准化也影响最终结果。')
        srows = summary[summary.method.eq('S')]
        lines.append(f'**稳定性与下一步**。S 的点估计有效 {int(srows.point_n.sum()):,}/{int(srows.planned_n.sum()):,}，'
                     f'主检验有效 {int(srows.test_n.sum()):,}/{int(srows.planned_n.sum()):,}。'
                     '四方法的逐场景结果为进一步研究提供了经验依据。p=10、其他 DGP 及更大的 n 可以检验这些表现的可迁移性，'
                     '并区分有限样本误差与更持久的偏移。理论工作需要推导位置调整与学习误差共同作用下的得分性质。'
                     '本次模拟无法确立一般的 Neyman 正交性或更高维的推断有效性。')
    return '## Full Simulation Findings\n\n'+'\n\n'.join(lines)


def export(destination,db,loaded,config):
    raw,failures = flatten_results(db,loaded,config)
    summary,nuisance = summarize_results(raw)
    paired = pair_results(raw,summary)
    assert len(summary)==2*len(config['errors'])*4 and len(paired)==2*len(config['errors'])*3
    assert len(raw)==len(loaded)*4
    for name,table in [('summary',summary),('replication_results',raw),('paired_comparisons',paired),
                       ('nuisance_diagnostics',nuisance),('failures',failures)]:
        atomic_csv(destination/f'full_{name}.csv',table)
    export_figures(destination,summary,nuisance)
    note = findings(summary,paired,nuisance,config['mode'])
    intro = (
        f'{config["mode"]} · 每场景 {config["R"]} 次 · {len(loaded):,} 个数据集 · {len(raw):,} 条方法记录。'
        'Score-test 覆盖率按有效检验计算，临界值 1.96，样本标准差使用 ddof=1。'
        'S 用调整后的得分标准差。位置方程在原假设下的样本均值为零，因此调整前后的得分均值数值一致。'
        'S 的 naive 检验仅用于诊断。M3 的 Y 诊断使用变换结果的条件均值，'
        'M2 学习条件中位数，表中使用条件均值作为共同诊断参照。'
        '配对分析分开使用点估计、检验的共同有效样本。覆盖最高的基准和最接近 95% 的基准分别标记。'
        '净交叉项的原始值见逐次结果，均值、标准差、分位数和相关系数见 nuisance 诊断。'
        '每场景预先选取每 100 次中的第一组做 Oracle 诊断，S 的 Oracle 使用该次实际拟合的阈值。'
        '失败按方法和步骤保留，正式汇总使用断点中有效的结果。'
    )
    best = paired[paired.is_highest_coverage_benchmark]
    tables = [('完整汇总',summary),('与覆盖最高的原始基准的逐场景差距',best),
              ('全部配对比较',paired),('第一阶段及 Oracle 诊断',nuisance),('失败清单',failures)]
    sections = ''.join(f'<h2>{title}</h2><div class="scroll">{table.to_html(index=False,na_rep="—",float_format=lambda x:f"{x:.6g}")}</div>'
                       for title,table in tables)
    body = ''.join('<p>'+html.escape(line.replace('**',''))+'</p>'
                   for line in note.split('\n\n')[1:])
    markup = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Spline-Ridge DML 全量模拟</title><style>
body{{font:15px/1.65 "Microsoft YaHei",sans-serif;margin:2.4rem;background:#f3f6fa;color:#152a3b}}
h1{{color:#175968}} p{{max-width:112ch}} h2{{margin-top:2rem}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;white-space:nowrap;background:white;font-size:13px}}
th,td{{border:1px solid #d5e1eb;text-align:right;padding:6px 8px}} th{{background:#1a4f67;color:white}}
tbody tr:nth-child(even){{background:#edf5fa}}
</style><h1>Spline-Ridge 与均值位置 Huber-DML</h1><p>{html.escape(intro)}</p>
<h2>Full Simulation Findings</h2>{body}{sections}</html>"""
    atomic_text(destination/'full_report.html',markup)
    return dict(summary=summary,paired=paired,nuisance=nuisance,findings=note,
                data_n=len(loaded),point_n=int(summary.point_n.sum()),test_n=int(summary.test_n.sum()),
                failure_events=len(failures))


def replay_sources(config):
    """复放历史 Pilot 的共同任务，正式结果仍逐组重新拟合。"""
    pilot = ROOT.parent/'Spline_Ridge_DML_Pilot'
    cfg = json.loads((pilot/'.checkpoints'/'full_config.json').read_text(encoding='utf-8'))
    assert sha((pilot/'Spline_Ridge_DML_Pilot.py').read_bytes())==cfg['code_hash']
    folder = pilot/'.checkpoints'/'full'/cfg['config_hash'][:16]
    replay = []
    with threadpool_limits(limits=1):
        for task in [(d,7,r) for d in range(2) for r in (0,39,99,199)]:
            d,e,r = task
            old = json.loads((folder/f'd{d}_e{e}_r{r:03d}.json').read_text(encoding='utf-8'))
            new = worker_packet(config,task,('S',))
            validate_packet(new,config,task,('S',))
            assert all(new[k]==old[k] for k in ('data_hash','fold_hash'))
            saved = next(row for row in old['rows'] if row['method']=='S')
            now = new['rows']['S']
            for k,old_key in [('theta_hat','theta'),('T','T'),('tau','tau'),('naive_T','naive_T'),
                              ('score_sd','score_sd')]:
                assert np.isclose(now[k],saved[old_key],rtol=1e-12,atol=1e-12),(task,k)
            for k in ('alpha_d','alpha_y'):
                assert now[k]==saved[k]
            replay.append(dict(task=list(task),theta=now['theta_hat'],T=now['T'],data_hash=new['data_hash'],
                               fold_hash=new['fold_hash'],alpha_d=now['alpha_d'],alpha_y=now['alpha_y']))
    return replay


def smoke_run(destination):
    config = config_for('smoke')
    with run_lock(destination):
        append_log(destination,'开始 Smoke')
        with checkpoint_db(destination,config) as db:
            loaded = collect(destination,db,config)
            assert len(loaded)==8 and len({p['worker_pid'] for _,p in loaded.values()})==WORKERS
            assert all(not failed_methods(p) for _,p in loaded.values()), 'Smoke 存在无效的点估计或主检验'
        with checkpoint_db(destination,config) as db:
            restored = load_packets(db,config)
            assert restored==loaded, '断点回读不一致'
            same = collect(destination,db,config)
            assert restored==same, 'Smoke 续跑重复计算了已完成任务'
            result = export(destination,db,restored,config)
        replay = replay_sources(config_for('full'))
        report = dict(passed=True,workers=WORKERS,data_n=result['data_n'],point_n=result['point_n'],
                      test_n=result['test_n'],resume_verified=True,pilot_replay=replay)
        atomic_json(destination/'validation.json',report)
        save_progress(destination,config,restored,8,8,0,8,'complete')
        append_log(destination,'Smoke 通过，8 组数据和 32 条方法记录，历史 Pilot 8/8 一致')
        return report


def full_run(destination,retry_failed=False):
    config = config_for('full')
    started = time.perf_counter()
    with run_lock(destination):
        atomic_json(destination/'configuration.json',config)
        append_log(destination,'正式实验开始')
        with checkpoint_db(destination,config) as db:
            loaded = collect(destination,db,config,retry_failed)
            assert len(loaded)==16_000
            previous = json.loads((destination/'progress.json').read_text(encoding='utf-8'))
            atomic_json(destination/'progress.json',dict(previous,status='exporting'))
            result = export(destination,db,loaded,config)
        report = {k:result[k] for k in ('data_n','point_n','test_n','failure_events')}
        report.update(config_hash=config['config_hash'],seconds=time.perf_counter()-started)
        atomic_json(destination/'completed.json',report)
        atomic_json(destination/'progress.json',dict(previous,status='complete',completed=len(loaded),eta_seconds=0))
        append_log(destination,f'正式完成，{report}')
        return result


def execute_notebook(path):
    """无人值守执行 Notebook 自身，各单元完成后原子保存执行状态。"""
    import nbformat
    from nbclient import NotebookClient

    path = Path(path)
    notebook = nbformat.read(path,as_version=4)
    def persist(**_):
        atomic_text(path,nbformat.writes(notebook))
    client = NotebookClient(notebook,kernel_name='environment2',timeout=None,
                            resources={'metadata':{'path':str(path.parent)}},
                            on_cell_executed=persist)
    print(f'开始执行 {path}',flush=True)
    try:
        client.execute()
    finally:
        persist()
    print('Notebook 已执行并保存。',flush=True)


if __name__=='__main__':
    multiprocessing.freeze_support()
    if len(sys.argv)==3 and sys.argv[1]=='--execute-notebook':
        execute_notebook(sys.argv[2])
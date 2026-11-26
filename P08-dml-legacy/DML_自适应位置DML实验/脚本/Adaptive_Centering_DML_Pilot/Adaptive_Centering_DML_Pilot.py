# -*- coding: utf-8 -*-
"""自适应位置 DML 实验。默认先自检，再执行 R=200；支持断点续跑。"""
import sys
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.patches import Patch
import html



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

REQUIRED_VERSIONS = {'numpy':'2.3.5','scipy':'1.17.1','scikit-learn':'1.7.2','joblib':'1.5.3'}

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

SOURCE_HASHES = {'Lognormal_Pilot_R200.py': '8b8ac81045ffd125c941e77c1f1c872e51bcbbf1112833382f368b59dff26a91', 'DML_p5_StudentT_5Methods_ScoreTest.ipynb': '804f7148c0c88e3cd5ad8b7f5406ccf027c63153f64d6a1e7cb7131b4d2f5e6a', 'Adaptive_Lasso_DML.py': 'e843649024ecca7bc9577f18586de620e8e7827374bf7d50db14f319860c7665'}

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

def standardized_lognormal(rng,sigma,n):
    """总体均值中心化、总体方差标准化。"""
    mu=math.exp(sigma*sigma/2)
    return (np.exp(sigma*rng.normal(size=n))-mu)/(mu*math.sqrt(math.expm1(sigma*sigma)))

def init_worker(barrier):
    global worker_limits
    worker_limits=threadpool_limits(limits=1)
    if any(lib['num_threads']!=1 for lib in threadpool_info()):
        raise RuntimeError('Worker 数值计算库线程数不为 1')
    if barrier is not None:
        barrier.wait()

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / '.checkpoints'
PREFIX = 'Adaptive_Centering_'
METHODS = ('A', 'B', 'C', 'D', 'E')
CENTERED = METHODS[:3]
METHOD_NAMES = dict(A='A 均值 Huber', B='B 中位数 Huber', C='C Huber M位置',
                    D='D Dukes', E='E LAD-DML')
ERROR_LABELS = {7:'Student-t(2.1)', 14:'Lognormal(2)'}
LOGNORMAL_SIGMAS = {14:2.0}
WORKERS = 8
ORACLE_N = 500_000
DERIVATIVE_STEPS = (1e-3, 2.5e-4)
PAIRS = (('A','B'), ('A','C'), ('B','C'),
         ('A','D'), ('B','D'), ('C','D'), ('A','E'), ('B','E'), ('C','E'))


def config_for(mode):
    config = dict(mode=mode, n=1000, R=2 if mode == 'smoke' else 200,
                  p=P, K=K, base_seed=BASE_SEED,
                  spec=dict(kernel=DEFAULT_KERNEL_CONFIG, m3_solver=M3_SOLVER),
                  dependencies={name: metadata.version(name) for name in REQUIRED_VERSIONS},
                  python=platform.python_version(),
                  code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config['config_hash'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config


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


def adaptive_side(features, target, folds, config, dgp, error, rep, robust, label):
    # 与原 M5 逐折使用同一个种子、同一特征和同一套内部 CV。
    prediction = np.empty(len(target))
    for k, (train, test) in enumerate(folds):
        splits = list(KFold(INNER_K, shuffle=True,
            random_state=seed(config,dgp,error,rep,50,k,0)).split(train))
        try:
            prediction[test], _ = adaptive_prediction(
                features[train], target[train], features[test], splits, robust)
        except Exception as exc:
            fail = NumericalFailure('adaptive_fit_failed', f'{label} fold={k}: {type(exc).__name__}: {exc}')
            fail.fold = k
            raise fail from exc
    if not np.isfinite(prediction).all():
        raise ArithmeticError(f'{label}_nonfinite_prediction')
    return prediction


def dukes_at_candidate(observed,folds,config,dgp,error,rep,g_hat,candidate):
    """只替换 transformed outcome 学习器，保留 corrected score 及求解逻辑。"""
    X,D,Y = observed
    m_hat = adaptive_side(feature_dictionary(X),Y-candidate*D,folds,
                          config,dgp,error,rep,True,'Dukes transformed Y')
    return dukes_from_initial(D,Y,candidate,g_hat,m_hat,folds,config['spec']['kernel']), m_hat


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


def oracle_replicate(method, observed, dgp, tau):
    X,D,Y = observed
    m,g = nuisance_functions(X,DGP_NAMES[dgp])
    w = D-m
    residual = Y-THETA_NULL*D-g
    return {f'oracle_{key}':value for key,value in
            centered_test(method,w,residual,tau).items()
            if key in ('T','score_mean','score_sd','a','psi_mean')}


def dukes_estimate(observed, folds, config, task, m, w, u):
    objective = DukesPointObjective(observed,folds,config,*task,m)
    theta, diag = dukes_point(objective,ols_point(w,u))
    # 保留原 OLS 残差锚，以统一的一阶段预测构造。
    if not dukes_solution_valid(theta,diag,len(w),config['spec']['m3_solver']):
        raise ArithmeticError('Dukes_point_validation_failed')
    return dict(theta=theta, point_score_residual=abs(diag['root_score_mean']),
                point_equation_T=diag['point_equation_T'], point_evaluations=diag['evaluations'],
                point_solution_type=diag['solution_type'])


def dukes_null(observed, folds, config, task, m):
    out, transformed = dukes_at_candidate(observed,folds,config,*task,m,THETA_NULL)
    test = standardize_score(out['scores'],CRITICAL)
    return dict(T=test['T'], score_mean=test['score_mean'], score_sd=test['score_sd'],
                transformed_prediction=transformed)


def worker_packet(config, task):
    start = time.perf_counter()
    dgp,error,rep = task
    generator = generate_observed if error == 7 else generate_pilot_observed
    observed, folds, _, audit = generator(config,*task)
    X,D,Y = observed
    F = feature_dictionary(X)
    rows = {method:dict(method=method,dgp_id=dgp,error_id=error,replication_id=rep,
                        failures=[]) for method in METHODS}
    shared = dict(failures=[])
    record_stage(shared,'nuisance_D',lambda:dict(m=adaptive_side(
        F,D,folds,config,*task,False,'D')))
    m = shared.get('m')
    if m is not None:
        record_stage(shared,'nuisance_Y',lambda:dict(ell=adaptive_side(
            F,Y,folds,config,*task,True,'Y')))
    ell = shared.get('ell')
    tau = None
    if m is not None and ell is not None:
        w,u = D-m,Y-ell
        nuisance = nuisance_diagnostics(observed,np.array([m,ell]),dgp)
        threshold = dict(failures=[])
        def estimate_threshold():
            anchor, value, scale = huber_threshold(w,u)
            return dict(anchor=anchor,tau=value,scale=scale)
        record_stage(threshold,'threshold',estimate_threshold)
        tau = threshold.get('tau')
        for method in (*CENTERED,'E'):
            rows[method].update(nuisance)
        for method in CENTERED:
            rows[method]['failures'].extend(threshold['failures'])
        if tau is not None:
            e = u-THETA_NULL*w
            for method in CENTERED:
                row = rows[method]
                record_stage(row,'point',lambda: centered_point(method,w,u,threshold['anchor'],tau))
                record_stage(row,'test',lambda: centered_test(method,w,e,tau))
                record_stage(row,'naive',lambda: naive_centered_test(method,w,e,tau))
                record_stage(row,'oracle',lambda: oracle_replicate(method,observed,dgp,tau))
        row = rows['E']
        record_stage(row,'point',lambda:dict(theta=lad_point(w,u)[0]))
        record_stage(row,'test',lambda:{key:value for key,value in
            standardize_score(residual_scores('M2',w,u,THETA_NULL),CRITICAL).items()
            if key in ('T','score_mean','score_sd')})
    for method in (*CENTERED,'E'):
        rows[method]['failures'].extend(shared['failures'])
    row = rows['D']
    if m is None:
        row['failures'].extend(shared['failures'])
    else:
        if ell is not None:
            record_stage(row,'point',lambda:dukes_estimate(observed,folds,config,task,m,D-m,Y-ell))
        else:
            row['failures'].extend(shared['failures'])
        record_stage(row,'test',lambda:dukes_null(observed,folds,config,task,m))
        transformed = row.pop('transformed_prediction',None)
        m_true,g_true = nuisance_functions(X,DGP_NAMES[dgp])
        row.update(d_nuisance_mse=float(np.mean((m-m_true)**2)),
                   d_observed_mse=float(np.mean((m-D)**2)))
        if transformed is not None:
            row.update(y_nuisance_mse=float(np.mean((transformed-g_true)**2)),
                       y_observed_mse=float(np.mean((transformed-(Y-THETA_NULL*D))**2)))
    return dict(task=list(task), rows=list(rows.values()), config_hash=config['config_hash'],
                data_hash=audit['data_hash'],fold_hash=audit['fold_hash'],
                prediction_hash=sha(np.array([m,ell]).tobytes()) if ell is not None else None,
                tau=tau, worker_pid=os.getpid(), seconds=time.perf_counter()-start)


def analytic_psi(error, a, tau):
    """在已知误差分布下精确积分裁剪后的 psi，仅用于 oracle。"""
    if error == 7:
        from scipy.stats import t
        unit = math.sqrt(.1/2.1)
        df = 2.1
        lower, upper = (a-tau)/unit, (a+tau)/unit
        pl, pu = t.cdf([lower,upper],df)
        def partial(x):
            return -unit*(df+x*x)/(df-1)*t.pdf(x,df)
        truncated = partial(upper)-partial(lower)
    else:
        sigma = LOGNORMAL_SIGMAS[error]
        mu = math.exp(sigma**2/2)
        sd = mu*math.sqrt(math.expm1(sigma**2))
        def normal_boundary(x):
            value = mu+sd*x
            return math.log(value)/sigma if value>0 else -np.inf
        lower,upper = map(normal_boundary,(a-tau,a+tau))
        pl,pu = norm.cdf([lower,upper])
        truncated = mu*(norm.cdf(upper-sigma)-norm.cdf(lower-sigma)-pu+pl)/sd
    return float(truncated-a*(pu-pl)+tau*(1-pu-pl))


def population_position(method,error,tau):
    if method == 'A' or error == 7:
        return 0.
    mu = math.exp(LOGNORMAL_SIGMAS[error]**2/2)
    sd = mu*math.sqrt(math.expm1(LOGNORMAL_SIGMAS[error]**2))
    median = (1-mu)/sd
    if method == 'B':
        return median
    return brentq(lambda a:analytic_psi(error,a,tau), median-tau, tau+abs(median))


def oracle_population(packets, config):
    # 每组固定为该组实际阈值的中位数。独立大样本用于位置，另一独立样本用于差分。
    output = []
    for dgp in range(2):
        for error in ERROR_LABELS:
            thresholds = [p['tau'] for p in packets if p['task'][:2]==[dgp,error] and p['tau'] is not None]
            tau = float(np.median(thresholds))
            rng = np.random.default_rng(seed(config,dgp,error,0,201))
            evaluate = np.random.default_rng(seed(config,dgp,error,0,202))
            def draw(generator, n):
                if error == 7:
                    return standardized_student_t(generator,2.1,n)
                return standardized_lognormal(generator,2.,n)
            calibration_e = draw(rng,ORACLE_N)
            calibration_x1 = rng.normal(size=ORACLE_N)
            e = draw(evaluate,ORACLE_N)
            x1 = evaluate.normal(size=ORACLE_N)
            # 在相同 X、误差上配对 +V、-V 降低数值诊断的 Monte Carlo 方差。
            # 对 V 的积分与原 DGP 一致，保留有限样本下 X1 和误差的抽样波动。
            for method in CENTERED:
                a = position(method,calibration_e,tau)
                a_true = population_position(method,error,tau)
                empirical_psi = float(np.mean(np.clip(calibration_e-a,-tau,tau)))
                psi_theory = analytic_psi(error,a_true,tau)
                for direction, h, h_cal, mean_h in (
                    ('constant',np.ones(ORACLE_N),np.ones(ORACLE_N),1.),
                    ('X1',x1,calibration_x1,0.)):
                    for step in DERIVATIVE_STEPS:
                        shift = THETA_NULL*step*h
                        plus = position(method,calibration_e+THETA_NULL*step*h_cal,tau)
                        minus = position(method,calibration_e-THETA_NULL*step*h_cal,tau)
                        def derivative(a_plus,a_minus):
                            # 先对 +V 和 -V 的得分取均值，V 项逐对抵消。
                            return -.5*h*(np.clip(e+shift-a_plus,-tau,tau)
                                           + np.clip(e-shift-a_minus,-tau,tau))
                        fixed = derivative(a,a)
                        profiled = derivative(plus,minus)
                        theory = -mean_h*psi_theory
                        fixed_mean, profile_mean = float(fixed.mean()), float(profiled.mean())
                        output.append(dict(diagnostic='gateaux',dgp_id=dgp,error_id=error,
                            method=method,direction=direction,step=step,tau=tau,
                            calibration_n=ORACLE_N,evaluation_pairs=ORACLE_N,
                            oracle_a=a,theory_a=a_true,oracle_psi_mean=empirical_psi,
                            theory_psi_mean=psi_theory,partial_numeric=fixed_mean,
                            profile_numeric=profile_mean,
                            partial_eval_MCSE=float(fixed.std(ddof=1)/np.sqrt(ORACLE_N)),
                            profile_eval_MCSE=float(profiled.std(ddof=1)/np.sqrt(ORACLE_N)),
                            partial_theory=theory,profile_theory=theory,
                            partial_minus_theory=fixed_mean-theory,
                            profile_minus_theory=profile_mean-theory,
                            numeric_a_derivative=(plus-minus)/(2*step),
                            theory_a_derivative=THETA_NULL*mean_h))
    return output


def collect(config, smoke):
    folder = CHECKPOINTS / config['mode'] / config['config_hash'][:16]
    tasks = [(d,e,r) for r in range(config['R']) for d in range(2) for e in ERROR_LABELS]
    packets, pending = {}, {}
    for task in tasks:
        path = folder / f'd{task[0]}_e{task[1]}_r{task[2]:03d}.json'
        if path.exists():
            packet = json.loads(path.read_text(encoding='utf-8'))
            if packet['task']!=list(task) or packet['config_hash']!=config['config_hash']:
                raise RuntimeError(f'断点不匹配 {path}')
            packets[task] = packet
        else:
            pending[task] = path
    if pending:
        context = multiprocessing.get_context('spawn')
        barrier = context.Barrier(WORKERS, timeout=180) if smoke else None
        with ProcessPoolExecutor(max_workers=WORKERS,mp_context=context,
                                 initializer=init_worker,initargs=(barrier,)) as executor:
            futures = {executor.submit(worker_packet,config,task):task for task in pending}
            with tqdm(total=len(tasks),initial=len(packets),desc='Smoke' if smoke else '正式 R200',
                      unit='组',mininterval=5) as progress:
                for future in as_completed(futures):
                    task = futures[future]
                    packet = future.result()
                    atomic_json(pending[task],packet)
                    packets[task] = packet
                    progress.update()
    return [packets[task] for task in tasks]


def flatten(packets, config, smoke):
    rows, failures = [], []
    for packet in packets:
        assert packet['config_hash']==config['config_hash']
        assert [row['method'] for row in packet['rows']]==list(METHODS)
        for row in packet['rows']:
            common = {key:row[key] for key in ('dgp_id','error_id','replication_id','method')}
            assert tuple(common[key] for key in ('dgp_id','error_id','replication_id'))==tuple(packet['task'])
            for failure in row['failures']:
                failures.append(dict(common,**failure))
            data = {key:value for key,value in row.items() if key!='failures'}
            assert (row.get('theta') is not None and row.get('T') is not None) or row['failures']
            if row['method'] in CENTERED:
                assert (row.get('naive_T') is not None and row.get('oracle_T') is not None) or row['failures']
                if row['method']=='C' and row.get('T') is not None:
                    assert abs(row['psi_mean']) < 1e-10
                if row.get('T') is not None and row.get('naive_T') is not None:
                    assert abs(row['score_mean']-row['naive_score_mean']) < 1e-10
            if smoke:
                assert not row['failures'], (common,row['failures'])
            rows.append(data)
    frame = pd.DataFrame(rows).set_index(['dgp_id','error_id','replication_id','method']).sort_index()
    assert frame.index.is_unique and len(frame)==4*config['R']*5
    return frame, pd.DataFrame(failures,columns=['dgp_id','error_id','replication_id','method',
                                                'stage','type','fold','message'])


def describe(frame):
    output = []
    for (dgp,error,method),group in frame.groupby(level=['dgp_id','error_id','method']):
        theta,T = group.theta.dropna(), group['T'].dropna()
        naive = group.naive_T.dropna()
        err = theta-THETA_TRUE
        delta = (group['T']-group.oracle_T).dropna()
        output.append(dict(dgp_id=dgp,error_id=error,error=ERROR_LABELS[error],
            method=method, planned=len(group),point_n=len(theta),test_n=len(T),
            point_fail=len(group)-len(theta),test_fail=len(group)-len(T),
            any_point_or_test_fail=int((group.theta.isna()|group['T'].isna()).sum()),
            coverage=float((T.abs()<=CRITICAL).mean()),naive_n=len(naive),
            naive_coverage=float((naive.abs()<=CRITICAL).mean()),bias=float(err.mean()),
            rmse=float(np.sqrt(np.mean(err**2))),mean_T=float(T.mean()),sd_T=float(T.std(ddof=1)),
            left_reject=float((T < -CRITICAL).mean()),right_reject=float((T > CRITICAL).mean()),
            mean_a=group.a.mean(),mean_psi=group.psi_mean.mean(),mean_lambda=group.location_lambda.mean(),
            y_nuisance_mse=group.y_nuisance_mse.mean(),d_nuisance_mse=group.d_nuisance_mse.mean(),
            y_observed_mse=group.y_observed_mse.mean(),d_observed_mse=group.d_observed_mse.mean(),
            y_oof_n=int(group.y_nuisance_mse.notna().sum()), d_oof_n=int(group.d_nuisance_mse.notna().sum()),
            oracle_comparison_n=len(delta),mean_T_estimated_matched=(group.oracle_T+delta).mean(),
            mean_T_oracle_matched=group.oracle_T.loc[delta.index].mean(),
            mean_T_estimated_minus_oracle=delta.mean()))
    return pd.DataFrame(output)


def paired(frame, baseline=None):
    rows = []
    for (dgp,error),group in frame.groupby(level=['dgp_id','error_id']):
        group = group.droplevel(['dgp_id','error_id'])
        comparisons = list(PAIRS)
        if baseline is not None:
            comparisons.extend((method,'M5历史') for method in CENTERED)
        for left,right in comparisons:
            a = group.xs(left,level='method')
            b = (baseline.xs((dgp,error),level=['dgp_id','error_id']) if right=='M5历史'
                 else group.xs(right,level='method'))
            t = pd.concat([a['T'],b['T']],axis=1,keys=['a','b']).dropna()
            theta = pd.concat([a.theta,b.theta],axis=1,keys=['a','b']).dropna()-THETA_TRUE
            acceptance = t.abs()<=CRITICAL
            diff = (acceptance.a.astype(float)-acceptance.b.astype(float))*100
            rmse_a,rmse_b = np.sqrt((theta**2).mean())
            rows.append(dict(dgp_id=dgp,error_id=error,error=ERROR_LABELS[error],left=left,right=right,
                test_pairs=len(t),point_pairs=len(theta),
                coverage_left=acceptance.a.mean(),coverage_right=acceptance.b.mean(),
                coverage_diff_pp=diff.mean(),coverage_diff_MCSE_pp=diff.std(ddof=1)/np.sqrt(len(t)) if len(t)>1 else np.nan,
                bias_left=theta.a.mean(),bias_right=theta.b.mean(),bias_diff=(theta.a-theta.b).mean(),
                rmse_left=rmse_a,rmse_right=rmse_b,rmse_diff=rmse_a-rmse_b,
                rmse_ratio=rmse_a/rmse_b if rmse_b>0 else np.nan))
    return pd.DataFrame(rows)


def history(packets):
    # 历史 M5 是额外配对基线，不增加本次拟合方法数。读前先核对数据、划分和可用的预测哈希。
    desktop = ROOT.parent
    source = desktop/'DML_5Methods_ScoreTest_Results'/'full'/'full_f7d73b2c1cee1e1a'/'evaluated_records.csv'
    ln = desktop/'Lognormal_Pilot_R200'/'.checkpoints'/'full'
    if not source.exists() or not ln.is_dir():
        print('历史 M5 文件缺失，跳过历史配对。',flush=True)
        return None
    historical = pd.read_csv(source,low_memory=False)
    historical = historical[(historical.method=='M5')&(historical.error_id==7)&
                            (historical.replication_id<200)].set_index(['dgp_id','replication_id'])
    result = []
    for packet in packets:
        d,e,r = packet['task']
        if e==7:
            old = historical.loc[(d,r)]
        else:
            old = json.loads((ln/f'd{d}_e{e}_r{r:03d}.json').read_text(encoding='utf-8'))['results']['M5']
        for key in ('data_hash','fold_hash','prediction_hash'):
            a,b = packet[key],old.get(key)
            if key=='prediction_hash' and (a is None or pd.isna(b)):
                continue
            if a!=b:
                raise ArithmeticError(f'M5 历史配对校验失败 {(d,e,r)} {key}')
        result.append(dict(dgp_id=d,error_id=e,replication_id=r,
                           theta=old['theta_hat'],T=old['T']))
    return pd.DataFrame(result).set_index(['dgp_id','error_id','replication_id'])


def export_plot(summary, path):
    configure_plot_style()
    fig, axes = plt.subplots(4,4,figsize=(17,11),layout='constrained')
    colors = ['#00728b','#287fa8','#183f78','#9ba4b1','#bbc3ce']
    panels = [('coverage','Coverage（%）',100,1),('bias','Bias',1,3),
              ('rmse','RMSE',1,3),('mean_T','Mean(T)',1,2)]
    for row,(dgp,error) in enumerate((d,e) for d in range(2) for e in ERROR_LABELS):
        subset = summary[(summary.dgp_id==dgp)&(summary.error_id==error)].set_index('method').loc[list(METHODS)]
        for column,(key,label,scale,digits) in enumerate(panels):
            ax=axes[row,column]
            bars=ax.bar(list(METHODS),subset[key]*scale,color=colors,width=.66,zorder=3)
            ax.bar_label(bars,fmt=f'%.{digits}f',padding=3,fontsize=8)
            if key=='coverage':
                ax.axhline(95,color='#667083',ls='--',lw=.9)
                ax.set_ylim(0,112)
            else:
                ax.axhline(0,color='#667083',lw=.7)
                ax.margins(y=.3)
            ax.grid(axis='y',alpha=.19)
            ax.set_title(f'DGP{dgp+1} · {ERROR_LABELS[error]} · {label}',fontsize=10)
    fig.suptitle('自适应全局位置与 Huber-DML · 五方法对照',fontsize=19)
    fig.legend([Patch(color=c) for c in colors],[METHOD_NAMES[m] for m in METHODS],
               loc='outside lower center',ncol=5,frameon=False)
    fig.savefig(path,dpi=165)
    plt.close(fig)


def conclusions(summary, comparisons, orth):
    text = []
    for (d,e),group in summary.groupby(['dgp_id','error_id']):
        center = group[group.method.isin(CENTERED)].set_index('method')
        ranks = center.coverage.sort_values(ascending=False)
        closest = center.mean_T.abs().idxmin()
        label = f'DGP{d+1} / {ERROR_LABELS[e]}'
        text.append(f'{label}：A/B/C 中 {ranks.index[0]} 的 Coverage 最高（{ranks.iloc[0]:.1%}），'
                    f'{closest} 的 Mean(T) 最接近 0（{center.loc[closest,"mean_T"]:+.3f}）。')
        subset = comparisons[(comparisons.dgp_id==d)&(comparisons.error_id==e)]
        for ref in ('M5历史','D','E'):
            lines = subset[(subset.left.isin(CENTERED))&(subset.right==ref)]
            if lines.empty:
                continue
            stats = '；'.join(f'{x.left} Coverage {x.coverage_diff_pp:+.1f} pp'
                f'（MCSE {x.coverage_diff_MCSE_pp:.1f} pp），Bias 差 {x.bias_diff:+.4f}，'
                f'RMSE 比值 {x.rmse_ratio:.2f}' for x in lines.itertuples())
            text.append(f'{label} 对 {ref}：{stats}。')
        values = orth[(orth.diagnostic=='gateaux')&(orth.dgp_id==d)&(orth.error_id==e)&
                      (orth.direction=='constant')&(orth.step==min(DERIVATIVE_STEPS))]
        derivative = '；'.join(f'{x.method} 总导数 {x.profile_numeric:+.5f}（理论 {x.profile_theory:+.5f}）'
                             for x in values.itertuples())
        delta = '；'.join(f'{m} {center.loc[m,"mean_T_estimated_minus_oracle"]:+.3f}' for m in CENTERED)
        text.append(f'{label}：{derivative}。配对的估计 nuisance − oracle nuisance 的 Mean(T) 差：{delta}。')
    return text


def export(packets,config,destination,smoke):
    frame,failures = flatten(packets,config,smoke)
    summary = describe(frame)
    baseline = None if smoke else history(packets)
    comparisons = paired(frame,baseline)
    ortho = oracle_population(packets,config)
    gateaux = pd.DataFrame(ortho)
    diagnostics = []
    for (d,e,m),group in frame.groupby(level=['dgp_id','error_id','method']):
        if m not in CENTERED:
            continue
        valid = group[['T','oracle_T','score_mean','oracle_score_mean','score_sd','oracle_score_sd']].dropna()
        diff = valid['T']-valid.oracle_T
        diagnostics.append(dict(diagnostic='oracle_vs_estimated',dgp_id=d,error_id=e,method=m,
            paired_n=len(valid),mean_T_estimated=valid['T'].mean(),mean_T_oracle=valid.oracle_T.mean(),
            mean_T_difference=diff.mean(),mean_T_difference_MCSE=diff.std(ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else np.nan,
            coverage_estimated=(valid['T'].abs()<=CRITICAL).mean(),
            coverage_oracle=(valid.oracle_T.abs()<=CRITICAL).mean(),
            mean_score_difference=(valid.score_mean-valid.oracle_score_mean).mean(),
            mean_score_SD_difference=(valid.score_sd-valid.oracle_score_sd).mean()))
    assert len(gateaux)==48 and len(diagnostics)==12
    assert len(summary)==20 and len(comparisons)==(36 if baseline is None else 48)
    assert np.isfinite(gateaux[['partial_numeric','profile_numeric','profile_theory']]).all().all()
    orth = pd.DataFrame(ortho+diagnostics)
    orth['error'] = orth.error_id.map(ERROR_LABELS)
    destination.mkdir(parents=True,exist_ok=True)
    for name,table in dict(summary=summary,paired=comparisons,orthogonality=orth,failures=failures).items():
        table.to_csv(destination/f'{PREFIX}{name}.csv',index=False,encoding='utf-8-sig')
    export_plot(summary,destination/f'{PREFIX}对照图.png')
    insights = conclusions(summary,comparisons,orth)
    intro = (
        'A/B/C 采用同一 OLS-MAD 阈值，双侧 1.96；主检验用 s−λq 的样本标准差。'
        'B 的 λ 使用稳健 Silverman 带宽和高斯核密度。D/E 沿用 corrected score 和无截距 LAD score，'
        'Naive 栏只适用于 A/B/C。D 的 Y nuisance MSE 以原假设下 transformed outcome 的总体条件均值为目标，'
        'A/B/C/E 则以 E[Y|X] 为目标。配对点估计和配对检验各用相应共同有效样本。'
        '历史 M5 比较使用历史同一 replication 的原始无位置 Huber 结果，核对数据、fold 和可用预测哈希。'
        ' Oracle 仅用于诊断。每场景将阈值固定在有效 τ 的中位数，用独立 50 万样本估计位置，另用 50 万误差/X1 '
        '配合 V 的对偶抽样估计导数。总体偏导为 −E[h]E[ψ] + θ₀E[Vh·I]；'
        'profile 总导数再减 a′E[V·I]。原 DGP 中 V 独立且均值为 0，后二项为 0，'
        '故二者都等于 −E[h]E[ψ]。理论列使用对应总体位置的精确分布积分。'
        'C 在此 oracle DGP 下使该导数为零；对于中心化 X1 方向三个方法理论导数都为零。'
        '对称 t 误差下三个总体位置都为 0。Outcome nuisance 的一阶项在这个 oracle DGP 下也因 V 独立且均值为 0 而消失。'
        '这些结论依赖 DGP；位置调整和样本 ψ 中心化不足以保证一般学习器极限下的正交性。'
        'Oracle 对照衡量替换预测后的统计量变化，不能单独分解有限样本偏差及方差校准。'
    )
    named = [('五方法完整汇总',summary),('共同有效样本上的配对比较',comparisons),
             ('固定位置偏导与 profile 总导数',gateaux),('同一数据和阈值下的 Oracle 对照',pd.DataFrame(diagnostics)),
             ('失败记录',failures)]
    sections = ''.join(f'<h2>{name}</h2><div class="scroll">{table.to_html(index=False,na_rep="—",float_format=lambda f:f"{f:.6g}")}</div>'
                       for name,table in named)
    items = ''.join(f'<li>{html.escape(line)}</li>' for line in insights)
    markup = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>自适应位置 DML Pilot</title>
<style>
body{{font:15px/1.65 "Microsoft YaHei",sans-serif;color:#15283e;margin:2.8rem;background:#f3f6fa}}
h1{{font-size:32px;color:#153f70}} h2{{margin-top:2rem}} p{{max-width:108ch}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;background:white;width:100%;font-size:13px;white-space:nowrap}}
th,td{{border:1px solid #d8e2ed;padding:6px 9px;text-align:right}} th{{background:#173f70;color:white}}
tbody tr:nth-child(even){{background:#edf4fa}} li{{margin:.45rem 0}}
</style><h1>自适应位置与 Huber-DML</h1>
<p>{config['mode']} · R={config['R']} / 场景 · {len(frame)} 条方法记录 · 点估计有效 {int(summary.point_n.sum())}
· 主检验有效 {int(summary.test_n.sum())}</p>
<p>{html.escape(intro)}</p><h2>配对结果与诊断</h2><ul>{items}</ul>{sections}</html>"""
    (destination/f'{PREFIX}结果表.html').write_text(markup,encoding='utf-8')
    for name,table in named:
        print(f'\n{name}\n{table.to_string(index=False,float_format=lambda f:f"{f:.5g}")}',flush=True)
    if not smoke:
        print('\n'+'\n'.join(insights),flush=True)
    return dict(point_n=int(summary.point_n.sum()), test_n=int(summary.test_n.sum()),
                failure_stages=len(failures))


def main():
    import argparse
    import msvcrt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='只执行八进程自检')
    args = parser.parse_args()
    for name,version in REQUIRED_VERSIONS.items():
        if metadata.version(name)!=version:
            raise RuntimeError(f'{name} 必须是原实验版本 {version}')
    CHECKPOINTS.mkdir(exist_ok=True)
    with (CHECKPOINTS/'run.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        smoke = config_for('smoke')
        marker = CHECKPOINTS/f'smoke_{smoke["config_hash"][:16]}.json'
        if not marker.exists():
            smoke_packets = collect(smoke,True)
            pids = sorted({packet['worker_pid'] for packet in smoke_packets})
            assert len(pids)==WORKERS, f'Smoke 实际进程数 {len(pids)}'
            result = export(smoke_packets,smoke,CHECKPOINTS/'smoke_output',True)
            atomic_json(marker,dict(config=smoke,worker_pids=pids,**result))
        print('Smoke 通过，八个独立进程，40 条方法记录。',flush=True)
        if args.smoke:
            return
        config = config_for('full')
        start = time.perf_counter()
        atomic_json(CHECKPOINTS/'full_config.json',dict(config=config,sources=SOURCE_HASHES))
        packets = collect(config,False)
        result = export(packets,config,ROOT,False)
        atomic_json(CHECKPOINTS/'completed.json',dict(**result,seconds=time.perf_counter()-start,
                    config_hash=config['config_hash']))
        print(f'正式实验完成：{len(packets)} 组，结果位于 {ROOT}',flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
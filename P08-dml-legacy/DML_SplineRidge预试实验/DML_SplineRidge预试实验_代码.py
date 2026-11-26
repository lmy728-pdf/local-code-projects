# -*- coding: utf-8 -*-
"""样条 Ridge 第一阶段与历史自适应学习器的配对实验。默认自检后运行 R=200。"""
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):
        stream.reconfigure(encoding='utf-8',errors='replace')

import hashlib
import html
import importlib.metadata as metadata
import json
import math
import multiprocessing
import platform
import time
import traceback
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar, minimize
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, RidgeCV, Lasso, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

# 以下统计定义逐一复用上一轮实验，保留原始 A、E、D 的求解和检验。
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

ERROR_LABELS = {7:'Student-t(2.1)', 14:'Lognormal(2)'}

LOGNORMAL_SIGMAS = {14:2.0}

WORKERS = 8

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

PRIOR_SOURCE_HASH = 'b55c218eb46e8b895f4a020031eb27561f56c95606bf96848be5425977eb4157'


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / '.checkpoints'
PRIOR = ROOT.parent / 'Adaptive_Centering_DML_Pilot'
PREFIX = 'Spline_Ridge_'
METHODS = ('S','A','E','D')
METHOD_NAMES = dict(S='S 样条 Ridge', A='A 均值 Huber', E='E LAD-DML', D='D Dukes')
ALPHAS = np.logspace(-4,4,17)
KNOTS = 5


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


def infer_baselines(rows,observed,folds,config,task):
    X,D,Y = observed
    features = feature_dictionary(X)
    shared = dict(failures=[])
    record_stage(shared,'nuisance_D',lambda:dict(m=adaptive_side(
        features,D,folds,config,*task,False,'D')))
    m = shared.get('m')
    if m is not None:
        record_stage(shared,'nuisance_Y',lambda:dict(ell=adaptive_side(
            features,Y,folds,config,*task,True,'Y')))
    ell = shared.get('ell')
    for method in ('A','E'):
        rows[method]['failures'].extend(shared['failures'])
        rows[method].update(prediction_errors(observed,task[0],m,ell))
    if m is not None and ell is not None:
        w,u = D-m,Y-ell
        mean_huber(rows['A'],observed,task,m,ell)
        row = rows['E']
        record_stage(row,'point',lambda:dict(theta=lad_point(w,u)[0]))
        record_stage(row,'test',lambda:{k:v for k,v in
            standardize_score(residual_scores('M2',w,u,THETA_NULL),CRITICAL).items()
            if k in ('T','score_mean','score_sd')})
    row = rows['D']
    row['failures'].extend(shared['failures'])
    if m is not None:
        record_stage(row,'test',lambda:dukes_null(observed,folds,config,task,m))
    transformed = row.pop('transformed_prediction',None)
    row.update(prediction_errors(observed,task[0],m,transformed,transformed=True))
    return m,ell


def replay_dukes_point(row,observed,folds,config,task,m,ell):
    if m is not None and ell is not None:
        _,D,Y = observed
        record_stage(row,'point',lambda:dukes_estimate(observed,folds,config,task,m,D-m,Y-ell))


def same_scalar(expected,actual):
    if expected is None or actual is None:
        return expected is None and actual is None
    return bool(np.isclose(expected,actual,rtol=1e-11,atol=1e-13))


def verify_archived(packet,archive_hash,rows,audit,task,expected_prediction_hash,config):
    assert packet['task']==list(task) and packet['config_hash']==archive_hash, '旧断点标识'
    for key in ('data_hash','fold_hash'):
        assert packet[key]==audit[key], key
    assert packet['prediction_hash']==expected_prediction_hash, 'A/E 共享 OOF 预测哈希'
    old = {row['method']:row for row in packet['rows']}
    assert len(old)==5 and set(old)==set('ABCDE'), '旧方法记录不完整'
    for method in ('A','E','D'):
        archived = old[method]
        current = rows[method]
        assert [archived[k] for k in ('dgp_id','error_id','replication_id')]==list(task)
        keys = ['T','score_mean','score_sd']
        if method=='A':
            keys.extend(('theta','naive_T','naive_score_mean','naive_score_sd'))
            assert same_scalar(packet.get('tau'),current.get('tau')), 'A 阈值'
            assert same_scalar(archived.get('oracle_T'),current.get('oracle_fixed_tau_T')), 'A 原阈值 Oracle'
        if method=='E' or config['mode']=='smoke':
            keys.append('theta')
        # 旧实验在共享 Y 拟合失败时未单独汇总可用的 D 侧预测。
        keys.extend(k for k in ('y_nuisance_mse','d_nuisance_mse') if archived.get(k) is not None)
        for key in keys:
            assert same_scalar(archived.get(key),current.get(key)), f'{method} {key}'
        if archived.get('T') is not None:
            assert same_scalar(archived['T'],np.sqrt(config['n'])*archived['score_mean']/archived['score_sd'])
        assert archived.get('theta') is not None and archived.get('T') is not None or archived['failures']
        stages = {'test','nuisance_D','nuisance_Y'}
        if method!='D' or config['mode']=='smoke':
            stages |= {'point','threshold','naive'}
        def signature(row):
            return sorted((f['stage'],f['type'],str(f['fold']),f['message'])
                          for f in row['failures'] if f['stage'] in stages)
        assert signature(archived)==signature(current), f'{method} 失败状态'
    dukes = old['D']
    if dukes.get('theta') is not None:
        assert expected_prediction_hash is not None and np.isfinite(dukes['theta'])
        assert abs(dukes['point_equation_T'])<=config['n']**(-M3_SOLVER['moment_T_power'])
        assert 1<=dukes['point_evaluations']<=M3_SOLVER['max_evaluations']
    else:
        assert any(f['stage'] in ('point','nuisance_D','nuisance_Y') for f in dukes['failures'])
    return old


def worker_packet(config,task):
    started = time.perf_counter()
    generator = generate_observed if task[1]==7 else generate_pilot_observed
    observed,folds,_,audit = generator(config,*task)
    validate_folds(folds,config['n'])
    rows = {m:dict(method=m,dgp_id=task[0],error_id=task[1],replication_id=task[2],failures=[])
            for m in METHODS}
    spline = rows['S']
    def fit_spline():
        prediction,alpha = spline_oof(observed,folds,config,task)
        return dict(prediction=prediction,alpha_d=[a[0] for a in alpha],alpha_y=[a[1] for a in alpha])
    record_stage(spline,'nuisance',fit_spline)
    prediction = spline.pop('prediction',None)
    if prediction is not None:
        spline.update(prediction_errors(observed,task[0],*prediction))
        mean_huber(spline,observed,task,*prediction)
    m,ell = infer_baselines(rows,observed,folds,config,task)
    archived = None
    reason = config['archive_reason']
    if config['mode']=='smoke':
        replay_dukes_point(rows['D'],observed,folds,config,task,m,ell)
    if config['archive_folder']:
        try:
            path = Path(config['archive_folder'])/packet_name(task)
            packet = json.loads(path.read_text(encoding='utf-8'))
            prediction_hash = sha(np.array([m,ell]).tobytes()) if ell is not None else None
            archived = verify_archived(packet,config['archive_hash'],rows,audit,task,prediction_hash,config)
        except (AssertionError,ValueError,TypeError,KeyError,OSError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
    if archived is None:
        if config['mode']!='smoke':
            replay_dukes_point(rows['D'],observed,folds,config,task,m,ell)
    else:
        # 保留上一轮实际估计、检验与失败信息。新增诊断沿用已逐组核对的预测。
        for method in ('A','E','D'):
            old = archived[method]
            row = rows[method]
            for key,value in old.items():
                if key=='failures' or key=='T' or key=='theta' or key.startswith(('point_','score_','naive_')):
                    row[key]=value
    oracle = dict(failures=[])
    def common_oracle():
        X,D,Y = observed
        true_m,true_g = nuisance_functions(X,DGP_NAMES[task[0]])
        true_ell = THETA_TRUE*true_m+true_g
        _,tau,_ = huber_threshold(D-true_m,Y-true_ell)
        return dict(oracle_common_tau=tau,oracle_common_T=oracle_replicate('A',observed,task[0],tau)['oracle_T'])
    record_stage(oracle,'oracle_common',common_oracle)
    for method in ('S','A'):
        rows[method]['failures'].extend(oracle['failures'])
        rows[method].update({k:v for k,v in oracle.items() if k!='failures'})
    return dict(task=list(task),config_hash=config['config_hash'],rows=list(rows.values()),
                data_hash=audit['data_hash'],fold_hash=audit['fold_hash'],worker_pid=os.getpid(),
                archived=archived is not None,archive_reason=None if archived is not None else reason,
                seconds=time.perf_counter()-started)


def packet_name(task):
    d,e,r = task
    return f'd{d}_e{e}_r{r:03d}.json'


def expected_tasks(repetitions):
    return [(d,e,r) for r in range(repetitions) for d in range(2) for e in ERROR_LABELS]


def design(mode):
    return dict(mode=mode,n=1000,R=2 if mode=='smoke' else 200,p=P,K=K,base_seed=BASE_SEED,
                spec=dict(kernel=DEFAULT_KERNEL_CONFIG,m3_solver=M3_SOLVER),
                dependencies={name:metadata.version(name) for name in REQUIRED_VERSIONS},
                python=platform.python_version())


def archive_for(settings):
    try:
        source = PRIOR/'Adaptive_Centering_DML_Pilot.py'
        assert sha(source.read_bytes())==PRIOR_SOURCE_HASH, '历史方法代码哈希不匹配'
        checks = PRIOR/'.checkpoints'
        if settings['mode']=='full':
            prior = json.loads((checks/'full_config.json').read_text(encoding='utf-8'))['config']
            evidence = json.loads((checks/'completed.json').read_text(encoding='utf-8'))
        else:
            candidates = [json.loads(f.read_text(encoding='utf-8')) for f in checks.glob('smoke_*.json')]
            evidence, = [x for x in candidates if x['config']['code_hash']==PRIOR_SOURCE_HASH]
            prior = evidence['config']
        assert prior['code_hash']==PRIOR_SOURCE_HASH
        assert all(prior[k]==v for k,v in settings.items()), '种子、DGP 规格或数值环境不同'
        digest = sha(json.dumps({k:v for k,v in prior.items() if k!='config_hash'},sort_keys=True).encode())
        assert digest==prior['config_hash']==evidence.get('config_hash',prior['config_hash'])
        folder = checks/settings['mode']/digest[:16]
        tasks = expected_tasks(settings['R'])
        assert {p.name for p in folder.glob('*.json')}=={packet_name(t) for t in tasks}, '旧断点数或 ID 不符'
        summary_file = (PRIOR if settings['mode']=='full' else checks/'smoke_output')/'Adaptive_Centering_summary.csv'
        summary = pd.read_csv(summary_file).set_index(['dgp_id','error_id','method'])
        archives = [json.loads((folder/packet_name(t)).read_text(encoding='utf-8')) for t in tasks]
        assert all(p['task']==list(t) and p['config_hash']==digest for p,t in zip(archives,tasks))
        all_records = [r for p in archives for r in p['rows']]
        assert len(all_records)==len(tasks)*5
        assert sum(r.get('T') is not None for r in all_records)==evidence['test_n']
        assert sum(r.get('theta') is not None for r in all_records)==evidence['point_n']
        for (d,e,m),group in pd.DataFrame(all_records).groupby(['dgp_id','error_id','method']):
            if m not in ('A','E','D'):
                continue
            old = summary.loc[d,e,m]
            assert len(group)==settings['R'] and group.replication_id.is_unique
            T,theta = group['T'].dropna(),group.theta.dropna()
            assert len(T)==old.test_n and len(theta)==old.point_n
            assert same_scalar(float((T.abs()<=CRITICAL).mean()),old.coverage)
            assert same_scalar(float((theta-THETA_TRUE).mean()),old.bias)
            assert same_scalar(float(np.sqrt(((theta-THETA_TRUE)**2).mean())),old.rmse)
        return dict(archive_folder=str(folder),archive_hash=digest,archive_reason=None)
    except (AssertionError,ValueError,KeyError,TypeError,OSError) as exc:
        return dict(archive_folder=None,archive_hash=None,archive_reason=f'{type(exc).__name__}: {exc}')


def config_for(mode):
    settings = design(mode)
    settings.update(archive_for(settings))
    settings['code_hash'] = sha(Path(__file__).read_bytes())
    settings['config_hash'] = sha(json.dumps(settings,sort_keys=True).encode())
    return settings


def collect(config):
    smoke = config['mode']=='smoke'
    tasks = expected_tasks(config['R'])
    folder = CHECKPOINTS/config['mode']/config['config_hash'][:16]
    records,pending = {},[]
    for task in tasks:
        path = folder/packet_name(task)
        if path.exists():
            packet = json.loads(path.read_text(encoding='utf-8'))
            assert packet['config_hash']==config['config_hash'] and packet['task']==list(task)
            records[task] = packet
        else:
            pending.append(task)
    if pending:
        ctx = multiprocessing.get_context('spawn')
        barrier = ctx.Barrier(WORKERS,timeout=180) if smoke and len(pending)==WORKERS else None
        with ProcessPoolExecutor(max_workers=WORKERS,mp_context=ctx,
                initializer=init_worker,initargs=(barrier,)) as pool:
            futures = {pool.submit(worker_packet,config,t):t for t in pending}
            with tqdm(total=len(tasks),initial=len(records),desc='Smoke' if smoke else '正式 R200',
                      unit='组',mininterval=2) as bar:
                for future in as_completed(futures):
                    task = futures[future]
                    packet = future.result()
                    atomic_json(folder/packet_name(task),packet)
                    records[task] = packet
                    bar.update()
    return [records[t] for t in tasks]


def flatten(packets,config):
    rows,failures = [],[]
    for p in packets:
        assert p['config_hash']==config['config_hash']
        assert tuple(row['method'] for row in p['rows'])==METHODS
        for row in p['rows']:
            key = {k:row[k] for k in ('dgp_id','error_id','replication_id','method')}
            assert list(key.values())[:3]==p['task']
            assert (row.get('theta') is not None and row.get('T') is not None) or row['failures']
            assert row.get('T') is None or same_scalar(row['T'],np.sqrt(config['n'])*row['score_mean']/row['score_sd'])
            rows.append({**{k:v for k,v in row.items() if k!='failures'},
                         'inference_source':'archived_verified' if row['method']!='S' and p['archived'] else 'fitted',
                         'archive_verification_note':p['archive_reason'] if row['method']!='S' else None})
            failures.extend(dict(key,**failure) for failure in row['failures'])
    frame = pd.DataFrame(rows).set_index(['dgp_id','error_id','replication_id','method']).sort_index()
    assert len(frame)==4*config['R']*len(METHODS) and frame.index.is_unique
    assert frame.xs('S',level='method').oracle_common_T.equals(frame.xs('A',level='method').oracle_common_T)
    return frame,pd.DataFrame(failures,columns=['dgp_id','error_id','replication_id','method','stage','type','fold','message'])


DIAGNOSTIC_METRICS = ('y_nuisance_mse','d_nuisance_mse','y_prediction_bias','d_prediction_bias',
                      'y_observed_mse','d_observed_mse','delta_m_delta_ell','delta_m2','delta_product_net')
SUMMARY = ('coverage','bias','rmse','mean_T','sd_T','left_reject','right_reject')


def summarize(frame):
    results,diag = [],[]
    for (d,e,m),g in frame.groupby(level=['dgp_id','error_id','method']):
        T,theta,naive = g['T'].dropna(),g.theta.dropna(),g.naive_T.dropna()
        errors = theta-THETA_TRUE
        common = dict(dgp_id=d,error_id=e,error=ERROR_LABELS[e],method=m)
        stats = {k:g[k].mean() for k in DIAGNOSTIC_METRICS}
        tests = dict(coverage=(T.abs()<=CRITICAL).mean(),bias=errors.mean(),
                    rmse=np.sqrt((errors**2).mean()),mean_T=T.mean(),sd_T=T.std(ddof=1),
                    left_reject=(T< -CRITICAL).mean(),right_reject=(T>CRITICAL).mean())
        naive_matched = g[['T','naive_T']].dropna()
        results.append(dict(common,planned=len(g),point_n=len(theta),test_n=len(T),
            point_fail=len(g)-len(theta),test_fail=len(g)-len(T),
            either_fail=int((g.theta.isna()|g['T'].isna()).sum()),
            **tests,naive_n=len(naive),naive_coverage=(naive.abs()<=CRITICAL).mean(),
            naive_matched_n=len(naive_matched),
            adjusted_minus_naive_coverage_pp=100*((naive_matched['T'].abs()<=CRITICAL).mean()
                                                -(naive_matched.naive_T.abs()<=CRITICAL).mean()),
            archived_n=int(g.inference_source.eq('archived_verified').sum()),
            y_oof_n=int(g.y_nuisance_mse.notna().sum()),d_oof_n=int(g.d_nuisance_mse.notna().sum()),
            y_target=g.y_target.dropna().iloc[0] if g.y_target.notna().any() else '',
            **stats))
        oracle = {}
        for kind in ('common','fixed_tau'):
            both = g[['T',f'oracle_{kind}_T']].dropna()
            diff = both['T']-both[f'oracle_{kind}_T']
            oracle.update({f'{kind}_n':len(both),f'{kind}_estimated_mean_T':both['T'].mean(),
                f'{kind}_oracle_mean_T':both[f'oracle_{kind}_T'].mean(),
                f'{kind}_estimated_minus_oracle_mean_T':diff.mean()})
        diag.append(dict(common,aggregation='method',**stats,**oracle))
    for (d,e),g in frame.groupby(level=['dgp_id','error_id']):
        g = g.droplevel(['dgp_id','error_id'])
        a,s = g.xs('A',level='method'),g.xs('S',level='method')
        metrics = ['T','oracle_common_T','oracle_fixed_tau_T',*DIAGNOSTIC_METRICS]
        sa = pd.concat([s[metrics],a[metrics]],axis=1,keys=['S','A'])
        matched = sa.dropna(subset=[(m,k) for m in ('S','A') for k in ('T','oracle_common_T')])
        for method in ('S','A'):
            sub = matched[method]
            diag.append(dict(dgp_id=d,error_id=e,error=ERROR_LABELS[e],method=method,
                aggregation='S_A_common_replications',common_n=len(matched),
                **{k:sub[k].mean() for k in DIAGNOSTIC_METRICS},
                common_estimated_mean_T=sub['T'].mean(),common_oracle_mean_T=sub.oracle_common_T.mean(),
                common_estimated_minus_oracle_mean_T=(sub['T']-sub.oracle_common_T).mean(),
                fixed_tau_oracle_mean_T=sub.oracle_fixed_tau_T.mean(),
                fixed_tau_estimated_minus_oracle_mean_T=(sub['T']-sub.oracle_fixed_tau_T).mean()))
    return pd.DataFrame(results),pd.DataFrame(diag)


def paired_results(frame):
    rows = []
    for (d,e),group in frame.groupby(level=['dgp_id','error_id']):
        group = group.droplevel(['dgp_id','error_id'])
        s = group.xs('S',level='method')
        for ref in METHODS[1:]:
            other = group.xs(ref,level='method')
            T = pd.concat([s['T'],other['T']],axis=1,keys=['s','ref']).dropna()
            theta = pd.concat([s.theta,other.theta],axis=1,keys=['s','ref']).dropna()-THETA_TRUE
            yes = T.abs()<=CRITICAL
            coverage_difference = 100*(yes.s.astype(float)-yes.ref.astype(float))
            rmse_s,rmse_ref = np.sqrt((theta**2).mean())
            rows.append(dict(dgp_id=d,error_id=e,error=ERROR_LABELS[e],method='S',reference=ref,
                test_pairs=len(T),point_pairs=len(theta),coverage_S=yes.s.mean(),coverage_ref=yes.ref.mean(),
                coverage_diff_pp=coverage_difference.mean(),
                coverage_diff_MCSE_pp=coverage_difference.std(ddof=1)/np.sqrt(len(T)) if len(T)>1 else np.nan,
                bias_S=theta.s.mean(),bias_ref=theta.ref.mean(),bias_diff=(theta.s-theta.ref).mean(),
                rmse_S=rmse_s,rmse_ref=rmse_ref,rmse_diff=rmse_s-rmse_ref,
                rmse_ratio=rmse_s/rmse_ref if rmse_ref>0 else np.nan))
    return pd.DataFrame(rows)


def export_plot(summary,destination):
    configure_plot_style()
    fig,axes = plt.subplots(4,6,figsize=(21,10.7),layout='constrained')
    colors = ['#087c79','#27558a','#d5914e','#8492a6']
    panels = [('coverage','覆盖率 (%)',100,1),('bias','Bias',1,4),('rmse','RMSE',1,4),
              ('mean_T','Mean(T)',1,2),('y_nuisance_mse','Y OOF MSE',1,3),
              ('d_nuisance_mse','D OOF MSE',1,3)]
    for i,(d,e) in enumerate((d,e) for d in range(2) for e in ERROR_LABELS):
        values = summary[(summary.dgp_id==d)&(summary.error_id==e)].set_index('method').loc[list(METHODS)]
        for j,(metric,title,scale,decimals) in enumerate(panels):
            ax = axes[i,j]
            labels = ['S','A','E','D*'] if metric=='y_nuisance_mse' else list(METHODS)
            bars = ax.bar(labels,values[metric]*scale,color=colors,width=.67,zorder=3)
            ax.bar_label(bars,fmt=f'%.{decimals}f',fontsize=8,padding=3)
            ax.set_title(f'DGP{d+1} · {ERROR_LABELS[e]}\n{title}',fontsize=10)
            ax.grid(axis='y',alpha=.22)
            if metric=='coverage':
                ax.set_ylim(0,113)
                ax.axhline(95,ls='--',lw=.8,color='#7a8795')
            else:
                ax.margins(y=.25)
                ax.axhline(0,lw=.65,color='#7a8795')
    fig.suptitle('Spline-Ridge 第一阶段的 DML 推断对照',fontsize=19)
    fig.legend([Patch(color=c) for c in colors],[METHOD_NAMES[m] for m in METHODS],
               loc='outside lower center',ncol=4,frameon=False,
               title='* Dukes 的 Y MSE 以变换结果的条件均值为目标')
    fig.savefig(destination,dpi=165)
    plt.close(fig)


def takeaways(paired,diag):
    text = []
    for (d,e),comparisons in paired.groupby(['dgp_id','error_id']):
        label = f'DGP{d+1} · {ERROR_LABELS[e]}'
        ds = diag[(diag.dgp_id==d)&(diag.error_id==e)&(diag.aggregation=='S_A_common_replications')].set_index('method')
        s,a = ds.loc['S'],ds.loc['A']
        text.append(f'{label}，S 与 A 在共同有效样本上的 Y MSE 为 {s.y_nuisance_mse:.5f} 与 '
                    f'{a.y_nuisance_mse:.5f}，D MSE 为 {s.d_nuisance_mse:.5f} 与 {a.d_nuisance_mse:.5f}。'
                    f'净预测误差交叉项为 {s.delta_product_net:+.5f} 与 {a.delta_product_net:+.5f}。'
                    f'Mean(T) 为 {s.common_estimated_mean_T:+.3f} 与 {a.common_estimated_mean_T:+.3f}，'
                    f'共同 Oracle 为 {s.common_oracle_mean_T:+.3f}。')
        for row in comparisons.itertuples():
            text.append(f'{label}，S 对 {row.reference} 的覆盖差为 {row.coverage_diff_pp:+.1f} 个百分点'
                        f'（MCSE {row.coverage_diff_MCSE_pp:.1f}，配对数 {row.test_pairs}）。'
                        f'Bias 差为 {row.bias_diff:+.5f}，RMSE 比值为 {row.rmse_ratio:.3f}。')
    return text


def export(packets,config,destination):
    frame,failures = flatten(packets,config)
    summary,diag = summarize(frame)
    paired = paired_results(frame)
    assert len(summary)==16 and len(paired)==12 and len(diag)==24
    destination.mkdir(parents=True,exist_ok=True)
    raw_columns = [*DIAGNOSTIC_METRICS,'y_target','T','naive_T','oracle_common_T','oracle_common_tau',
                   'oracle_fixed_tau_T','tau','inference_source','archive_verification_note']
    raw = frame[raw_columns].reset_index()
    raw['aggregation']='replication'
    raw['error']=raw.error_id.map(ERROR_LABELS)
    raw = pd.concat([diag,raw],ignore_index=True)
    outputs = dict(summary=summary,paired=paired,nuisance_diagnostics=raw,failures=failures)
    for key,table in outputs.items():
        table.to_csv(destination/f'{PREFIX}{key}.csv',index=False,encoding='utf-8-sig')
    export_plot(summary,destination/f'{PREFIX}对照图.png')
    archived = sum(p['archived'] for p in packets)
    notes = takeaways(paired,diag)
    sections = [('四方法完整汇总',summary),('S 与 A、E、D 的逐次配对',paired),
                ('第一阶段预测误差与 Oracle 对照',diag),('估计、检验与诊断失败记录',failures)]
    display = ''.join(f'<h2>{title}</h2><div class="scroll">{table.to_html(index=False,na_rep="—",float_format=lambda v:f"{v:.6g}")}</div>'
                      for title,table in sections)
    intro = (
        f'{config["mode"]} · 每场景 {config["R"]} 组 · {len(frame)} 条方法记录 · 历史验证并复用 {archived}/{len(packets)} 组。'
        '逐组核对数据、折划分、预测哈希、检验统计量与失败状态。'
        'A/E 估计及 D 检验可直接重算核对，自检亦重算 D 点估计。'
        '需要恢复的历史预测仅供诊断和复核。'
        '样条使用三次 B-spline、5 个分位数结点、常数外推、原线性项和两两交互项。'
        '三折验证中逐折拟合结点及标准化，RidgeCV 依据平方损失选择正则参数，再在完整外层训练折重拟合。'
        'S 与 A 使用相同的均值位置 Huber 方程、OLS-MAD 阈值规则、位置调整后的 Score-test 及双侧 1.96 临界值。'
        '预测偏差为预测减真值的均值。D 的 Y 指标以变换结果条件均值为目标，Y 与 D 的交叉项只用于原 Y nuisance。'
        '共同 Oracle 的阈值由 Oracle 残差按同一规则估计，并且另报固定各方法实际阈值的 Oracle 结果。'
        '净交叉项对应未截断残差乘积中的一项，在这里用于诊断。'
        'Huber 截断、均值位置和标准化也会影响 T，因此该项无法单独解释覆盖率。'
        '配对检验与点估计各用双方共同有效的重复。'
    )
    items = ''.join(f'<li>{html.escape(x)}</li>' for x in notes)
    markup = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Spline-Ridge DML Pilot</title>
<style>
body{{font:15px/1.65 "Microsoft YaHei",sans-serif;color:#152a3c;background:#f4f7fa;margin:2.7rem}}
h1{{font-size:32px;color:#18546a}} h2{{margin-top:2rem}} p{{max-width:105ch}}
.scroll{{overflow:auto}} table{{border-collapse:collapse;width:100%;white-space:nowrap;background:white;font-size:13px}}
th,td{{border:1px solid #d7e3ea;padding:6px 8px;text-align:right}} th{{background:#1a5065;color:white}}
tbody tr:nth-child(even){{background:#edf5f8}} li{{margin:.4rem 0}}
</style><h1>Spline-Ridge 与自适应第一阶段</h1><p>{html.escape(intro)}</p><h2>主要配对结果</h2><ul>{items}</ul>{display}</html>"""
    (destination/f'{PREFIX}结果表.html').write_text(markup,encoding='utf-8')
    for title,table in sections:
        print(f'\n{title}\n{table.to_string(index=False,float_format=lambda v:f"{v:.5g}")}',flush=True)
    if config['mode']=='full':
        print('\n'+'\n'.join(notes),flush=True)
    return dict(point_n=int(summary.point_n.sum()),test_n=int(summary.test_n.sum()),
                failure_stages=len(failures),archived_packets=archived,
                data_packets=len(packets),worker_pids=sorted({p['worker_pid'] for p in packets}))


def main():
    import argparse
    import msvcrt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='仅做极小规模自检')
    args = parser.parse_args()
    for package,version in REQUIRED_VERSIONS.items():
        if metadata.version(package)!=version:
            raise RuntimeError(f'{package} 需与原实验的 {version} 一致')
    CHECKPOINTS.mkdir(exist_ok=True)
    with (CHECKPOINTS/'run.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        start = time.perf_counter()
        smoke = config_for('smoke')
        smoke_marker = CHECKPOINTS/f'smoke_{smoke["config_hash"][:16]}.json'
        if not smoke_marker.exists():
            packets = collect(smoke)
            assert len({p['worker_pid'] for p in packets})==WORKERS, '未使用八个自检进程'
            assert all(not row['failures'] for p in packets for row in p['rows']), '自检存在数值失败'
            result = export(packets,smoke,CHECKPOINTS/'smoke_output')
            atomic_json(smoke_marker,dict(config=smoke,**result))
        else:
            result = json.loads(smoke_marker.read_text(encoding='utf-8'))
        print(f'Smoke 通过。8 个 Worker，32 条有效方法记录，历史配对核验 {result["archived_packets"]}/8 组。',flush=True)
        if args.smoke:
            return
        config = config_for('full')
        if config['archive_folder'] is None:
            print(f'历史结果无法验证，按原代码重跑 A/E/D。{config["archive_reason"]}',flush=True)
        else:
            print('历史清单、代码及汇总核对通过，开始逐组核对和拟合。',flush=True)
        atomic_json(CHECKPOINTS/'full_config.json',config)
        packets = collect(config)
        result = export(packets,config,ROOT)
        atomic_json(CHECKPOINTS/'completed.json',dict(**result,seconds=time.perf_counter()-start,
                    config_hash=config['config_hash']))
        print(f'完成 {len(packets)} 组，结果位于 {ROOT}',flush=True)


if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
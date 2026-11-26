# -*- coding: utf-8 -*-
r"""Adaptive Lasso / Robust Adaptive Lasso nuisance + original Huber-DML.

运行：D:\Anoconda\envs\environment2\python.exe -B Adaptive_Lasso_DML.py
先 R=3 检查，再 R=200；全部设置预先固定，不根据 coverage 调参。
不修改原 Notebook，不写 CSV/JSON/日志/checkpoint，逐次记录只保留内存。
磁盘输出仅为桌面 Adaptive_Lasso_对照表.png 与 Adaptive_Lasso_对照图.png。

正则化方法是本实验明确实现的适配，不声称已证明其 DML 推断或 Oracle 性质。
M4 在原始 5 维 X 上使用原 HGBR；M6 均使用同一完整二次字典。
因此相对 M4 的差异包含学习器与字典的共同影响，不能只归因于 L1 惩罚。

数值实现参考：
https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Lasso.html
https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.HuberRegressor.html
https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import sys
sys.dont_write_bytecode=True
for _stream in (sys.stdout,sys.stderr):
    if hasattr(_stream,'reconfigure'):
        _stream.reconfigure(encoding='utf-8',errors='replace')
import math
import time
import warnings
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import brentq, minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from joblib import Parallel,delayed,parallel_config
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

N=1000
P=5
K=5
R=200
SMOKE_R=3
BASE_SEED=20261103
N_JOBS=4
THETA_TRUE=1.0  # 只进入独立评价步骤；原生成器 Y=D+g+epsilon。
CRITICAL=1.96
ERROR_IDS=(0,7)
ERROR_LABELS={0:'Gaussian',7:'t(2.1)'}
METHODS=('M4','M6-A','M6-B')
DGP_LABELS=('DGP1 线性','DGP2 非线性')
INNER_K=3
# 选择训练数据决定的 lambda_max 的比例。相同预定网格用于全部场景。
LAMBDA_RATIOS=np.logspace(0,-3,13)
ADAPTIVE_GAMMA=1.0
WEIGHT_EPS=1e-3  # 标准化系数尺度下的固定稳定项，不使用真实非零系数。
RIDGE_ALPHA=1.0
PILOT_HUBER_EPS=1.35
NUISANCE_HUBER_KAPPA=1.345
KKT_TOL=2e-6

SOURCE_NOTEBOOK_SHA256='41ffadee98ab10aeb76b99c68f1cd308129a09c04dde29a80305ee4f0d6f4f55'
SOLVER={'min_denominator': 1e-10, 'm3_condition_limit': 100000000.0, 'score_tol': 1e-08, 'kappa': 1.345, 'm3_steps': 1, 'm3_path': 'm(theta)=m(anchor)-(theta-anchor)*g; G,M,alpha fixed; beta reprojected analytically', 'm3_se': 'frozen_path_centered_sandwich_ddof0'}
HGBR_PARAMS={'categorical_features': 'from_dtype', 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 1, 'learning_rate': 0.05, 'loss': 'squared_error', 'max_bins': 255, 'max_depth': None, 'max_features': 1.0, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'quantile': None, 'random_state': 0, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
DGP_NAMES=('DGP1_linear', 'DGP2_nonlinear')
T_GRID=(20.0, 10.0, 5.0, 3.0, 2.6, 2.3, 2.1)

# 下列 12 个函数/类直接复制原 Notebook；保持源码与统计公式。
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


CONFIG=dict(n=N,p=P,K=K,R=R,mode='full',base_seed=BASE_SEED,
            spec=dict(learner=HGBR_PARAMS.copy()))


def desktop_path():
    import winreg
    registry_path=r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,registry_path) as key:
        value,_=winreg.QueryValueEx(key,'Desktop')
    return Path(os.path.expandvars(value)).resolve()


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


def regularized_crossfit(observed,folds,config,dgp,error,rep,robust_y=False,d_predictions=None):
    X,D,Y=observed
    F=feature_dictionary(X)
    predictions=np.empty((2,len(D)))
    if d_predictions is not None:
        predictions[0]=d_predictions
    seen=np.zeros(len(D),dtype=int)
    diagnostics=[]
    for k,(train,test) in enumerate(folds):
        if np.intersect1d(train,test).size:
            raise ArithmeticError('outer_fold_leakage')
        inner_seed=seed(config,dgp,error,rep,50,k,0)
        splits=list(KFold(INNER_K,shuffle=True,random_state=inner_seed).split(train))
        targets=((1,Y,robust_y),) if d_predictions is not None else ((0,D,False),(1,Y,robust_y))
        for j,target,robust in targets:
            try:
                predictions[j,test],diag=adaptive_prediction(F[train],target[train],F[test],splits,robust)
            except Exception as exc:
                raise RuntimeError(f'outer_fold={k}; target={"D" if j==0 else "Y"}; {type(exc).__name__}: {exc}') from exc
            diagnostics.append(dict(target=j,fold=k,**diag))
        seen[test]+=1
    if not np.all(seen==1) or not np.isfinite(predictions).all():
        raise ArithmeticError('OOF_predictions_incomplete')
    return predictions,diagnostics


def estimate_from_predictions(observed,predictions):
    """三方法唯一共用的二阶段入口：原版 m4，未传入真值。"""
    w=observed[1]-predictions[0]
    u=observed[2]-predictions[1]
    out=m4(w,u)
    se=out['se']
    valid=se is not None and math.isfinite(se) and se>0
    return dict(theta_hat=out['theta_hat'],se=se,point_ok=True,ci_ok=valid,
                lower=out['theta_hat']-CRITICAL*se if valid else None,
                upper=out['theta_hat']+CRITICAL*se if valid else None,
                tau=out['diagnostics']['tau'],error_message='' if valid else 'original_Huber_invalid_SE')


def evaluate_nuisance(observed,predictions,dgp):
    """仅评价调用真实结构函数；不向任何拟合/调参函数返回或传入真值。"""
    X,D,Y=observed
    m_true,g_true=nuisance_functions(X,DGP_NAMES[dgp])
    mean_y=THETA_TRUE*m_true+g_true
    return dict(y_mse=float(np.mean((predictions[1]-mean_y)**2)),
                d_mse=float(np.mean((predictions[0]-m_true)**2)),
                y_observed_mse=float(np.mean((predictions[1]-Y)**2)),
                d_observed_mse=float(np.mean((predictions[0]-D)**2)))


def run_replication(dgp,error,rep):
    records=[]
    with threadpool_limits(limits=1):
        observed,folds,_,audit=generate_observed(CONFIG,dgp,error,rep)
        common=dict(dgp=dgp,error=error,rep=rep,data_hash=audit['data_hash'],fold_hash=audit['fold_hash'])
        d_shared=None
        # M6-A/B 的 D 侧先独立完成，确保 A 的 Y 失败不会阻断 B。
        try:
            X,D,Y=observed; F=feature_dictionary(X)
            d_shared=np.empty(len(D)); d_diags=[]
            for k,(train,test) in enumerate(folds):
                splits=list(KFold(INNER_K,shuffle=True,random_state=seed(CONFIG,dgp,error,rep,50,k,0)).split(train))
                d_shared[test],diag=adaptive_prediction(F[train],D[train],F[test],splits,False)
                d_diags.append(dict(target=0,fold=k,**diag))
            d_error=None
        except Exception as exc:
            d_error=f'D nuisance: {type(exc).__name__}: {exc}'
        for method in METHODS:
            record=dict(common,method=method,point_ok=False,ci_ok=False,nuisance_ok=False,
                        theta_hat=None,se=None,error_message='',diagnostics=[])
            started=time.perf_counter()
            try:
                if method=='M4':
                    predictions=crossfit(observed,folds,CONFIG,dgp,error,rep)
                    diagnostics=[]
                else:
                    if d_error is not None:
                        raise ArithmeticError(d_error)
                    predictions,diagnostics=regularized_crossfit(observed,folds,CONFIG,dgp,error,rep,
                                                                 robust_y=method=='M6-B',d_predictions=d_shared)
                    np.testing.assert_array_equal(predictions[0],d_shared)
                    diagnostics=d_diags+diagnostics
                record.update(nuisance_ok=True,diagnostics=diagnostics,
                              prediction_hash=sha(predictions.tobytes()),
                              d_prediction_hash=sha(predictions[0].tobytes()))
                # 先完成估计；真值仅用于下方独立误差与覆盖率评价。
                record.update(estimate_from_predictions(observed,predictions))
                record.update(evaluate_nuisance(observed,predictions,dgp))
                record['estimation_error']=record['theta_hat']-THETA_TRUE
                if record['ci_ok']:
                    record['covered']=record['lower']<=THETA_TRUE<=record['upper']
                    record['ci_length']=record['upper']-record['lower']
            except Exception as exc:
                record['error_message']=f'{type(exc).__name__}: {exc}'
            record['elapsed_seconds']=time.perf_counter()-started
            records.append(record)
    return records


def simulate(reps,description):
    tasks=((d,e,r) for d in range(2) for e in ERROR_IDS for r in reps)
    total=4*len(reps)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        stream=Parallel(n_jobs=N_JOBS,return_as='generator_unordered',pre_dispatch=N_JOBS,batch_size=1)(
            delayed(run_replication)(d,e,r) for d,e,r in tasks)
        bundles=list(tqdm(stream,total=total,desc=description,unit='次',mininterval=3))
    return [r for bundle in bundles for r in bundle]


def check_smoke(records):
    """功能与最优性检查，不把 coverage 高低当作程序正确标准。"""
    if len(records)!=4*SMOKE_R*3:
        raise RuntimeError('smoke 记录数错误')
    failures=[r for r in records if not r['point_ok'] or not r['ci_ok']]
    if failures:
        for r in failures:
            print('SMOKE FAILURE',r['dgp'],r['error'],r['rep'],r['method'],r['error_message'],flush=True)
        raise RuntimeError('smoke 存在未解释的收敛/计算失败；停止，不自动进入 R=200')
    for d in range(2):
        for e in ERROR_IDS:
            for rep in range(SMOKE_R):
                group=[r for r in records if (r['dgp'],r['error'],r['rep'])==(d,e,rep)]
                if len(group)!=3 or len({r['data_hash'] for r in group})!=1 or len({r['fold_hash'] for r in group})!=1:
                    raise RuntimeError('三方法数据/外层 folds 不一致')
                a=next(r for r in group if r['method']=='M6-A')
                b=next(r for r in group if r['method']=='M6-B')
                if a['d_prediction_hash']!=b['d_prediction_hash']:
                    raise RuntimeError('M6-A/B 的 D 预测不同')
    max_kkt=max(diag['max_kkt'] for r in records for diag in r['diagnostics'])
    if max_kkt>KKT_TOL:
        raise RuntimeError('smoke KKT 检查失败')
    print(f'SMOKE PASSED: 12 份数据、36 个方法结果；共享数据/folds/D 预测一致，最大 KKT={max_kkt:.3g}。',flush=True)
    print('smoke 使用原 full 种子，作为正式运行每组最初 3 次复用，不额外重拟合。',flush=True)


def summarize(records):
    rows=[]
    for d in range(2):
        for e in ERROR_IDS:
            for method in METHODS:
                group=sorted((r for r in records if (r['dgp'],r['error'],r['method'])==(d,e,method)),key=lambda r:r['rep'])
                if len(group)!=R or [r['rep'] for r in group]!=list(range(R)):
                    raise RuntimeError('正式运行重复数/编号不完整')
                point=[r for r in group if r['point_ok']]
                ci=[r for r in group if r['ci_ok']]
                nuisance=[r for r in group if 'y_mse' in r]
                errors=np.array([r['theta_hat']-THETA_TRUE for r in point])
                cov=float(np.mean([r['covered'] for r in ci])) if ci else np.nan
                row=dict(dgp=d,error=e,method=method,attempted=R,n_point=len(point),n_ci=len(ci),
                         n_nuisance=len(nuisance),success_rate=len(ci)/R,
                         bias=float(errors.mean()) if len(point) else np.nan,
                         rmse=float(np.sqrt(np.mean(errors**2))) if len(point) else np.nan,
                         coverage=cov,mean_se=float(np.mean([r['se'] for r in ci])) if ci else np.nan,
                         ci_length=float(np.mean([r['ci_length'] for r in ci])) if ci else np.nan,
                         cov_mcse=math.sqrt(cov*(1-cov)/len(ci)) if ci else np.nan,
                         failure_counts=Counter(r['error_message'] for r in group if r['error_message']))
                for key in ('y_mse','d_mse','y_observed_mse','d_observed_mse'):
                    row[key]=float(np.mean([r[key] for r in nuisance])) if nuisance else np.nan
                diagnostics=[diag for r in group for diag in r['diagnostics']]
                row['max_kkt']=max((x['max_kkt'] for x in diagnostics),default=np.nan)
                row['grid_edge_rate']=float(np.mean([x['grid_edge'] for x in diagnostics])) if diagnostics else np.nan
                rows.append(row)
    for row in rows:
        base=next(x for x in rows if (x['dgp'],x['error'],x['method'])==(row['dgp'],0,row['method']))
        row['deterioration_pp']=100*(base['coverage']-row['coverage'])
    return rows


def print_results(rows,records):
    print('\n正式结果：n=1000, p=5, R=200/场景，三方法共享相同数据和外层 5 折。')
    print('nuisance MSE 是对真实条件均值的 OOF 预测误差，真值只用于评价。')
    print('DGP  Error     Method Point/CI   Success      Bias        RMSE      Coverage   Mean SE     CI length    Y MSE       D MSE')
    for q in rows:
        print(f"{q['dgp']+1:3d}  {ERROR_LABELS[q['error']]:9s} {q['method']:5s} "
              f"{q['n_point']:3d}/{q['n_ci']:<3d} {q['success_rate']:8.1%} "
              f"{q['bias']:+.6f}   {q['rmse']:.6f}   {q['coverage']:8.1%}   "
              f"{q['mean_se']:.6f}   {q['ci_length']:.6f}   {q['y_mse']:.6f}   {q['d_mse']:.6f}")
    print('\nGaussian -> t(2.1) coverage deterioration (百分点):')
    for q in rows:
        if q['error']==7:
            print(f"DGP{q['dgp']+1} {q['method']}: {q['deterioration_pp']:+.1f} pp")
    print('\n第一阶段诊断（补充：对含噪观测值的 MSE；不作为 oracle 训练目标）：')
    for q in rows:
        print(f"DGP{q['dgp']+1} {ERROR_LABELS[q['error']]} {q['method']}: "
              f"Y observed MSE={q['y_observed_mse']:.6f}, D observed MSE={q['d_observed_mse']:.6f}, "
              f"max KKT={q['max_kkt']:.3g}, lambda 网格边界选择率={q['grid_edge_rate']:.1%}")
    print('\n相对 M4 的配对比较（共同有效重复；MSE 比 <1 表示第一阶段改善）：')
    for d in range(2):
        for e in ERROR_IDS:
            original={r['rep']:r for r in records if (r['dgp'],r['error'],r['method'])==(d,e,'M4') and r['ci_ok']}
            for method in METHODS[1:]:
                alternatives={r['rep']:r for r in records if (r['dgp'],r['error'],r['method'])==(d,e,method) and r['ci_ok']}
                pair=sorted(set(original)&set(alternatives))
                if not pair:
                    print(f'DGP{d+1} {ERROR_LABELS[e]} {method}: 无共同有效重复');continue
                ym=np.mean([alternatives[r]['y_mse'] for r in pair])/np.mean([original[r]['y_mse'] for r in pair])
                dm=np.mean([alternatives[r]['d_mse'] for r in pair])/np.mean([original[r]['d_mse'] for r in pair])
                diff=np.array([float(alternatives[r]['covered'])-float(original[r]['covered']) for r in pair])
                mcse=float(np.std(diff,ddof=1)/np.sqrt(len(pair))) if len(pair)>1 else np.nan
                print(f'DGP{d+1} {ERROR_LABELS[e]} {method}: N={len(pair)}, Y MSE ratio={ym:.4f}, D MSE ratio={dm:.4f}, '
                      f'Coverage 差={100*diff.mean():+.1f} pp (配对 MCSE={100*mcse:.2f} pp)')
    failures=[r for r in records if r['error_message']]
    print(f'\n失败记录数：{len(failures)}/{len(records)}；未回退、未删除、未换样本。')
    if failures:
        for (method,message),count in Counter((r['method'],r['error_message']) for r in failures).most_common():
            print(f'{method} x{count}: {message}')
    print('Coverage 按有效区间计算；有效次数与成功率如表，Bias/RMSE 按有效点估计计算。')
    print('R=200 时 Coverage 步长 0.5 个百分点；本实验不证明任何 DML Oracle/推断性质。')


def configure_plots():
    available={f.name for f in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun'):
        if name in available:
            plt.rcParams['font.sans-serif']=[name,'DejaVu Sans'];break
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f5f7fb',
                         'axes.facecolor':'white','savefig.facecolor':'#f5f7fb'})


def add_table(ax,columns,data,widths=None):
    ax.axis('off')
    table=ax.table(cellText=data,colLabels=columns,loc='center',cellLoc='center',
                   colWidths=widths,bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#d6dfed')
        cell.set_facecolor('#203552' if i==0 else ('white' if i%2 else '#eaf0f8'))
        if i==0:cell.set_text_props(color='white',weight='bold')
    return table


def table_png(rows,path):
    fig=plt.figure(figsize=(21,15))
    fig.suptitle('第一阶段 Adaptive Lasso：能否改善 Huber-DML？',fontsize=23,weight='bold',y=.97)
    fig.text(.5,.935,'原 DGP / 种子 / 二阶段 Huber 不变；n=1000，p=5，5-fold，每场景 R=200',ha='center',fontsize=13)
    grid=fig.add_gridspec(3,1,left=.028,right=.972,top=.9,bottom=.13,height_ratios=[2.8,2.35,1.0],hspace=.23)
    ax=fig.add_subplot(grid[0]);ax.set_title('最终估计与推断（Coverage 按有效区间计算）',loc='left',pad=12)
    data=[]
    for q in rows:
        data.append([DGP_LABELS[q['dgp']],ERROR_LABELS[q['error']],q['method'],
                     f"{q['n_point']}/{q['n_ci']}",f"{100*q['success_rate']:.1f}%",f"{q['bias']:+.5f}",
                     f"{q['rmse']:.5f}",f"{100*q['coverage']:.1f}%",f"{q['mean_se']:.5f}",f"{q['ci_length']:.5f}"])
    add_table(ax,['DGP','误差','方法','点/区间 N','成功率','Bias','RMSE','Coverage','Mean SE','平均 CI length'],data,
              [.105,.09,.075,.09,.085,.11,.10,.10,.10,.145])
    ax=fig.add_subplot(grid[1]);ax.set_title('第一阶段 OOF 预测 MSE：对真实 E[Y|X] 和 E[D|X]，仅评价使用真值',loc='left',pad=12)
    data=[]
    for q in rows:
        base=next(x for x in rows if (x['dgp'],x['error'],x['method'])==(q['dgp'],q['error'],'M4'))
        data.append([DGP_LABELS[q['dgp']],ERROR_LABELS[q['error']],q['method'],str(q['n_nuisance']),
                     f"{q['y_mse']:.6f}",f"{q['d_mse']:.6f}",f"{q['y_mse']/base['y_mse']:.3f}",f"{q['d_mse']/base['d_mse']:.3f}"])
    add_table(ax,['DGP','误差','方法','有效 N','Y nuisance MSE','D nuisance MSE','Y MSE / M4','D MSE / M4'],data,
              [.125,.1,.085,.08,.17,.17,.135,.135])
    ax=fig.add_subplot(grid[2]);ax.set_title('Gaussian → t(2.1)：Coverage deterioration（百分点，正值表示下降）',loc='left',pad=12)
    data=[]
    for d in range(2):
        data.append([DGP_LABELS[d]]+[f"{next(q['deterioration_pp'] for q in rows if (q['dgp'],q['error'],q['method'])==(d,7,m)):+.1f} pp" for m in METHODS])
    add_table(ax,['DGP',*METHODS],data,[.25]*4)
    fig.text(.03,.09,'M4：原始 X + HGBR；M6-A：Y/D 均为 Adaptive Lasso；M6-B：Y 为 Huber+Adaptive L1，D 与 M6-A 共用。',fontsize=11)
    fig.text(.03,.065,'M6 共同字典为原变量、平方项、全部两两交互。权重、标准化和 lambda 均只从训练折学习；内层 3 折选择 lambda。',fontsize=11)
    fig.text(.03,.04,'三方法共用原版无截距 Huber、OLS 残差 MAD 阈值、ddof=1 sandwich 与 1.96 临界值；未回退失败方法。',fontsize=11)
    fig.text(.03,.016,'比较同时改变了第一阶段学习器与特征字典，不能把相对 M4 的差异仅归因于正则化；R=200 为本次小规模诊断。',fontsize=10,color='#52617a')
    fig.savefig(path,dpi=160);plt.close(fig)


def comparison_png(rows,path):
    colors=('#61758f','#287fb1','#d17832')
    fig,axes=plt.subplots(4,2,figsize=(17,18))
    fig.suptitle('Adaptive Lasso nuisance + 原版 Huber-DML',fontsize=23,weight='bold',y=.985)
    fig.text(.5,.959,'三个方法共享数据与外层五折；每场景 R=200，n=1000，p=5',ha='center',fontsize=12)
    for d in range(2):
        for i,metric in enumerate(('coverage','rmse','bias')):
            ax=axes[i,d];x=np.arange(2)
            for j,method in enumerate(METHODS):
                qs=[next(q for q in rows if (q['dgp'],q['error'],q['method'])==(d,e,method)) for e in ERROR_IDS]
                values=[q[metric]*(100 if metric=='coverage' else 1) for q in qs]
                bars=ax.bar(x+(j-1)*.24,values,.24,color=colors[j],label=method)
                fmt=(lambda v:f'{v:.1f}%') if metric=='coverage' else ((lambda v:f'{v:+.4f}') if metric=='bias' else (lambda v:f'{v:.4f}'))
                ax.bar_label(bars,labels=[fmt(v) for v in values],padding=4,fontsize=9)
            if metric=='coverage':
                ax.axhline(95,color='#b94343',ls='--',lw=1.1);ax.set_ylim(0,112)
            if metric=='bias':
                ax.axhline(0,color='#7b879b',lw=1);ax.margins(y=.3)
            if metric=='rmse':ax.margins(y=.25)
            ax.set(xticks=x,xticklabels=['Gaussian','t(2.1)'],title=DGP_LABELS[d]+' · '+metric.upper())
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            ax.legend(fontsize=9,loc='best',ncol=3)
        ax=axes[3,d]
        values=[next(q['deterioration_pp'] for q in rows if (q['dgp'],q['error'],q['method'])==(d,7,m)) for m in METHODS]
        bars=ax.bar(np.arange(3),values,.58,color=colors)
        ax.bar_label(bars,labels=[f'{v:+.1f} pp' for v in values],padding=5,fontsize=11)
        ax.axhline(0,color='#7b879b',lw=1);ax.margins(y=.3)
        ax.set(xticks=np.arange(3),xticklabels=METHODS,title=DGP_LABELS[d]+' · Gaussian → t(2.1) Coverage deterioration',ylabel='百分点')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.065,right=.975,bottom=.07,top=.92,hspace=.4,wspace=.20)
    fig.text(.065,.035,'成功率、Mean SE、CI length 与真实 nuisance MSE 见对照表。Coverage deterioration 较小仍需结合绝对覆盖率判断。',fontsize=11)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    started=time.perf_counter()
    print('Adaptive Lasso DML：先 R=3 smoke，通过后运行每场景 R=200。',flush=True)
    print('固定 20 项二次字典，gamma=1，权重稳定项 0.001；13 个 lambda 比例，内层 3 折。',flush=True)
    print('普通路线 Ridge(1) pilot + weighted Lasso；稳健 Y 路线 Huber pilot + Huber/weighted-L1。',flush=True)
    print('KKT tolerance=',KKT_TOL,'；优化失败显式记录，不替换估计方法。',flush=True)
    smoke=simulate(range(SMOKE_R),'R=3 smoke')
    check_smoke(smoke)
    records=smoke+simulate(range(SMOKE_R,R),'R=200 正式模拟')
    rows=summarize(records)
    print_results(rows,records)
    configure_plots()
    desktop=desktop_path()
    table_path=desktop/'Adaptive_Lasso_对照表.png'
    chart_path=desktop/'Adaptive_Lasso_对照图.png'
    table_png(rows,table_path)
    comparison_png(rows,chart_path)
    print(f'\n完成：{4*R} 份数据、{len(records)} 个方法记录。耗时 {(time.perf_counter()-started)/60:.2f} 分钟。')
    print('对照表：',table_path,'\n对照图：',chart_path,flush=True)
    # 保持返回值供同一交互会话核查，直接运行时结束即释放，不落盘。
    return rows


if __name__=='__main__':
    FINAL_SUMMARY=main()

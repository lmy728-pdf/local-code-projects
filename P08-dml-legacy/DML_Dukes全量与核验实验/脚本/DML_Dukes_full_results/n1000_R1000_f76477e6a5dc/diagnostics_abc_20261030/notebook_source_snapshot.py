# READ-ONLY SOURCE SNAPSHOT. Do not execute: contains the original launch cell.

# %% Notebook cell 2; code cell 1
# 运行参数：n 与 R 是不同的量
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

# %% Notebook cell 4; code cell 2
import os
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

# %% Notebook cell 6; code cell 3
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

def generate_data_student(n: int, error_case, dgp_name: str, seed: int, theta0: float = THETA_0):
    """生成一个 Student-t 或 N(0,1) 结果误差下的部分线性模型样本。"""
    rng = np.random.default_rng(seed)
    sigma_x = ar1_covariance(P, rho=0.5)
    X = rng.multivariate_normal(np.zeros(P), sigma_x, size=n)
    m0, g0 = nuisance_functions(X, dgp_name)

    V = rng.normal(0.0, 1.0, size=n)
    if error_case == NORMAL_BASELINE:
        U = rng.normal(0.0, 1.0, size=n)
    else:
        U = standardized_student_t(rng, df=float(error_case), size=n)
    D = m0 + V
    Y = theta0 * D + g0 + U
    return X, D, Y, U

def make_nuisance_model(kind: str, seed: int):
    """根据估计量类型创建条件均值或条件中位数学习器。"""
    common = dict(BOOSTING_PARAMS)
    common["random_state"] = seed
    if kind == "mean":
        return HistGradientBoostingRegressor(loss="squared_error", **common)
    if kind == "median":
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **common)
    raise ValueError("kind 必须是 'mean' 或 'median'。")

def cross_fitted_residuals(X, D, Y, kind: str, n_folds: int, seed: int):
    """K-fold cross-fitting：返回处理与结果的 out-of-fold 残差。"""
    d_hat = np.empty_like(D, dtype=float)
    y_hat = np.empty_like(Y, dtype=float)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        model_d = make_nuisance_model(kind, seed + 100 * fold + 1)
        model_y = make_nuisance_model(kind, seed + 100 * fold + 2)
        model_d.fit(X[train_idx], D[train_idx])
        model_y.fit(X[train_idx], Y[train_idx])
        d_hat[test_idx] = model_d.predict(X[test_idx])
        y_hat[test_idx] = model_y.predict(X[test_idx])

    return D - d_hat, Y - y_hat

def fit_mean_dml(d_res: np.ndarray, y_res: np.ndarray):
    """第二阶段残差 OLS，并由正交得分的经验影响函数计算标准误。"""
    jacobian = np.mean(d_res**2)
    if jacobian <= 1e-12:
        return np.nan, np.nan
    theta_hat = np.mean(d_res * y_res) / jacobian
    score = d_res * (y_res - theta_hat * d_res)
    influence = score / jacobian
    se_hat = np.sqrt(np.var(influence, ddof=1) / len(d_res))
    return float(theta_hat), float(se_hat)

# %% Notebook cell 7; code cell 4
def lognormal_ar1_covariance(p: int, rho: float = 0.5) -> np.ndarray:
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])

def lognormal_nuisance_functions(X: np.ndarray, dgp_name: str):
    x1, x2, x3, x4, x5 = X.T
    if dgp_name == "DGP1_linear":
        m0 = 0.80*x1 - 0.50*x2 + 0.30*x3 + 0.20*x4 - 0.10*x5
        g0 = 1.00*x1 + 0.50*x2 - 0.50*x3 + 0.25*x4 + 0.10*x5
    elif dgp_name == "DGP2_nonlinear":
        m0 = (0.60*np.sin(x1) + 0.35*(x2**2 - 1.0)
              + 0.25*x3*x4 - 0.20*np.tanh(x5))
        g0 = (1.20*np.sin(x1) + 0.70*(x2**2 - 1.0)
              + 0.50*x3*x4 + 0.30*np.tanh(x5))
    else:
        raise ValueError(f"未知 DGP: {dgp_name}")
    return m0, g0

def standardized_centered_lognormal(rng: np.random.Generator, sigma: float, size: int):
    """生成理论均值 0、方差 1 的右偏 lognormal 误差。"""
    if sigma <= 0:
        raise ValueError("lognormal sigma 必须大于 0。")
    z = rng.normal(0.0, 1.0, size=size)
    w = np.exp(sigma * z)
    mean_w = np.exp(0.5 * sigma**2)
    sd_w = np.sqrt(np.expm1(sigma**2) * np.exp(sigma**2))
    return (w - mean_w) / sd_w

def generate_data_lognormal(n: int, error_case, dgp_name: str, seed: int, theta0: float = THETA_0):
    rng = np.random.default_rng(seed)
    sigma_x = lognormal_ar1_covariance(P, rho=0.5)
    X = rng.multivariate_normal(np.zeros(P), sigma_x, size=n)
    m0, g0 = lognormal_nuisance_functions(X, dgp_name)
    V = rng.normal(0.0, 1.0, size=n)
    if error_case == NORMAL_BASELINE:
        U = rng.normal(0.0, 1.0, size=n)
    else:
        U = standardized_centered_lognormal(rng, float(error_case), n)
    D = m0 + V
    Y = theta0 * D + g0 + U
    return X, D, Y, U

def generate_data(n, error_case, dgp_name, seed, theta0=THETA_0, error_family="student_t"):
    if error_family == "lognormal":
        return generate_data_lognormal(n, error_case, dgp_name, seed, theta0)
    if error_family in ("student_t", "gaussian"):
        return generate_data_student(n, "N(0,1)" if error_family == "gaussian" else error_case,
                                     dgp_name, seed, theta0)
    raise ValueError(f"Unknown error family: {error_family}")

# %% Notebook cell 9; code cell 5
DEFAULT_KERNEL_CONFIG = {
    "implementation": "python_gaussian_nw_loocv_v1",
    "kernel": "gaussian_order2", "regtype": "local_constant", "bwtype": "fixed",
    "cv": "leave_one_out_squared_error", "scale": "sample_sd_times_fold_n_power_minus_0.2",
    "lower_multiplier": 0.05, "upper_multiplier": 20.0, "grid_size": 41,
    "refine": True, "log_bandwidth_xatol": 1e-6, "refine_maxiter": 80,
    "constant_index": "mean_response", "final_include_self": True,
}

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

def _learner(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=100,
        max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
        random_state=seed,
    )

def crossfit_initial(X, D, Y, theta, folds, crossfit_seed, treatment_predictions=None):
    """Return g_hat=OOF D|X and m_hat=OOF (Y-theta D)|X.

    folds is a list of (train_indices, test_indices). A supplied treatment OOF
    vector is reused verbatim; its caller must establish training provenance.
    """
    D, Y = _finite_vector(D, "D"), _finite_vector(Y, "Y")
    X = np.asarray(X, dtype=float)
    if len(D) != len(Y) or X.ndim != 2 or X.shape[0] != len(D) or not np.all(np.isfinite(X)):
        raise NumericalFailure("invalid_input", "X/D/Y dimensions or finite values invalid")
    if not np.isfinite(theta):
        raise NumericalFailure("invalid_theta", "theta must be finite")
    folds = validate_folds(folds, len(D))
    g_hat = np.empty(len(D)) if treatment_predictions is None else _finite_vector(treatment_predictions, "g_hat").copy()
    if len(g_hat) != len(D):
        raise NumericalFailure("invalid_input", "Treatment prediction length differs")
    m_hat = np.empty(len(D))
    response = Y - theta * D
    for k, (train, test) in enumerate(folds):
        try:
            if treatment_predictions is None:
                g_hat[test] = _learner(crossfit_seed + 100 * k + 1).fit(X[train], D[train]).predict(X[test])
            m_hat[test] = _learner(crossfit_seed + 100 * k + 2).fit(X[train], response[train]).predict(X[test])
        except Exception as exc:
            raise NumericalFailure("initial_fit", str(exc), k) from exc
    _finite_vector(g_hat, "g_hat")
    _finite_vector(m_hat, "m_hat")
    return {"g_hat": g_hat, "m_hat": m_hat}

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

def score_test(scores):
    scores = _finite_vector(scores, "scores")
    if len(scores) < 2:
        raise NumericalFailure("insufficient_scores", "At least two scores required")
    mean, sd = float(np.mean(scores)), float(np.std(scores, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise NumericalFailure("zero_or_nonfinite_score_variance", "Score SD is zero or nonfinite")
    statistic = float(np.sqrt(len(scores)) * mean / sd)
    if not np.isfinite(statistic):
        raise NumericalFailure("nonfinite_statistic", "Test statistic is nonfinite")
    return {"score_mean": mean, "score_sd": sd, "score_variance": sd*sd, "T": statistic,
            "p_value": float(2 * norm.sf(abs(statistic))),
            "accepted": bool(abs(statistic) <= norm.ppf(0.975))}

# %% Notebook cell 11; code cell 6
METHODS = ("original_mean_dml", "standard_score", "dukes_dr_gcm")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def dump_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def make_data(config, scenario, replication):
    seed = config["base_seed"] + 100000 * config["legacy_dgp_id"] + 1000 * scenario["legacy_error_id"] + replication
    family = "lognormal" if scenario["error_family"] == "lognormal" else "student_t"
    data = generate_data(config["n"], scenario["error_case"], config["dgp"], seed, theta0=config["theta0"], error_family=family)
    folds = list(KFold(config["n_folds"], shuffle=True, random_state=seed + config["fold_seed_offset"]).split(data[0]))
    return data, folds, seed

def initial_a(X, D, Y, folds, seed):
    d_hat, y_hat = np.empty_like(D), np.empty_like(Y)
    for k, (train, test) in enumerate(folds):
        model_d = make_nuisance_model("mean", seed + 100 * k + 1)
        model_y = make_nuisance_model("mean", seed + 100 * k + 2)
        model_d.fit(X[train], D[train])
        model_y.fit(X[train], Y[train])
        d_hat[test] = model_d.predict(X[test])
        y_hat[test] = model_y.predict(X[test])
    return d_hat, y_hat

def compute_task(config, scenario, replication):
    started = time.perf_counter()
    data, folds, seed = make_data(config, scenario, replication)
    X, D, Y, U = data
    cfseed = seed + config["fold_seed_offset"]
    fold_ids = np.empty(len(D), dtype=np.int64)
    for k, (train, test) in enumerate(folds):
        assert len(np.intersect1d(train, test)) == 0
        fold_ids[test] = k
    fold_hash = hashlib.sha256(fold_ids.tobytes()).hexdigest()
    base = dict(scenario=scenario["name"], error_family=scenario["error_family"], error_parameter=scenario["error_parameter"], dgp=config["dgp"], legacy_dgp_id=config["legacy_dgp_id"], legacy_error_id=scenario["legacy_error_id"], p=config["p"], n=config["n"], replication=replication, config_hash=config["config_hash"], dataset_seed=seed, crossfit_seed=cfseed, fold_hash=fold_hash, model_seeds=[{"treatment":cfseed+100*k+1,"outcome":cfseed+100*k+2} for k in range(config["n_folds"])], sample_u_mean=float(U.mean()), sample_u_variance=float(U.var(ddof=1)), sample_max_abs_u=float(np.abs(U).max()), sample_max_u=float(U.max()))
    rows, diagnostics = [], []
    arrays = {"fold_ids": fold_ids}
    timings = {}
    def fail(method, error):
        rows.append(dict(base, method=method, status="failed", covered=None, failure_type=getattr(error, "failure_type", type(error).__name__), failure_fold=getattr(error, "fold", None), failure_message=str(error)))
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.perf_counter()
        try:
            g_hat, ell_hat = initial_a(X, D, Y, folds, cfseed)
            dr, yr = D-g_hat, Y-ell_hat
            estimate, se = fit_mean_dml(dr, yr)
            if not np.isfinite([estimate, se]).all() or se <= 0:
                raise ValueError("Non-finite or degenerate A estimate/SE")
            low, high = estimate-1.96*se, estimate+1.96*se
            rows.append(dict(base, method=METHODS[0], status="ok", theta_hat=estimate, theta_se=se, ci_low=low, ci_high=high, ci_length=high-low, covered=bool(low <= config["theta0"] <= high), legacy_covered=bool(low <= config["theta0"] <= high), coverage_type="wald_coverage"))
            arrays.update(g_hat=g_hat, ell_hat=ell_hat, d_res=dr, y_res=yr)
        except Exception as error:
            fail(METHODS[0], error)
            g_hat = None
        timings["a_seconds"] = time.perf_counter()-t
        t = time.perf_counter()
        initial = None
        try:
            initial = crossfit_initial(X, D, Y, config["theta0"], folds, cfseed, treatment_predictions=g_hat)
            arrays.update(g_hat=initial["g_hat"], m_hat=initial["m_hat"])
            scores_b = (D-initial["g_hat"])*(Y-config["theta0"]*D-initial["m_hat"])
            test_b = score_test(scores_b)
            arrays["scores_B"] = scores_b
            rows.append(dict(base, method=METHODS[1], status="ok", theta_tested=config["theta0"], coverage_type="inverted_test_coverage_at_truth", covered=bool(test_b["accepted"]), **test_b))
        except Exception as error:
            fail(METHODS[1], error)
        timings["bc_initial_seconds"] = time.perf_counter()-t
        t = time.perf_counter()
        try:
            if initial is None:
                raise ValueError("Initial B/C predictions unavailable")
            result = dukes_from_initial(D, Y, config["theta0"], initial["g_hat"], initial["m_hat"], folds, config["kernel"])
            test_c = score_test(result["scores"])
            arrays.update(result["arrays"])
            arrays["scores_C"] = result["scores"]
            diagnostics = [dict(base, **fold) for fold in result["fold_diagnostics"]]
            rows.append(dict(base, method=METHODS[2], status="ok", theta_tested=config["theta0"], coverage_type="inverted_test_coverage_at_truth", covered=bool(test_c["accepted"]), **test_c))
            assert np.array_equal(result["standard_scores"], scores_b)
        except Exception as error:
            # Avoid a duplicate success if the sharing assertion itself fails.
            rows = [row for row in rows if row["method"] != METHODS[2]]
            fail(METHODS[2], error)
        timings["kernel_update_seconds"] = time.perf_counter()-t
    elapsed = time.perf_counter()-started
    for row in rows:
        row.update(elapsed_seconds=elapsed, warnings=[f"{w.category.__name__}: {w.message}" for w in caught], **timings)
    return {"rows": rows, "folds": diagnostics, "timings": timings, "elapsed_seconds": elapsed}, arrays

def persist_task(config, scenario, replication, run_dir):
    result, arrays = compute_task(config, scenario, replication)
    stem = f"{scenario['name']}_{replication:04d}"
    array_path = Path(run_dir) / "arrays" / (stem + ".npz")
    np.savez_compressed(array_path, **arrays)
    result["arrays_sha256"] = sha256(array_path)
    result["arrays_file"] = str(array_path.relative_to(run_dir))
    dump_json(Path(run_dir) / "tasks" / (stem + ".json"), result)
    return stem, result["elapsed_seconds"], sum(r["status"] != "ok" for r in result["rows"])

# %% Notebook cell 13; code cell 7
METHODS = ("original_mean_dml", "standard_score", "dukes_dr_gcm")
METHOD_LABELS = dict(zip(METHODS, ("A: original Mean-DML", "B: standard score", "C: Dukes DR-GCM (Python)")))
SCENARIO_LABELS = {"gaussian": "Gaussian", "t2.1": "Student-t(2.1)", "lognormal2": "Centered lognormal(2)"}
ESTIMATE_FIELDS = ("theta_hat", "theta_se", "ci_low", "ci_high", "ci_length")
SUMMARY_FIELDS = ["scenario", "method", "coverage_type", "planned", "attempted", "pending", "valid", "failed", "covered_count", "coverage_valid", "failure_rate", "failure_rate_denominator", "failure_fraction_planned", "planned_coverage_lower", "planned_coverage_upper", "mcse", "wilson_low", "wilson_high", "coverage_minus_gaussian", "bias", "estimate_sd", "mean_se", "rmse", "mean_ci_length"]
PAIR_FIELDS = ["scenario", "comparison", "method_left", "method_right", "planned", "common_attempted", "valid_pairs", "left_failed", "right_failed", "nonpaired_attempted", "missing_pairs_planned", "mean_difference", "mcse", "count_minus1", "count_zero", "count_plus1", "planned_difference_lower", "planned_difference_upper"]

def wilson(successes, count, level=0.95):
    """Conditional binomial Wilson interval; count is the valid-run denominator."""
    if count == 0:
        return None, None
    z = NormalDist().inv_cdf((1.0 + level) / 2.0)
    p = successes / count
    den = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / den
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return max(0.0, center - half), min(1.0, center + half)

def scenario_names(config, rows):
    configured = config.get("scenarios", [])
    if isinstance(configured, dict):
        names = list(configured)
    else:
        names = [s if isinstance(s, str) else s.get("name", s.get("id", s.get("scenario"))) for s in configured]
    if any(s is None for s in names):
        raise ValueError("Every configured scenario needs name, id, or scenario")
    return names or list(dict.fromkeys(r["scenario"] for r in rows))

def validate_records(rows, config):
    """Refuse duplicated keys, mixed configurations, and fabricated B/C estimates."""
    planned = int(config["replications"])
    if planned < 1:
        raise ValueError("replications must be positive")
    allowed_scenarios = set(scenario_names(config, rows))
    keys, hashes = set(), set()
    for row in rows:
        key = (row["scenario"], int(row["replication"]), row["method"])
        if key in keys:
            raise ValueError(f"Duplicate task/method key: {key}")
        keys.add(key)
        if key[0] not in allowed_scenarios or key[2] not in METHODS or not 0 <= key[1] < planned:
            raise ValueError(f"Unexpected scenario, method, or replication: {key}")
        if row.get("status") not in ("ok", "failed"):
            raise ValueError(f"Unknown status: {key}")
        if row["status"] == "ok" and not isinstance(row.get("covered"), bool):
            raise ValueError(f"Successful result must have a Boolean coverage indicator: {key}")
        if row["status"] == "failed" and row.get("covered") is not None:
            raise ValueError(f"A failed method must have covered=null: {key}")
        if row["method"] != METHODS[0] and any(row.get(field) is not None for field in ESTIMATE_FIELDS):
            raise ValueError(f"B/C point estimates and interval endpoints were not computed: {key}")
        if row.get("config_hash"):
            hashes.add(row["config_hash"])
    if len(hashes) > 1 or (hashes and config.get("config_hash") and hashes != {config["config_hash"]}):
        raise ValueError("Cannot merge results from different configurations")

def summarize_records(rows, config):
    validate_records(rows, config)
    planned = int(config["replications"])
    truth = float(config.get("theta0", 1.0))
    summary = []
    for scenario in scenario_names(config, rows):
        for method in METHODS:
            subset = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            valid = [r for r in subset if r["status"] == "ok"]
            attempted, nv = len(subset), len(valid)
            failed = attempted - nv
            successes = sum(r["covered"] for r in valid)
            coverage = successes / nv if nv else None
            lo, hi = wilson(successes, nv)
            item = dict.fromkeys(SUMMARY_FIELDS)
            item.update(scenario=scenario, method=method,
                        coverage_type="wald_coverage" if method == METHODS[0] else "inverted_test_coverage_at_truth",
                        planned=planned, attempted=attempted, pending=planned-attempted,
                        valid=nv, failed=failed, covered_count=successes, coverage_valid=coverage,
                        failure_rate=failed/attempted if attempted else None,
                        failure_rate_denominator="attempted", failure_fraction_planned=failed/planned,
                        planned_coverage_lower=successes/planned,
                        planned_coverage_upper=(successes+planned-nv)/planned,
                        mcse=math.sqrt(coverage*(1-coverage)/nv) if nv else None,
                        wilson_low=lo, wilson_high=hi)
            if method == METHODS[0] and valid:
                estimates = [float(r["theta_hat"]) for r in valid]
                if not all(math.isfinite(x) for x in estimates):
                    raise ValueError("Successful A point estimates must be finite")
                item.update(bias=mean(estimates)-truth,
                            estimate_sd=stdev(estimates) if nv > 1 else None,
                            mean_se=mean(float(r["theta_se"]) for r in valid),
                            rmse=math.sqrt(mean((x-truth)**2 for x in estimates)),
                            mean_ci_length=mean(float(r["ci_length"]) for r in valid))
            summary.append(item)
    gaussian = {r["method"]: r["coverage_valid"] for r in summary if r["scenario"] == "gaussian"}
    for row in summary:
        base = gaussian.get(row["method"])
        if row["coverage_valid"] is not None and base is not None:
            row["coverage_minus_gaussian"] = row["coverage_valid"] - base
    return summary

def paired_records(rows, config):
    validate_records(rows, config)
    planned = int(config["replications"])
    pairs = []
    for scenario in scenario_names(config, rows):
        lookup = {m: {int(r["replication"]): r for r in rows if r["scenario"] == scenario and r["method"] == m} for m in METHODS}
        for left, right, label in ((METHODS[2], METHODS[1], "C-B"), (METHODS[2], METHODS[0], "C-A"), (METHODS[1], METHODS[0], "B-A")):
            ls, rs = lookup[left], lookup[right]
            common = set(ls) & set(rs)
            good = [k for k in sorted(common) if ls[k]["status"] == rs[k]["status"] == "ok"]
            delta = [int(ls[k]["covered"])-int(rs[k]["covered"]) for k in good]
            n = len(delta)
            missing = planned-n
            pairs.append(dict(scenario=scenario, comparison=label, method_left=left, method_right=right,
                              planned=planned, common_attempted=len(common), valid_pairs=n,
                              left_failed=sum(r["status"] == "failed" for r in ls.values()),
                              right_failed=sum(r["status"] == "failed" for r in rs.values()),
                              nonpaired_attempted=len(set(ls)|set(rs))-n, missing_pairs_planned=missing,
                              mean_difference=mean(delta) if n else None,
                              mcse=stdev(delta)/math.sqrt(n) if n > 1 else None,
                              count_minus1=delta.count(-1), count_zero=delta.count(0), count_plus1=delta.count(1),
                              planned_difference_lower=(sum(delta)-missing)/planned,
                              planned_difference_upper=(sum(delta)+missing)/planned))
    return pairs

def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)

def write_csv(path, rows, preferred=()):
    fields = list(preferred)
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    if not fields:
        fields = ["scenario", "replication", "method", "status", "failure_type", "failure_fold"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    temporary.replace(path)

def plot_comparison(path, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenarios = list(dict.fromkeys(r["scenario"] for r in summary))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.5*len(scenarios), 6), squeeze=False, sharey=True)
    colors = ("#486581", "#D99724", "#247C72")
    for ax, scenario in zip(axes[0], scenarios):
        subset = [r for r in summary if r["scenario"] == scenario]
        for j, row in enumerate(subset):
            c = row["coverage_valid"]
            if c is not None:
                ax.errorbar(j, c, yerr=[[max(0, c-row["wilson_low"])], [max(0, row["wilson_high"]-c)]], fmt="o", color=colors[j], markersize=9, capsize=5)
                ax.annotate(f"{c:.3f}", (j, row["wilson_high"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10)
            ax.text(j, -.18, f"valid {row['valid']}/{row['planned']}\nfailed {row['failed']}; pending {row['pending']}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, clip_on=False)
        ax.axhline(.95, linestyle="--", linewidth=1.1, color="#9B3547")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario), fontsize=12)
        ax.set_xticks(range(3), ["A\nWald", "B\nScore inversion", "C\nDR score inversion"])
        ax.set_ylim(0, 1.08)
        ax.set_xlim(-.5, 2.5)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Coverage among valid runs (Wilson 95% interval)")
    fig.suptitle("DGP2 | n = 1000, p = 5 | truth-only coverage screening", fontsize=14)
    fig.text(.5, .015, "Dashed line: 0.95. B/C evaluate inclusion of the truth; interval endpoints and widths are not computed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(path)

def pct(value):
    return "未计算" if value is None else f"{100*value:.1f}%"

def pp(value):
    return "未计算" if value is None else f"{100*value:+.1f} 个百分点"

def diagnostic_counts(folds):
    """Count explicitly marked bandwidth boundaries, with no guessing from h values."""
    counts = {"boundary": 0, "known": 0, "constant": 0}
    def walk(record):
        for key, value in record.items():
            lower = key.lower()
            if isinstance(value, dict):
                walk(value)
            elif "boundary" in lower and "side" not in lower:
                if isinstance(value, bool):
                    counts["known"] += 1
                    counts["boundary"] += int(value)
            elif "constant" in lower and isinstance(value, bool):
                counts["constant"] += int(value)
    for row in folds:
        walk(row)
    return counts

def decision_note(summary, pairs, folds, run_dir):
    complete = all(r["pending"] == 0 for r in summary)
    failures = sum(r["failed"] for r in summary)
    c_rows = [r for r in summary if r["method"] == METHODS[2]]
    diag = diagnostic_counts(folds)
    frequent_boundary = diag["known"] > 0 and diag["boundary"]/diag["known"] >= .2
    under = [SCENARIO_LABELS.get(r["scenario"], r["scenario"]) for r in c_rows if r["wilson_high"] is not None and r["wilson_high"] < .95]
    if not complete:
        lead = "筛查尚未完成，本文件是进度报告，不能据此决定是否更换选题。所有计划重复将按预先固定的配置完成，不能按中途 coverage 选停止点。"
    elif failures:
        lead = "本轮计划重复已完成尝试，但存在算法失败；先核查失败与实现，再决定选题。下述有效运行上的 coverage 与配对差不代表完整计划样本上的无条件性能。"
    elif frequent_boundary:
        lead = "本轮筛查已完成，但辅助带宽较多命中有限搜索边界，结果存在实现敏感性，暂不据此作更换选题的决定。最有价值的下一步是固定若干折，对照作者 R/np 实现核验带宽与完整 score。"
    elif under:
        lead = "本轮 Python 移植筛查中，C 在 " + "、".join(under) + " 的 Wilson 区间上界仍低于 95%，留下欠覆盖信号。须结合已完成的基准与方法验证讨论剩余问题，不能直接宣称发现新的研究问题，也不自动扩展到 Huber。"
    else:
        lead = "本轮特定三场景中，C 没有显示 Wilson 区间整体低于 95% 的明确欠覆盖信号。若相对 A/B 的改善与实现验证相符，可支持降低继续原选题的优先级；这不能证明等价于 95%，也不能证明全部重尾 DML 问题已经解决。"
    if complete and all(r["coverage_valid"] is not None for r in c_rows):
        absolute = "、".join(f"{SCENARIO_LABELS.get(r['scenario'], r['scenario'])} {pct(r['coverage_valid'])}" for r in c_rows)
        by_pair = {(r["scenario"], r["comparison"]): r for r in pairs}
        cb = [by_pair[(r["scenario"], "C-B")] for r in c_rows]
        ca = [by_pair[(r["scenario"], "C-A")] for r in c_rows]
        cb_text = "、".join(pp(r["mean_difference"]) for r in cb)
        cb_mcse = "、".join("未计算" if r["mcse"] is None else f"{100*r['mcse']:.2f} 个百分点" for r in cb)
        ca_text = "、".join(pp(r["mean_difference"]) for r in ca)
        evidence = f"已完成预先固定的 {sum(r['planned'] for r in c_rows)} 个数据集。C 的 coverage 依次为 {absolute}；相对 B 的配对改善依次为 {cb_text}（配对 MCSE：{cb_mcse}），相对 A 的配对差依次为 {ca_text}。"
        if len(under) == len(c_rows):
            evidence += "三个场景都仍明显远离 95%，这不支持“此实现已解决当前设计的欠覆盖”这一判断。"
        evidence += "\n\n"
        lead = evidence + lead
        if frequent_boundary:
            lead += "带宽边界频繁出现且尚无作者 R 软件对照，不能把本移植结果直接归结为原论文方法失败，也不能据此断言选题已经解决或必然值得继续。"
    lines = ["# 本轮选题筛查记录", "", lead, "", "## 绝对覆盖率与 Monte Carlo 不确定性", "",
             "A 是原 Mean-DML Wald 区间；B/C 是在真值 θ=1 处检验所对应反演置信集的包含率。下表 Wilson 95% 区间和 MCSE 均以有效运行数为分母。", "",
             "| 场景 | 方法 | 有效/已尝试/计划 | 失败/待运行 | coverage | MCSE | Wilson 95% | 全计划覆盖率缺失界限 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        label = "ABC"[METHODS.index(r["method"])]
        lines.append(f"| {SCENARIO_LABELS.get(r['scenario'], r['scenario'])} | {label} | {r['valid']}/{r['attempted']}/{r['planned']} | {r['failed']}/{r['pending']} | {pct(r['coverage_valid'])} | {pct(r['mcse'])} | [{pct(r['wilson_low'])}, {pct(r['wilson_high'])}] | [{pct(r['planned_coverage_lower'])}, {pct(r['planned_coverage_upper'])}] |")
    lines += ["", "尚未运行的重复属于 pending，不是算法失败。failure_rate 的分母是 attempted；所有计划运行完成时等于 (R−Rv)/R。全计划界限把失败及尚未运行的结果视为未知，不把它们记成成功覆盖。", "", "## 同一数据集上的配对比较", "",
              "| 场景 | 差值 | 有效配对 | 差值 | 配对 MCSE | d=−1/0/+1 次数 | 全计划保守界限 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append(f"| {SCENARIO_LABELS.get(r['scenario'], r['scenario'])} | {r['comparison']} | {r['valid_pairs']} | {pp(r['mean_difference'])} | {pct(r['mcse'])} | {r['count_minus1']}/{r['count_zero']}/{r['count_plus1']} | [{pp(r['planned_difference_lower'])}, {pp(r['planned_difference_upper'])}] |")
    lines += ["", "A→C 包含学习目标变化与修正两个影响；B→C 才更直接反映修正步骤。差值只在共同有效的重复上逐对计算，MCSE 使用配对差的样本标准差。缺失配对的差值按 [−1,1] 给出保守界限；共同成功子集可能偏向较容易样本。少量不一致配对时，正态近似应谨慎。", "", "## 重尾相对 Gaussian 的变化", "",
              "以下是同方法 coverage_tail−coverage_Gaussian 的描述差，未提供跨场景差值标准误。不同误差家族不能仅凭 replication 编号被当作同一个数据集。", ""]
    for r in summary:
        if r["scenario"] != "gaussian":
            lines.append(f"- {SCENARIO_LABELS.get(r['scenario'],r['scenario'])}，{'ABC'[METHODS.index(r['method'])]}：{pp(r['coverage_minus_gaussian'])}。")
    lines += ["", "## 正确性与数值诊断", ""]
    validation = run_dir / "audit" / "validation.md"
    if validation.exists():
        lines += ["原实现一致性、相同样本与 folds、串并行复现和方法代数检查详见 [validation.md](../../audit/validation.md)；本汇总器不把单靠 coverage 接近旧汇总当作复现证据。"]
    else:
        lines += ["本 notebook 的原型脚本已做过原函数一致性与公式校验；notebook 交付时仅做静态检查、未执行实验，当前实际完成数以上表为准。此汇总不能冒称与作者 R/np 完成软件对照。"]
    lines += [f"本轮按方法统计的失败记录为 {failures} 条。带宽诊断中明确记录边界状态 {diag['known']} 项，命中边界 {diag['boundary']} 项；常数索引标记 {diag['constant']} 项。报告将边界比例 ≥20% 作为工程敏感性提醒（报告规则，不是新的显著性检验），不据此调带宽追逐 95%。", "",
              "本实现是明示搜索规则的 Python 核回归移植，与作者 R/np 带宽优化未做逐项软件复现；独立固定数组公式验证不能替代作者环境交叉核验。算法失败、非有限值与零 score 方差不会被记为接受真值。", "", "## 决策范围与后续价值", "",
              "本轮只评价真值是否被包含，尚未求置信集端点、宽度、信息量、功效或非连通性。高 coverage 也可能由过度保守产生，不能宣称推断问题全面解决。R=1000 时 95% 附近的 MCSE 约 0.69 个百分点；应以实际区间和配对误差判断，区间包含 0.95 不是等价证明。三场景属于探索性筛查，未进行多重比较的正式推断。", "",
              "如果边界或软件差异影响判断，优先核验作者实现；若只剩 coverage 精度不足，最小后续是补同一 n=1000 的 R。研究样本量趋势才考虑 n=2000，可推广性才考虑 DGP1。后续扩展需要用户选择投入方向，本轮不自动扩大实验或实现 Huber。", "",
              "Dukes 修正主要处理 nuisance estimation 带来的推断偏差，并非有界影响的重尾专用方法。其有效性仍依赖矩、收敛、经验过程与非退化等条件；有限场景模拟不能证明固定 HGBR 满足理论条件。t(2.1) 有有限方差而无四阶矩，lognormal 有全部正阶矩但有限样本可能非常不稳定。", ""]
    saved_path = run_dir / "audit" / "saved_score_verification.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8-sig"))
        if saved.get("status") == "passed" and Path(saved.get("run_dir", "")).resolve() == run_dir.resolve():
            counts = saved["counts"]
            maxima = saved["maximum_absolute_differences"]
            verification = ["", "## 保存数组的独立复核与带宽位置", "",
                            f"[saved_score_verification.json](../../audit/saved_score_verification.json) 对已保存的 {counts['tasks_read']} 个数据集进行了独立公式复核：{counts['successful_C_verified']} 个成功 C 结果、{counts['scores_checked']:,} 个逐点 score、{counts['folds_checked']:,} 折均通过。核验范围包括折内直接投影、完整 score 方差、T 及与卡方表达式一致的 p 值；没有重新拟合带宽，也没有运行 R。",
                            "",
                            f"score 最大绝对差 {maxima['score_absolute']:.3g}，T 最大绝对差 {maxima['T_absolute']:.3g}，p 值最大绝对差 {maxima['p_absolute']:.3g}；更新矩条件最大绝对残差 {maxima['update_moment_absolute']:.3g}。这支持保存结果与已声明公式一致，不能单独验证有限带宽搜索规则与作者软件等价。",
                            "", "| 场景 | 辅助方向 | 核回归数 | 搜索内部 | 下边界 | 上边界 | 常数索引 |", "|---|---|---:|---:|---:|---:|---:|"]
            for r in saved["bandwidth_counts"]:
                verification.append(f"| {SCENARIO_LABELS.get(r['scenario'], r['scenario'])} | {r['direction']} | {r['total']} | {r['interior']} | {r['lower']} | {r['upper']} | {r['constant_index']} |")
            verification += ["", "G 是结果残差对处理预测的核回归，M 是处理残差对变换结果预测的核回归。多数边界命中位于上边界，可能反映较平坦的辅助关系，也可能受有限优化范围影响；这里不擅自扩大搜索范围或重新挑选结果。最小后续应固定若干已保存折，让作者 R/np 选择带宽，再在相同带宽下比对逐点预测、更新和完整 score。", ""]
            insertion = lines.index("## 决策范围与后续价值")
            lines[insertion:insertion] = verification
    return "\n".join(lines)

def summarize_run(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    rows, folds = [], []
    for task_path in sorted((run_dir / "tasks").glob("*.json")):
        task = json.loads(task_path.read_text(encoding="utf-8-sig"))
        rows.extend(task["rows"])
        folds.extend(task.get("folds", []))
    summary = summarize_records(rows, config)
    pairs = paired_records(rows, config)
    rows.sort(key=lambda r: (r["scenario"], r["replication"], METHODS.index(r["method"])))
    write_csv(run_dir / "raw_results.csv", rows, ("scenario", "replication", "method", "status", "covered"))
    write_csv(run_dir / "fold_diagnostics.csv", folds)
    write_csv(run_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(run_dir / "paired_differences.csv", pairs, PAIR_FIELDS)
    write_csv(run_dir / "failures.csv", [r for r in rows if r["status"] == "failed"])
    plot_comparison(run_dir / "coverage_comparison.png", summary)
    atomic_text(run_dir / "decision_note.md", decision_note(summary, pairs, folds, run_dir))
    return summary, pairs

# %% Notebook cell 15; code cell 8
COMPUTATIONAL_FUNCTIONS = ['ar1_covariance', 'nuisance_functions', 'standardized_student_t', 'generate_data_student', 'make_nuisance_model', 'cross_fitted_residuals', 'fit_mean_dml', 'lognormal_ar1_covariance', 'lognormal_nuisance_functions', 'standardized_centered_lognormal', 'generate_data_lognormal', 'generate_data', '_finite_vector', 'validate_folds', '_learner', 'crossfit_initial', '_nw_from_distances', 'nw_predict', 'select_nw', 'stable_projection', 'corrected_score', 'dukes_from_initial', 'score_test', 'sha256', 'clean', 'dump_json', 'make_data', 'initial_a', 'compute_task', 'persist_task', 'make_config', 'run_experiment', 'current_code_fingerprint']
SOURCE_PROVENANCE = {'_legacy_student.py': '229c8afa816dd4754e43fecb5f8ea81cf3561aca6f83f7bebc522c87b4fbbad9', '_legacy_lognormal.py': '678e1da764e4f8c564154ce2eb649d1f011b80b54070e133442c6efdc98a2e87', 'dukes_score.py': 'bd2869bee3561955f79061be33a555459f7832346ff008bbc4ac9264fec457a6', 'run_pilot.py': 'f34ff07acc40271a4d29668db7800b91e9bd4f17a4a934aba4143ef3e7caf7f4'}

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
            stream.write(json.dumps(event)+"\n")
        return folder
    finally:
        lock.unlink(missing_ok=True)

# %% Notebook cell 17; code cell 9
CONFIG = make_config()
RUN_DIR = run_experiment(CONFIG, workers=WORKERS)
print("正式实验完成：", RUN_DIR)

# %% Notebook cell 19; code cell 10
summary, paired = summarize_run(RUN_DIR)
display(pd.DataFrame(summary)[["scenario","method","planned","attempted","valid","failed","pending",
                               "coverage_valid","mcse","wilson_low","wilson_high"]])
display(pd.DataFrame(paired)[["scenario","comparison","valid_pairs","mean_difference","mcse",
                              "count_minus1","count_zero","count_plus1"]])
display(Image(filename=str(RUN_DIR/"coverage_comparison.png")))
display(Markdown((RUN_DIR/"decision_note.md").read_text(encoding="utf-8")))

"""Verbatim AST function extraction from DML_p5_lognormal_heavytail.ipynb, zero-based cells 4/6.
Source SHA256: e39d6551ccc0269f899f4e71c3a623808dbba00e9fc4b6c17129ff0a057fa354. See audit/extraction_provenance.json.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
P = 5
THETA_0 = 1.0
NORMAL_BASELINE = "N(0,1)"
BOOSTING_PARAMS = {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0}

# Source cell 4, lines 1-3
def ar1_covariance(p: int, rho: float = 0.5) -> np.ndarray:
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


# Source cell 4, lines 6-18
def nuisance_functions(X: np.ndarray, dgp_name: str):
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


# Source cell 4, lines 21-29
def standardized_centered_lognormal(rng: np.random.Generator, sigma: float, size: int):
    """生成理论均值 0、方差 1 的右偏 lognormal 误差。"""
    if sigma <= 0:
        raise ValueError("lognormal sigma 必须大于 0。")
    z = rng.normal(0.0, 1.0, size=size)
    w = np.exp(sigma * z)
    mean_w = np.exp(0.5 * sigma**2)
    sd_w = np.sqrt(np.expm1(sigma**2) * np.exp(sigma**2))
    return (w - mean_w) / sd_w


# Source cell 4, lines 32-44
def generate_data(n: int, error_case, dgp_name: str, seed: int, theta0: float = THETA_0):
    rng = np.random.default_rng(seed)
    sigma_x = ar1_covariance(P, rho=0.5)
    X = rng.multivariate_normal(np.zeros(P), sigma_x, size=n)
    m0, g0 = nuisance_functions(X, dgp_name)
    V = rng.normal(0.0, 1.0, size=n)
    if error_case == NORMAL_BASELINE:
        U = rng.normal(0.0, 1.0, size=n)
    else:
        U = standardized_centered_lognormal(rng, float(error_case), n)
    D = m0 + V
    Y = theta0 * D + g0 + U
    return X, D, Y, U


# Source cell 6, lines 1-8
def make_nuisance_model(kind: str, seed: int):
    common = dict(BOOSTING_PARAMS)
    common["random_state"] = seed
    if kind == "mean":
        return HistGradientBoostingRegressor(loss="squared_error", **common)
    if kind == "median":
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **common)
    raise ValueError("kind 必须是 'mean' 或 'median'。")


# Source cell 6, lines 11-22
def cross_fitted_residuals(X, D, Y, kind: str, n_folds: int, seed: int):
    d_hat = np.empty_like(D, dtype=float)
    y_hat = np.empty_like(Y, dtype=float)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        model_d = make_nuisance_model(kind, seed + 100*fold + 1)
        model_y = make_nuisance_model(kind, seed + 100*fold + 2)
        model_d.fit(X[train_idx], D[train_idx])
        model_y.fit(X[train_idx], Y[train_idx])
        d_hat[test_idx] = model_d.predict(X[test_idx])
        y_hat[test_idx] = model_y.predict(X[test_idx])
    return D - d_hat, Y - y_hat


# Source cell 6, lines 34-42
def fit_mean_dml(d_res: np.ndarray, y_res: np.ndarray):
    jacobian = np.mean(d_res**2)
    if jacobian <= 1e-12:
        return np.nan, np.nan
    theta_hat = np.mean(d_res*y_res) / jacobian
    score = d_res * (y_res - theta_hat*d_res)
    influence = score / jacobian
    se_hat = np.sqrt(np.var(influence, ddof=1) / len(d_res))
    return float(theta_hat), float(se_hat)

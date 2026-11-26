"""Verbatim AST function extraction from DML_p5_student_t_heavytail.ipynb, zero-based cells 4/6.
Source SHA256: dd93a34b9232832e359bde6c361fe68475d4efcebe2e54a8a34cae1d7117b614. See audit/extraction_provenance.json.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
P = 5
THETA_0 = 1.0
NORMAL_BASELINE = "N(0,1)"
BOOSTING_PARAMS = {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0}

# Source cell 4, lines 1-4
def ar1_covariance(p: int, rho: float = 0.5) -> np.ndarray:
    """构造 Sigma[j,k] = rho**|j-k| 的 AR(1) 型协方差矩阵。"""
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


# Source cell 4, lines 7-30
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


# Source cell 4, lines 33-37
def standardized_student_t(rng: np.random.Generator, df: float, size: int) -> np.ndarray:
    """生成均值 0、方差 1 的 Student-t 误差；要求 df > 2。"""
    if df <= 2:
        raise ValueError("Student-t 自由度必须大于 2，才能标准化为单位方差。")
    return rng.standard_t(df, size=size) * np.sqrt((df - 2.0) / df)


# Source cell 4, lines 40-54
def generate_data(n: int, error_case, dgp_name: str, seed: int, theta0: float = THETA_0):
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


# Source cell 6, lines 1-9
def make_nuisance_model(kind: str, seed: int):
    """根据估计量类型创建条件均值或条件中位数学习器。"""
    common = dict(BOOSTING_PARAMS)
    common["random_state"] = seed
    if kind == "mean":
        return HistGradientBoostingRegressor(loss="squared_error", **common)
    if kind == "median":
        return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **common)
    raise ValueError("kind 必须是 'mean' 或 'median'。")


# Source cell 6, lines 12-26
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


# Source cell 6, lines 39-48
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

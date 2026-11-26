# Frozen original definitions; no notebook launch cells or Dukes correction.
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
P=5
THETA_0=1.0
NORMAL_BASELINE="N(0,1)"
BOOSTING_PARAMS=dict(learning_rate=0.05,max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0)

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

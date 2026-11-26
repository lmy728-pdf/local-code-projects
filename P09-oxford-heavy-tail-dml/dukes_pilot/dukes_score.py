"""Python NW port of Dukes--Vansteelandt--Whitney (2024), Sec. 3.3.

This is NOT a software-equivalent reproduction of R npregbw. See method_spec.md.
No DGP truth is accessed here. Initial fits are cross-fitted; auxiliary fits are
within each held-out fold, including self for final fitted values.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor


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

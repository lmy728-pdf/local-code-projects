import numpy as np
from scipy.stats import norm

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

"""Scenario C design and estimators, using the frozen Scenario A methods."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import hashlib
import json
import time
from functools import lru_cache
import numpy as np
from scipy.stats import t, norm
from scipy.stats.qmc import Sobol
import legacy_core as legacy
import methods as a_methods

SIZES = (500, 1000, 2000, 4000, 8000)
METHODS = ("OLS", "Rank-W", "Huber", "LAD")
MASTER = 20261119
NAMESPACE = "rank-dml-scenario-c-v1"
MAD_SCALE = 1.482602218505602
W_TOL = 1e-12
Z95 = norm.ppf(.975)

def random_stream(stage, case, n, rep, stream):
    key = json.dumps([NAMESPACE, MASTER, stage, case, n, rep, stream],
                     separators=(",", ":"))
    entropy = np.frombuffer(hashlib.sha256(key.encode()).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(entropy)), key

@lru_cache(maxsize=5)
def tail_cutoff(n):
    return t.isf(n**(-.75), 3) / np.sqrt(3)

def generate(stage, case, n, rep):
    rng, key = random_stream(stage, case, n, rep, "data")
    probability = n**(-.75)
    if case == "C1":
        x = rng.uniform(size=(n, 1))
        rare = x[:, 0] <= probability
        truth = np.zeros((n, 2))
    else:
        x = np.column_stack((rng.standard_t(3, size=n)/np.sqrt(3),
                             rng.standard_normal((n, 9))))
        rare = x[:, 0] > tail_cutoff(n)
        truth = rare[:, None] * np.array([-1., -np.sqrt(n)])
    innovations = rng.standard_normal((n, 2))
    fold_rng, fold_key = random_stream(stage, case, n, rep, "outer-folds")
    fold = legacy.folds(n, 5, fold_rng)
    return x, truth + innovations, truth, innovations, rare, fold, key, fold_key

def sorted_wilcoxon(w, z, fold):
    """Find the leftmost minimizer via the right derivative of the exact Q_n.

    Sorted residuals give sum_{i<j}|r_i-r_j| = sum_j(2j-m+1)r_(j).
    At tied residuals, sorting on -w gives the right derivative. The final
    bracket has length <= W_TOL * max(1, abs(endpoints)).
    """
    n = len(w)
    parts = []
    for k in np.unique(fold):
        active = fold == k
        wk, zk = w[active], z[active]
        if len(wk) < 2:
            raise ValueError("Wilcoxon fold needs at least two observations")
        weights = (2*np.arange(len(wk)) - len(wk) + 1)/(n*(len(wk)-1))
        parts.append((wk, zk, weights))
    if not any(np.ptp(wk) > 0 for wk, _, _ in parts):
        raise ValueError("Wilcoxon objective has no informative pairs")

    def derivative(theta):
        slope = 0.
        for wk, zk, weights in parts:
            residual = zk-theta*wk
            order = np.argsort(residual)
            ordered = residual[order]
            if np.any(ordered[1:] == ordered[:-1]):
                order = np.lexsort((-wk, residual))
            slope -= np.dot(weights, wk[order])
        return slope

    lo, hi = -1., 1.
    for _ in range(60):
        if derivative(lo) < 0 and derivative(hi) >= 0:
            break
        lo *= 2
        hi *= 2
    else:
        raise ValueError("Could not bracket the Wilcoxon minimum")
    while hi-lo > W_TOL*max(1., abs(lo), abs(hi)):
        middle = (lo+hi)/2
        if derivative(middle) < 0:
            lo = middle
        else:
            hi = middle
    return (lo+hi)/2

def diagnostics(errors, n, weights=None):
    b, a = errors.T
    weights = np.full(len(b), 1/len(b)) if weights is None else weights
    norm_b = np.sqrt(weights @ (b*b))
    norm_a = np.sqrt(weights @ (a*a))
    norm_clip = np.sqrt(weights @ np.minimum(1., np.abs(a))**2)
    covariance = weights @ (b*a) - (weights @ b)*(weights @ a)
    return dict(norm_b=float(norm_b), norm_a=float(norm_a),
                norm_clipped_a=float(norm_clip), root_n_product=float(np.sqrt(n)*norm_b*norm_a),
                root_n_truncated=float(np.sqrt(n)*norm_b*norm_clip),
                cov_b_a=float(covariance))

@lru_cache(maxsize=15)
def diagnostic_grid(stage, n):
    """Independent, probability-weighted conditional evaluation grid for C2."""
    rng, _ = random_stream(stage, "C2", n, -1, "diagnostic-grid")
    p = n**(-.75)
    # Scrambled Sobol, 1024 X values per region. Common grid across fitted folds.
    uniform = Sobol(d=10, scramble=True, rng=rng).random_base2(10)
    common = norm.ppf(uniform[:, 1:])
    tail_x = np.column_stack((t.isf(p*uniform[:, 0], 3)/np.sqrt(3), common))
    other_x = np.column_stack((t.ppf((1-p)*uniform[:, 0], 3)/np.sqrt(3), common))
    design = np.concatenate((other_x, tail_x))
    truth = np.zeros((len(design), 2))
    truth[1024:] = [-1., -np.sqrt(n)]
    weights = np.repeat([(1-p)/1024, p/1024], 1024)
    return design, truth, weights

def condition_metrics(errors):
    if not len(errors):
        return None, None, None, None
    means = errors.mean(0).tolist()
    rmses = np.sqrt(np.mean(errors**2, axis=0)).tolist()
    return *means, *rmses

def run_replication(task):
    stage, case, n, rep = task
    start = time.perf_counter()
    record = dict(stage=stage, case=case, n=n, rep=rep)
    cache = None
    try:
        started = time.perf_counter()
        x, outcomes, truth, innovations, rare, fold, seed, fold_seed = generate(*task)
        prob = n**(-.75)
        record.update(seed_key=seed, fold_seed_key=fold_seed, rare_count=int(rare.sum()),
                      rare_probability=prob, expected_rare=n*prob,
                      cutoff=prob if case == "C1" else tail_cutoff(n))
        if case == "C1":
            errors = rare[:, None]*np.array([1., np.sqrt(n)])
            residuals, models = outcomes-errors, []
            record["population"] = dict(norm_b=n**(-.375), norm_a=n**.125,
                norm_clipped_a=n**(-.375), root_n_product=n**.25,
                root_n_truncated=n**(-.25), cov_b_a=np.sqrt(n)*prob*(1-prob))
        else:
            residuals, models = legacy.crossfit(x, outcomes, fold)
            errors = innovations-residuals
            record["penalties"] = [model.alpha.tolist() for model in models]
        record["nuisance_seconds"] = time.perf_counter()-started
        record["sample"] = diagnostics(errors, n)
        record["fold_diagnostics"] = []
        if models:
            grid, grid_truth, weights = diagnostic_grid(stage, n)
        for k in range(5):
            active = fold == k
            entry = dict(fold=k, sample_size=int(active.sum()), rare_count=int((active & rare).sum()),
                         sample=diagnostics(errors[active], n),
                         sample_inside=condition_metrics(errors[active & rare]),
                         sample_outside=condition_metrics(errors[active & ~rare]))
            if models:
                grid_error = models[k].predict(grid)-grid_truth
                entry.update(grid=diagnostics(grid_error, n, weights),
                             grid_inside=condition_metrics(grid_error[1024:]),
                             grid_outside=condition_metrics(grid_error[:1024]),
                             tail_extrapolation_fraction=float(np.mean(grid[1024:, 0]>models[k].bases[0][1])))
            record["fold_diagnostics"].append(entry)
        record["input_sha256"] = hashlib.sha256(residuals.tobytes()+fold.tobytes()).hexdigest()
        cache = (fold.astype(np.uint8), rare, errors)
    except Exception as exc:
        record.update(error=f"{type(exc).__name__}: {exc}",
                      seconds=time.perf_counter()-start)
        record["methods"] = [dict(method=name, estimate=None, se=None, valid_point=False,
                                  valid_interval=False, covered=False, failure=record["error"], seconds=0.)
                             for name in METHODS]
        return record, cache

    w, z = residuals.T
    record["methods"] = []
    for name in METHODS:
        tick = time.perf_counter()
        point, se, failure, meta = None, None, None, {}
        try:
            if name == "Rank-W":
                point = sorted_wilcoxon(w, z, fold)
                def rank_stat(theta):
                    return a_methods.deltas(w, z, fold, theta, "W").sum()
                gamma = a_methods.fivepoint(rank_stat, point, n)
                meta["slope"] = float(gamma)
                a_methods.positive(gamma, "nonpositive_rank_slope")
                se = np.sqrt(np.mean(w*w)/(12*n*gamma**2))
            elif name == "LAD":
                point, se, meta = a_methods.quantile_fit(w, z, fold, [.5])
            else:
                thresholds = None
                if name == "Huber":
                    scales = []
                    for k in range(5):
                        training = fold != k
                        inner_rng, _ = random_stream(stage, case, n, rep, f"huber-inner-{k}")
                        inner = legacy.folds(int(training.sum()), 3, inner_rng)
                        if models:
                            inner_residuals, _ = legacy.crossfit(x[training], outcomes[training], inner)
                        else:
                            inner_residuals = residuals[training]
                        wi, zi = inner_residuals.T
                        preliminary = sorted_wilcoxon(wi, zi, inner)
                        error = zi-preliminary*wi
                        scales.append(float(MAD_SCALE*np.median(abs(error-np.median(error)))))
                    thresholds = 1.345*np.array(scales)[fold]
                    meta["training_scales"] = scales
                point, influence, fit_meta = legacy.regression(w, z, fold, thresholds)
                se = np.sqrt(np.mean(influence**2)/n)
                meta.update(fit_meta)
            if not np.isfinite(se) or se <= 0 or not np.isfinite(point):
                raise ValueError("invalid estimate or standard error")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        valid_point = point is not None and np.isfinite(point)
        valid_interval = valid_point and failure is None
        r = dict(method=name, estimate=float(point) if valid_point else None,
                 root_n_estimate=float(np.sqrt(n)*point) if valid_point else None,
                 se=float(se) if valid_interval else None, valid_point=bool(valid_point),
                 valid_interval=bool(valid_interval), covered=False, failure=failure,
                 seconds=time.perf_counter()-tick, metadata=meta)
        if valid_interval:
            r.update(lo=float(point-Z95*se), hi=float(point+Z95*se),
                     covered=bool(abs(point)<=Z95*se), length=float(2*Z95*se))
        if name == "OLS" and valid_point:
            r["ols_centered"] = float(np.sqrt(n)*point-n**.25)
            r["ols_centered_scaled"] = r["ols_centered"]/n**.125
        record["methods"].append(r)
    record["seconds"] = time.perf_counter()-start
    return record, cache



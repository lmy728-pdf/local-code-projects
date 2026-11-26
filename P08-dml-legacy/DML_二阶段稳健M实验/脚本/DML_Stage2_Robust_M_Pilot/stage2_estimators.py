"""Second-stage fits only. No true parameter, scenario label, or DGP input."""
import numpy as np
from scipy.optimize import minimize
from stage2_losses import psi, objective_gradient, bread
from legacy_core import fit_mean_dml

SCORE_TOL = 1e-9
COND_LIMIT = 1e10


def validate_data(d, y):
    d, y = np.asarray(d, dtype=float), np.asarray(y, dtype=float)
    if d.ndim != 1 or y.shape != d.shape or len(d) < 4:
        raise ValueError('Expected equal 1D arrays with n >= 4')
    if not np.isfinite(np.r_[d, y]).all():
        raise ValueError('Nonfinite data')
    if np.mean(d*d) <= 1e-12:
        raise ValueError('Degenerate treatment residuals')
    return d, y


def preliminary_scale(d, y):
    d, y = validate_data(d, y)
    theta = np.mean(d*y)/np.mean(d*d)
    r = y-theta*d
    s = 1.4826*np.median(abs(r-np.median(r)))
    if not np.isfinite(s) or s <= 1e-12:
        raise ValueError('MAD scale is zero or nonfinite; no silent scale substitution')
    return float(s)


def sandwich(z, r, tm, tp):
    n = len(r)
    w = ((r > -tm) & (r < tp)).astype(float)
    A = z.T @ (w[:, None]*z)/n
    eigen = np.linalg.eigvalsh(A)
    condition = float(np.linalg.cond(A))
    if eigen.min() <= 1e-12 or not np.isfinite(condition) or condition > COND_LIMIT:
        raise ArithmeticError(f'Singular/ill-conditioned bread: eigmin={eigen.min()}, cond={condition}')
    u = z*psi(r, tm, tp)[:, None]
    centered = u-u.mean(axis=0)
    B = centered.T@centered/n  # User-specified population divisor n, NOT n-1.
    invA = np.linalg.solve(A, np.eye(A.shape[0]))
    V = invA@B@invA.T/n
    if not np.isfinite(V).all() or V[-1, -1] <= 0:
        raise ArithmeticError('Invalid sandwich variance')
    return V, A, B, u, condition, float(eigen.min())


def fit_huber(d, y, tau_minus, tau_plus, intercept):
    d, y = validate_data(d, y)
    tm, tp = float(tau_minus), float(tau_plus)
    if not np.isfinite([tm, tp]).all() or min(tm, tp) <= 0:
        raise ValueError('Thresholds must be positive finite constants')
    z = np.column_stack([np.ones(len(d)), d]) if intercept else d[:, None]
    if np.linalg.matrix_rank(z) != z.shape[1]:
        raise ArithmeticError('Rank deficient design')
    initial = np.linalg.lstsq(z, y, rcond=None)[0]
    fun = lambda b: objective_gradient(b, z, y, tm, tp)
    opt = minimize(fun, initial, jac=True, method='BFGS', options={'gtol': 1e-10, 'maxiter': 1000})
    beta = opt.x.copy()
    # Semismooth Newton polishing with loss-decreasing line search. Same convex
    # objective and fixed thresholds; acceptance uses the estimating equations.
    polish = 0
    for polish in range(30):
        value, grad = fun(beta)
        if np.max(abs(grad)) <= 1e-11:
            break
        A = bread(beta, z, y, tm, tp)
        if np.linalg.cond(A) > COND_LIMIT:
            break
        direction = np.linalg.solve(A, grad)
        accepted = False
        for k in range(35):
            candidate = beta-direction*(.5**k)
            nv, ng = fun(candidate)
            if nv <= value+1e-14*max(1., abs(value)) and np.linalg.norm(ng) < np.linalg.norm(grad):
                beta = candidate
                accepted = True
                break
        if not accepted:
            break
    value, grad = fun(beta)
    score_norm = float(np.max(abs(grad)))
    if not np.isfinite(beta).all() or score_norm > SCORE_TOL:
        raise ArithmeticError(f'Estimating equations not solved: norm={score_norm}; optimizer={opt.message}')
    r = y-z@beta
    V, A, B, u, condition, mineig = sandwich(z, r, tm, tp)
    near_knots = int(np.sum(np.isclose(r, -tm, atol=1e-10, rtol=0) | np.isclose(r, tp, atol=1e-10, rtol=0)))
    result = dict(theta_hat=float(beta[-1]), intercept=float(beta[0]) if intercept else 0.,
                  has_intercept=bool(intercept), se=float(np.sqrt(V[-1, -1])),
                  intercept_se=float(np.sqrt(V[0, 0])) if intercept else np.nan,
                  tau_minus=tm, tau_plus=tp, clip_left=float(np.mean(r < -tm)),
                  clip_right=float(np.mean(r > tp)), clipping_rate=float(np.mean((r < -tm) | (r > tp))),
                  score_intercept=float(u[:, 0].mean()) if intercept else np.nan,
                  score_theta=float(u[:, -1].mean()), score_norm=score_norm,
                  bread_condition=condition, bread_min_eigenvalue=mineig, near_knots=near_knots,
                  converged=True, optimizer_success=bool(opt.success), optimizer_message=str(opt.message),
                  optimizer_iterations=int(opt.nit), polish_iterations=polish, objective=value,
                  variance_convention='centered sandwich, divisor n, threshold fixed',
                  A00=float(A[0, 0]), A01=float(A[0, -1]) if intercept else np.nan,
                  A11=float(A[-1, -1]), B00=float(B[0, 0]),
                  B01=float(B[0, -1]) if intercept else np.nan, B11=float(B[-1, -1]),
                  V00=float(V[0, 0]), V01=float(V[0, -1]) if intercept else np.nan, V11=float(V[-1, -1]))
    arrays = dict(residual=r, psi=psi(r, tm, tp), estimating_score=u, bread=A, meat=B, covariance=V)
    return result, arrays


def fit_method(d, y, method, b_kappa=1.345, c_minus=2., c_plus=1.5, d_kappa=1.):
    """Sensitivity parameters explicit; no automatic tuning or distribution input."""
    d, y = validate_data(d, y)
    if method == 'A':
        theta, se = fit_mean_dml(d, y)  # Exact frozen original formula, ddof=1.
        if not np.isfinite([theta, se]).all() or se <= 0:
            raise ArithmeticError('Invalid original DML estimate')
        r = y-theta*d
        return dict(theta_hat=theta, se=se, intercept=0., has_intercept=False,
                    tau_minus=np.nan, tau_plus=np.nan, scale=np.nan,
                    clip_left=0., clip_right=0., clipping_rate=0., converged=True,
                    optimizer_success=True, optimizer_message='Exact original closed form',
                    score_norm=float(abs(np.mean(d*r))), score_theta=float(np.mean(d*r)),
                    variance_convention='original IF-SE, ddof=1'), dict(residual=r, estimating_score=d*r)
    s = preliminary_scale(d, y)
    if method == 'B':
        tm = tp = b_kappa*s
        intercept = False
    elif method == 'C':
        tm, tp, intercept = c_minus*s, c_plus*s, True
    elif method == 'D':
        tm = tp = d_kappa*s*np.sqrt(len(d)/np.log(len(d)))
        intercept = True
    else:
        raise ValueError('Unknown method')
    result, arrays = fit_huber(d, y, tm, tp, intercept)
    result['scale'] = s
    return result, arrays

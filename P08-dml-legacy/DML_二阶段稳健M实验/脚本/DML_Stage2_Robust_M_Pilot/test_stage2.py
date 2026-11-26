"""Deterministic numerical tests and Oracle diagnostics using existing datasets."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from stage2_losses import rho, psi, objective_gradient, bread
from stage2_estimators import fit_huber, fit_method, preliminary_scale, sandwich
from data_access import ROOT, preflight, load_replication, dump


def population_lognormal_score(a, tm, tp, sigma=2.):
    # Exact truncated lognormal first moments, ONLY for Oracle validation.
    mean = np.exp(.5*sigma*sigma)
    sd = np.sqrt(np.expm1(sigma*sigma)*np.exp(sigma*sigma))
    lo, hi = a-tm, a+tp
    def cdf(t):
        w = t*sd+mean
        return 0. if w <= 0 else float(norm.cdf(np.log(w)/sigma))
    def truncated_first(t):
        w = t*sd+mean
        if w <= 0: return 0.
        z = np.log(w)/sigma
        return float((mean*norm.cdf(z-sigma)-mean*norm.cdf(z))/sd)
    fl, fh = cdf(lo), cdf(hi)
    return -tm*fl + truncated_first(hi)-truncated_first(lo)-a*(fh-fl) + tp*(1-fh)


def numerical_tests():
    # Fixed non-random fixture; known OLS slope=2, intercept=1.
    d = np.array([-3., -2., -1., 0., 1., 2., 3.])
    e = np.array([1., -1., 0., 0., 0., -1., 1.])
    y = 1.+2.*d+e
    checks = {}
    for intercept in [False, True]:
        fit, arrays = fit_huber(d, y, 1e8, 1e8, intercept)
        np.testing.assert_allclose(fit['theta_hat'], 2., atol=1e-12)
        np.testing.assert_allclose(fit['intercept'], 1. if intercept else 0., atol=1e-12)
        checks['large_tau_'+str(intercept)] = fit['theta_hat']
    a, _ = fit_method(d, y, 'A')
    b, _ = fit_huber(d, y, 1e8, 1e8, False)
    # n vs n-1 intentional: Huber HC0 SE = old IF-SE * sqrt((n-1)/n).
    np.testing.assert_allclose(b['se'], a['se']*np.sqrt((len(d)-1)/len(d)), atol=1e-12)
    checks['large_tau_se_ratio'] = b['se']/a['se']
    max_gradient_error = 0.
    max_hessian_error = 0.
    for intercept in [False, True]:
        z = np.column_stack([np.ones(len(d)), d]) if intercept else d[:, None]
        beta = np.array([.17, 1.83]) if intercept else np.array([1.83])
        tm, tp, step = 1.7, .8, 1e-6
        _, gradient = objective_gradient(beta, z, y, tm, tp)
        numeric = np.zeros_like(beta)
        numeric_hessian = np.zeros((len(beta), len(beta)))
        for j in range(len(beta)):
            h = np.eye(len(beta))[j]*step
            vp, gp = objective_gradient(beta+h, z, y, tm, tp)
            vm, gm = objective_gradient(beta-h, z, y, tm, tp)
            numeric[j] = (vp-vm)/(2*step)
            numeric_hessian[:, j] = (gp-gm)/(2*step)
        max_gradient_error = max(max_gradient_error, float(np.max(abs(numeric-gradient))))
        max_hessian_error = max(max_hessian_error, float(np.max(abs(numeric_hessian-bread(beta, z, y, tm, tp)))))
        np.testing.assert_allclose(numeric, gradient, atol=1e-8)
        np.testing.assert_allclose(numeric_hessian, bread(beta, z, y, tm, tp), atol=1e-8)
        fit, arr = fit_huber(d, y, tm, tp, intercept)
        assert fit['score_norm'] < 1e-9
        # Independent scalar/2x2 IF construction verifies all covariance entries.
        residual = y-z@np.r_[fit['intercept'], fit['theta_hat']] if intercept else y-d*fit['theta_hat']
        w = ((residual > -tm)&(residual < tp)).astype(float)
        A = sum(w[i]*np.outer(z[i], z[i]) for i in range(len(d)))/len(d)
        u = np.array([z[i]*min(tp, max(-tm, residual[i])) for i in range(len(d))])
        IF = np.array([np.linalg.solve(A, ui-u.mean(axis=0)) for ui in u])
        independent = sum(np.outer(v, v) for v in IF)/len(d)**2
        np.testing.assert_allclose(arr['covariance'], independent, atol=1e-12)
    checks.update(max_gradient_error=max_gradient_error, max_hessian_error=max_hessian_error,
                  sandwich_independent_check=True)
    for tm, tp in [(1.345, 1.345), (2., 1.5)]:
        for knot in [-tm, tp]:
            left, right = rho(np.array([knot-1e-9, knot+1e-9]), tm, tp)
            assert abs(right-left) < 1e-7
    checks['loss_continuity'] = True
    # Exact robust solution fixture, balanced d and errors; clipping active.
    dd = np.repeat(np.array([-2., -1., 1., 2.]), 5)
    ee = np.tile(np.array([-10., -.5, 0., .5, 10.]), 4)
    fit, _ = fit_huber(dd, .7+1.3*dd+ee, 1., 1., True)
    np.testing.assert_allclose([fit['intercept'], fit['theta_hat']], [.7, 1.3], atol=1e-10)
    np.testing.assert_allclose(fit['clipping_rate'], .4, atol=1e-12)
    checks['fixed_fixture'] = dict(intercept=fit['intercept'], theta=fit['theta_hat'], clip=fit['clipping_rate'])
    for dd, yy in [(np.ones(10), np.arange(10.)), (np.arange(10.), np.full(10, np.nan))]:
        try: fit_huber(dd, yy, 1., 1., True)
        except (ValueError, ArithmeticError): pass
        else: raise AssertionError('Invalid input not rejected')
    checks['invalid_inputs_rejected'] = True
    # Sensitivity interfaces available but not used to select pilot settings.
    for k in [2., 3.]: fit_method(d, y, 'B', b_kappa=k)
    for k in [.5, 1.5]: fit_method(d, y, 'D', d_kappa=k)
    checks['sensitivity_interface'] = True
    return checks


def run_tests(output):
    output.mkdir(parents=True, exist_ok=True)
    cfg, info = preflight()
    checks = numerical_tests()
    rows = []
    # First 20 existing replications, same n and seeds. Diagnostics, not pilot inputs.
    for scene in cfg['scenarios']:
        for rep in range(20):
            _, _, audit, oracle = load_replication(cfg, scene, rep)
            d, y = oracle['d'], oracle['y']
            for method in ['B', 'C', 'D']:
                fit, _ = fit_method(d, y, method)
                row = dict(scenario=scene['name'], rep_id=rep, seed=audit['seed'], method=method, **fit)
                row['T'] = (fit['theta_hat']-cfg['theta0'])/fit['se']
                if scene['name']=='lognormal2' and method in ['C', 'D']:
                    tm, tp = fit['tau_minus'], fit['tau_plus']
                    pop_a = brentq(lambda a: population_lognormal_score(a, tm, tp), -10., 10., xtol=1e-13)
                    row.update(oracle_population_intercept=pop_a,
                               population_score_at_zero=population_lognormal_score(0., tm, tp),
                               population_score_at_location=population_lognormal_score(pop_a, tm, tp),
                               intercept_minus_population=fit['intercept']-pop_a)
                    assert abs(row['population_score_at_location']) < 1e-10
                rows.append(row)
    frame = pd.DataFrame(rows)
    summaries = []
    for (scene, method), group in frame.groupby(['scenario', 'method'], sort=False):
        mcse = group.theta_hat.std(ddof=1)/np.sqrt(len(group))
        delta = group.theta_hat.mean()-cfg['theta0']
        summaries.append(dict(scenario=scene, method=method, n=len(group), mean_theta=group.theta_hat.mean(),
                              theta_bias=delta, bias_mcse=mcse, within_4_mcse=abs(delta)<=4*mcse,
                              mean_intercept=group.intercept.mean(), mean_se=group.se.mean(),
                              empirical_sd=group.theta_hat.std(ddof=1), mean_T=group['T'].mean(),
                              max_equation_error=group.score_norm.max(), mean_clipping=group.clipping_rate.mean(),
                              mean_population_intercept=group.get('oracle_population_intercept', pd.Series(dtype=float)).mean(),
                              mean_population_score_at_zero=group.get('population_score_at_zero', pd.Series(dtype=float)).mean()))
    # Broad, pre-specified identification sanity gate; not an asymptotic proof.
    required = [r for r in summaries if (r['scenario']=='t2.1' and r['method']=='B') or
                (r['scenario']=='lognormal2' and r['method'] in ['C', 'D'])]
    frame.to_csv(output/'oracle_diagnostics.csv', index=False, encoding='utf-8-sig')
    dump(output/'oracle_summary.json', summaries)
    checks['oracle_identification_gate'] = all(r['within_4_mcse'] for r in required)
    dump(output/'numerical_validation.json', dict(checks=checks, source=info, oracle=summaries))
    assert checks['oracle_identification_gate'], 'Oracle identification sanity gate failed; inspect before pilot'
    return checks, summaries


if __name__=='__main__':
    result = run_tests(ROOT/'validation')
    print(json.dumps(result[0], indent=2))

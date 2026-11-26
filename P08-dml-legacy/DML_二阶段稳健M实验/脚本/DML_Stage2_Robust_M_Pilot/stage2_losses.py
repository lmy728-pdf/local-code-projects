"""Convex asymmetric Huber loss; thresholds are fixed during each fit."""
import numpy as np


def psi(r, tau_minus, tau_plus):
    return np.clip(r, -tau_minus, tau_plus)


def rho(r, tau_minus, tau_plus):
    r = np.asarray(r, dtype=float)
    return np.where(r < -tau_minus, -tau_minus*r - .5*tau_minus**2,
                    np.where(r > tau_plus, tau_plus*r - .5*tau_plus**2, .5*r*r))


def psi_prime(r, tau_minus, tau_plus):
    # At the two nondifferentiable knots choose 0; knot counts are reported.
    return ((r > -tau_minus) & (r < tau_plus)).astype(float)


def objective_gradient(beta, z, y, tau_minus, tau_plus):
    r = y - z @ beta
    return float(np.mean(rho(r, tau_minus, tau_plus))), -z.T @ psi(r, tau_minus, tau_plus) / len(y)


def bread(beta, z, y, tau_minus, tau_plus):
    w = psi_prime(y-z@beta, tau_minus, tau_plus)
    return z.T @ (w[:, None]*z) / len(y)

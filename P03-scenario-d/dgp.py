"""The three Scenario D data-generating processes and deterministic seed keys."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from legacy_core import covariates, folds, truth

MASTER = 20261120
SCENARIOS = ("D1", "D2", "D3")
MODES = ("smoke", "pilot", "main")
SIZES = (500, 1000, 2000)
COUNTS = dict(smoke=4, pilot=100, main=2000)
STREAMS = ("data", "folds", "inner0", "inner1", "inner2", "inner3", "inner4",
           "rank_ad_max_t", "rank_ad_sandwich_max_t", "odd_max_t")
METHODS = ("OLS", "Huber", "LAD", "CQR", "Bickel", "Rank-W", "Rank-N", "Rank-Ad")
D3_METHODS = ("Odd-only", "Fixed mixed")

def seed_key(mode, scenario, n, rep, stream):
    return [MASTER, SCENARIOS.index(scenario)+1, n, rep,
            MODES.index(mode)+1, STREAMS.index(stream)+1]

def rng(mode, scenario, n, rep, stream):
    return np.random.default_rng(np.random.SeedSequence(seed_key(mode,scenario,n,rep,stream)))

def target(scenario):
    return float(scenario=="D1")

def generate(mode, scenario, n, rep):
    data_rng = rng(mode, scenario, n, rep, "data")
    if scenario == "D1":
        x = covariates(n, data_rng)
        m, ell, _ = truth(x, 1, "scalar")
        v, innovation = data_rng.standard_normal((2,n))
        scale = np.where(x[:,0]<0, .5, 2.) / np.sqrt(2.125)
        error = scale*innovation
        d = m+v
        y = ell[:,0]+v+error
    else:
        x = None
        v = 2*data_rng.integers(0,2,n)-1.
        if scenario == "D2":
            innovation = data_rng.exponential(size=n)-1
            scale = np.ones(n)
            error = v*innovation
        else:
            innovation = data_rng.standard_normal(n)
            scale = np.where(v==1, 2., .5)
            error = scale*innovation
        d, y = v, error
    f = folds(n,5,rng(mode,scenario,n,rep,"folds"))
    return dict(x=x, d=d, y=y, v=v, error=error, innovation=innovation, scale=scale, folds=f)


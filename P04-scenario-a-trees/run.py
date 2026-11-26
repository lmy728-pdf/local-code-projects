"""Scenario A tree nuisances, checkpointing and the original nine estimators."""
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing, contextmanager
import csv
from functools import lru_cache
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import math
import multiprocessing
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "spline_core"))
import legacy_core as legacy
import methods
from filelock import FileLock, Timeout
import numpy as np
import psutil
from scipy.optimize import OptimizeWarning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits
from tqdm import tqdm

MANIFEST = json.loads((ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))
THEORY = json.loads((ROOT / "spline_core" / "theory.json").read_text(encoding="utf-8"))["laws"]
LEARNERS = ("rf", "gbdt")
PACKAGE_VERSIONS = {name: version(name) for name in ("numpy", "scipy", "scikit-learn")}
CACHE_SCHEMA = 1


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_text(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"), default=lambda v: v.item())


def comparable_json(value):
    if isinstance(value, np.ndarray):
        return comparable_json(value.tolist())
    if isinstance(value, (tuple, list)):
        return [comparable_json(v) for v in value]
    if isinstance(value, dict):
        return {k: comparable_json(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return comparable_json(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


@lru_cache(maxsize=1)
def seed_registry():
    path = ROOT / "baseline" / "main_seed_registry.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return {(int(r["n"]), r["law"], int(r["rep"])): json.loads(r["seed_key"])
                for r in csv.DictReader(stream)}


def generate(mode, n, law, rep):
    seed = (seed_registry()[n, law, rep] if mode == "main"
            else [methods.VERSION, 20261118, mode, n, law, rep, "data"])
    if seed[0] != 2026111401:
        if seed != [methods.VERSION, 20261118, mode, n, law, rep, "data"]:
            raise ValueError("Inconsistent Scenario A seed registry")
        return methods.generate(mode, n, law, rep), seed

    _, stage, cell, legacy_rep, stream = seed
    if (cell, legacy_rep, stream) != (dict(experiment="scalar", law=law, n=n, d=1, rho=0), rep, "data"):
        raise ValueError("Inconsistent historical seed registry")
    random = legacy.rng(stage, cell, rep, "data")
    if law == "sln100_tau050":
        # Original G6A tau=.50 generator, as replayed by the Spline Ridge audit.
        x = legacy.covariates(n, random)
        m, ell, _ = legacy.truth(x, 1, "scalar")
        v = random.normal(size=n)
        g, h = random.normal(size=(n, 1)), random.normal(size=(n, 1))
        e = (np.exp(g) + .5 * h - np.exp(.5)) / np.sqrt(np.exp(1) * np.expm1(1) + .25)
        oracle_z = v[:, None] + e
        d, y = m + v, ell + oracle_z
    else:
        x, d, y, v, oracle_z, _ = legacy.generate(cell, random)
    folds = legacy.folds(n, 5, legacy.rng(stage, cell, rep, "outer"))
    return (x, d, y.ravel(), v, oracle_z.ravel(), folds), seed


def sample_hash(x, d, y, folds):
    return digest(b"".join(np.asarray(a, dtype=dtype).tobytes(order="C")
                           for a, dtype in ((x, "<f8"), (d, "<f8"), (y, "<f8"), (folds, "<i8"))))


def model_seed(mode, n, law, rep, learner, fold, split, target):
    rng = methods.rng(mode, n, law, rep, f"trees/{learner}/{fold}/{split}/{target}")
    return int(rng.integers(0, 2**32, dtype=np.uint32))


def make_model(learner, seed):
    params = MANIFEST["learners"][learner] | {"random_state": seed}
    model_type = RandomForestRegressor if learner == "rf" else GradientBoostingRegressor
    return model_type(**params)


def fit_pair(x, response, train, learner, seeds):
    models = [make_model(learner, seed).fit(x[train], response[train, j])
              for j, seed in enumerate(seeds)]
    iterations = [int(model.n_estimators_) if learner == "gbdt" else len(model.estimators_)
                  for model in models]
    return models, iterations


def predict_pair(models, x):
    return np.column_stack([model.predict(x) for model in models])


def fit_nuisance(task):
    learner, mode, n, law, rep = task
    started = perf_counter()
    meta = dict(mode=mode, learner=learner, n=n, law=law, rep=rep, status="ok", diagnostics=[])
    payload = None
    try:
        (x, d, y, v, oracle_z, folds), seed = generate(mode, n, law, rep)
        meta["seed_key"] = seed
        meta["sample_sha256"] = sample_hash(x, d, y, folds)
        meta["fold_sha256"] = digest(folds.astype("<i8").tobytes())
        response = np.column_stack([d, y])
        m, ell, _ = legacy.truth(x, 1, "scalar")
        truth = np.column_stack([m, ell[:, 0]])
        fresh = legacy.covariates(MANIFEST["reference_n"], methods.rng(mode, n, law, rep, "diagnostic"))
        m_ref, ell_ref, _ = legacy.truth(fresh, 1, "scalar")
        truth_ref = np.column_stack([m_ref, ell_ref[:, 0]])
        predictions = np.full((n, 2), np.nan)
        scales = np.full(5, np.nan)
        for fold in range(5):
            valid = folds == fold
            train = ~valid
            fold_meta = dict(fold=fold, status="ok", n_train=int(train.sum()), n_valid=int(valid.sum()))
            meta["diagnostics"].append(fold_meta)
            try:
                seeds = [model_seed(mode, n, law, rep, learner, fold, "outer", target)
                         for target in ("d", "y")]
                models, rounds = fit_pair(x, response, train, learner, seeds)
                predictions[valid] = predict_pair(models, x[valid])
                reference_error = predict_pair(models, fresh) - truth_ref
                b, q = reference_error.T
                alignment = q - b
                c = alignment.mean()
                rmse_m, rmse_ell = np.sqrt(np.mean(reference_error**2, axis=0))
                rmse_a = np.sqrt(np.mean((alignment - c)**2))
                training_error = predict_pair(models, x[train]) - response[train]
                validation_error = predictions[valid] - response[valid]
                validation_truth_error = predictions[valid] - truth[valid]
                fold_meta.update(outer_seeds=seeds, outer_iterations=rounds,
                                 reference_rmse_m=rmse_m, reference_rmse_ell=rmse_ell,
                                 alignment_rmse=rmse_a, alignment_location=c,
                                 alignment_product=rmse_m*rmse_a, root_n_alignment_product=np.sqrt(n)*rmse_m*rmse_a,
                                 cov_b_a=np.mean((b-b.mean())*(alignment-c)),
                                 training_observed_rmse=np.sqrt(np.mean(training_error**2, axis=0)).tolist(),
                                 validation_observed_rmse=np.sqrt(np.mean(validation_error**2, axis=0)).tolist(),
                                 validation_truth_rmse=np.sqrt(np.mean(validation_truth_error**2, axis=0)).tolist())

                # Preserve the Spline experiment's Huber calibration protocol using
                # the current learner on three inner folds of the outer training set.
                fold_meta["component"] = "huber_inner"
                inner_x, inner_y = x[train], response[train]
                inner = legacy.folds(len(inner_x), 3, methods.rng(mode, n, law, rep, f"inner-{fold}"))
                inner_residual = np.empty_like(inner_y)
                inner_meta = []
                fold_meta["huber_inner"] = inner_meta
                for k in range(3):
                    inner_seeds = [model_seed(mode, n, law, rep, learner, fold, f"inner-{k}", target)
                                   for target in ("d", "y")]
                    inner_fit, rounds = fit_pair(inner_x, inner_y, inner != k, learner, inner_seeds)
                    inner_residual[inner == k] = inner_y[inner == k] - predict_pair(inner_fit, inner_x[inner == k])
                    inner_meta.append(dict(fold=k, seeds=inner_seeds, iterations=rounds))
                pre = legacy.direct_w(inner_residual[:, 0], inner_residual[:, 1], inner)
                err = inner_residual[:, 1] - pre * inner_residual[:, 0]
                scales[fold] = 1.482602218505602 * np.median(abs(err - np.median(err)))
                if not np.isfinite(scales[fold]) or scales[fold] <= 0:
                    raise ValueError("Invalid Huber training-fold scale")
                fold_meta["huber_scale"] = scales[fold]
                fold_meta.pop("component")
            except Exception as exc:
                fold_meta.update(status="failed", component=fold_meta.get("component", "outer"),
                                 failure=f"{type(exc).__name__}: {exc}")
                meta["status"] = "failed"

        if not np.isfinite(predictions).all():
            meta.update(status="failed", failure="Incomplete outer predictions")
        else:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, predictions=predictions, folds=folds,
                                d=d, y=y, w_oracle=v, z_oracle=oracle_z, scales=scales)
            payload = buffer.getvalue()
            meta["payload_sha256"] = digest(payload)
    except Exception:
        meta.update(status="failed", failure=traceback.format_exc())
    meta["nuisance_seconds"] = perf_counter() - started
    return comparable_json(meta), payload


def stage2(task):
    stage1_meta, payload = task
    meta = {key: stage1_meta.get(key) for key in
            ("learner", "mode", "n", "law", "rep", "seed_key", "sample_sha256", "fold_sha256")}
    if payload is None:
        reason = stage1_meta.get("failure") or "Stage 1 failed; see fold-level log"
        rows = [dict(method=name, estimate=None, se=None, valid_estimator=False,
                     valid_interval=False, covered=False, length=None, max_t_covered=None,
                     max_t_length=None, seconds=0, metadata={"failure": reason})
                for name in methods.METHODS]
    else:
        if digest(payload) != stage1_meta["payload_sha256"]:
            raise ValueError("Nuisance cache checksum mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as a:
            p = a["predictions"]
            cache = dict(w=a["d"]-p[:, 0], z=a["y"]-p[:, 1], folds=a["folds"],
                         w_oracle=a["w_oracle"], z_oracle=a["z_oracle"], scales=a["scales"])
        if digest(cache["folds"].astype("<i8").tobytes()) != meta["fold_sha256"]:
            raise ValueError("Cached outer folds do not match the recorded checksum")
        random = methods.rng(meta["mode"], meta["n"], meta["law"], meta["rep"], "max-t")
        rows = methods.estimate_all(cache, meta["law"], random, THEORY[meta["law"]])
        if not np.isfinite(cache["scales"]).all():
            # Preserve the other eight estimators even when Huber calibration failed.
            for row in rows:
                if row["method"] == "Huber":
                    row.update(estimate=None, se=None, valid_estimator=False, valid_interval=False,
                               covered=False, length=None, max_t_covered=None, max_t_length=None,
                               metadata={"failure": "Invalid inner Huber scale; see stage 1 fold records"})
    meta["rows"] = rows
    return comparable_json(meta), None


def initialize_worker():
    global _thread_limits
    _thread_limits = threadpool_limits(1)
    warnings.filterwarnings("ignore", message=r"Unrecognized options detected: .*threads.*", category=OptimizeWarning)


@lru_cache(maxsize=1)
def science_signature():
    h = hashlib.sha256()
    for f in [Path(__file__).resolve(), ROOT/"experiment_manifest.json",
              ROOT/"baseline"/"main_seed_registry.csv.gz",
              *(ROOT/"spline_core"/name for name in MANIFEST["sources"])]:
        data = f.read_bytes()
        if f.name in MANIFEST["sources"] and digest(data) != MANIFEST["sources"][f.name]:
            raise ValueError(f"Frozen source changed: {f.name}")
        h.update(data)
    h.update(json_text(dict(schema=CACHE_SCHEMA, packages=PACKAGE_VERSIONS)).encode())
    return h.hexdigest()


def db_path(mode, learner, phase):
    return ROOT / ("nuisance_cache" if phase == 1 else "checkpoints") / mode / f"{learner}.sqlite"


@contextmanager
def read_db(path, mode, learner, phase):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        stored = db.execute("SELECT signature, mode, learner, stage FROM config").fetchone()
        if stored != (science_signature(), mode, learner, phase):
            raise ValueError(f"Cache provenance differs: {path}")
        yield db


@contextmanager
def write_db(mode, learner, phase):
    path = db_path(mode, learner, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30)) as db:
        db.execute("PRAGMA journal_mode=DELETE")
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS config(signature TEXT, mode TEXT, learner TEXT, stage INTEGER)")
            stored = db.execute("SELECT signature, mode, learner, stage FROM config").fetchone()
            expected = science_signature(), mode, learner, phase
            if stored is None:
                db.execute("INSERT INTO config VALUES(?,?,?,?)", expected)
            elif stored != expected:
                raise ValueError(f"Existing checkpoint has a different configuration: {path}")
            db.execute("CREATE TABLE IF NOT EXISTS records(n INTEGER,law TEXT,rep INTEGER,"
                       "metadata TEXT,payload BLOB,PRIMARY KEY(n,law,rep))")
        yield db


def keys_for(mode, n=None, law=None, stop=None):
    if mode == "smoke":
        ns = (n or MANIFEST["smoke_n"],)
        laws = (law or MANIFEST["smoke_law"],)
    else:
        ns = (n,) if n else MANIFEST["sample_sizes"]
        laws = (law,) if law else methods.LAWS
    limit = MANIFEST[f"{mode}_reps"]
    if stop is not None and not 1 <= stop <= limit:
        raise ValueError(f"--stop must be between 1 and {limit}")
    return [(n, law, rep) for n in ns for law in laws for rep in range(stop or limit)]


def run_stage(mode, learner, phase, keys, workers, resume):
    with write_db(mode, learner, phase) as db:
        existing = set(db.execute("SELECT n,law,rep FROM records"))
        done = existing.intersection(keys)
        if done and not resume:
            raise ValueError("Results exist for this selection. Re-run with --resume.")
        pending = [key for key in keys if key not in existing]
        if not pending:
            print(f"{learner} Stage {phase}: {len(done):,} saved. Nothing to compute.")
            return
        if phase == 1:
            dispatch(db, fit_nuisance, ((learner, mode, *key) for key in pending),
                     keys, len(done), workers, f"{learner} Stage 1")
        else:
            with read_db(db_path(mode, learner, 1), mode, learner, 1) as cache:
                cached = set(cache.execute("SELECT n,law,rep FROM records"))
                missing = set(keys) - cached
                if missing:
                    raise ValueError(f"Stage 1 lacks {len(missing):,} required records; complete Stage 1 first.")

                def jobs():
                    for key in pending:
                        text, blob = cache.execute(
                            "SELECT metadata,payload FROM records WHERE n=? AND law=? AND rep=?", key
                        ).fetchone()
                        yield json.loads(text), blob

                dispatch(db, stage2, jobs(), keys, len(done), workers, f"{learner} Stage 2")


def dispatch(db, task, jobs, keys, done, workers, label):
    with ProcessPoolExecutor(max_workers=min(workers, len(keys)-done),
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        queue = iter(jobs)
        pending = set()
        with tqdm(total=len(keys), initial=done, desc=label, unit="rep", ascii=True, mininterval=1) as bar:
            while True:
                for _ in range(workers - len(pending)):
                    job = next(queue, None)
                    if job is None:
                        break
                    pending.add(pool.submit(task, job))
                if not pending:
                    break
                ready, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                if not ready:
                    print(f"{label}: {bar.n:,}/{len(keys):,} committed; {len(pending)} submitted", flush=True)
                    continue
                saved = []
                for job in ready:
                    meta, payload = job.result()
                    saved.append((meta["n"], meta["law"], meta["rep"], json_text(meta), payload))
                with db:
                    db.executemany("INSERT INTO records VALUES(?,?,?,?,?)", saved)
                bar.update(len(saved))


def effective_workers(requested):
    count = os.cpu_count() or 1
    try:
        count = len(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.Error):
        pass
    for path in (Path("/sys/fs/cgroup/cpu.max"),):
        if path.is_file():
            quota, period = path.read_text().split()
            if quota != "max":
                count = min(count, max(1, int(quota)//int(period)))
    quota_v1 = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    if quota_v1.is_file() and (quota := int(quota_v1.read_text())) > 0:
        period = int(quota_v1.with_name("cpu.cfs_period_us").read_text())
        count = min(count, max(1, quota//period))
    return max(1, min(requested, count))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "pilot", "main"))
    parser.add_argument("--learner", required=True, choices=LEARNERS)
    parser.add_argument("--stage", required=True, choices=("1", "2", "both"))
    parser.add_argument("--workers", type=int, default=12, choices=range(1, 13))
    parser.add_argument("--n", type=int, choices=MANIFEST["sample_sizes"])
    parser.add_argument("--law", choices=methods.LAWS)
    parser.add_argument("--stop", type=int, help="Optional prefix of replications per selected condition")
    parser.add_argument("--resume", action="store_true")
    opts = parser.parse_args()
    keys = keys_for(opts.mode, opts.n, opts.law, opts.stop)
    workers = effective_workers(opts.workers)
    print(f"{opts.mode} | {opts.learner} | stage {opts.stage} | {len(keys):,} replications | {workers} workers",
          flush=True)
    # Linux cloud volumes can reject flock. Keep execution locks on local storage.
    lock_path = (
        Path("/tmp") / f"rank-dml-scenario-a-{os.getuid()}-{digest(os.fsencode(ROOT))[:24]}.lock"
        if sys.platform.startswith("linux")
        else ROOT / "checkpoints" / "running.lock"
    )
    lock = FileLock(lock_path)
    try:
        with lock.acquire(timeout=0):
            for phase in ((1, 2) if opts.stage == "both" else (int(opts.stage),)):
                run_stage(opts.mode, opts.learner, phase, keys, workers, opts.resume)
    except Timeout as exc:
        raise SystemExit("This project already has an active run.") from exc


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

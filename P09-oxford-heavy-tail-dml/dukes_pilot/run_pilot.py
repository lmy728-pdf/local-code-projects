"""Isolated, resumable paired pilot. Never executes a notebook."""
from __future__ import annotations
import os
# Applied before numerical libraries are loaded, including in spawned workers.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import legacy_adapter as legacy
from dukes_score import crossfit_initial, dukes_from_initial, score_test, DEFAULT_KERNEL_CONFIG

ROOT = Path(__file__).resolve().parent
METHODS = ("original_mean_dml", "standard_score", "dukes_dr_gcm")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fixed_config():
    import scipy, sklearn, joblib
    config = json.loads((ROOT / "configs/pilot.json").read_text(encoding="utf-8"))
    config["kernel"] = dict(DEFAULT_KERNEL_CONFIG, **config.get("kernel", {}))
    config["environment"] = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}
    config["hgbr_params_seed_zero"] = legacy.make_nuisance_model("mean", 0).get_params()
    names = ["run_pilot.py", "legacy_adapter.py", "dukes_score.py", "method_spec.md"]
    names.extend(p.name for p in sorted(ROOT.glob("_legacy*.py")))
    config["implementation_sha256"] = {name: sha256(ROOT / name) for name in names}
    config["config_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return config


def audit_gate():
    for name in ("legacy_audit.md", "source_manifest.json", "environment.json"):
        if not (ROOT / "audit" / name).exists():
            raise RuntimeError(f"Read-only audit is required first: audit/{name}")
    # Integrity verification traverses only the audited manifest, never notebooks.
    manifest = json.loads((ROOT / "audit/source_manifest.json").read_text(encoding="utf-8"))
    checked = []
    def walk(value):
        if isinstance(value, dict):
            path = value.get("path", value.get("absolute_path"))
            expected = value.get("sha256")
            if path and expected:
                if sha256(path) != expected:
                    raise RuntimeError(f"Audited source changed: {path}")
                checked.append(path)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(manifest)
    if len(checked) < 4:
        raise RuntimeError("Source manifest did not contain enough verifiable source files")
    return {"source_files_verified": len(checked), "status": "passed"}


def make_data(config, scenario, replication):
    seed = config["base_seed"] + 100000 * config["legacy_dgp_id"] + 1000 * scenario["legacy_error_id"] + replication
    family = "lognormal" if scenario["error_family"] == "lognormal" else "student_t"
    data = legacy.generate_data(config["n"], scenario["error_case"], config["dgp"], seed, theta0=config["theta0"], error_family=family)
    folds = list(KFold(config["n_folds"], shuffle=True, random_state=seed + config["fold_seed_offset"]).split(data[0]))
    return data, folds, seed


def initial_a(X, D, Y, folds, seed):
    d_hat, y_hat = np.empty_like(D), np.empty_like(Y)
    for k, (train, test) in enumerate(folds):
        model_d = legacy.make_nuisance_model("mean", seed + 100 * k + 1)
        model_y = legacy.make_nuisance_model("mean", seed + 100 * k + 2)
        model_d.fit(X[train], D[train])
        model_y.fit(X[train], Y[train])
        d_hat[test] = model_d.predict(X[test])
        y_hat[test] = model_y.predict(X[test])
    return d_hat, y_hat


def compute_task(config, scenario, replication):
    started = time.perf_counter()
    data, folds, seed = make_data(config, scenario, replication)
    X, D, Y, U = data
    cfseed = seed + config["fold_seed_offset"]
    fold_ids = np.empty(len(D), dtype=np.int64)
    for k, (train, test) in enumerate(folds):
        assert len(np.intersect1d(train, test)) == 0
        fold_ids[test] = k
    fold_hash = hashlib.sha256(fold_ids.tobytes()).hexdigest()
    base = dict(scenario=scenario["name"], error_family=scenario["error_family"], error_parameter=scenario["error_parameter"], dgp=config["dgp"], legacy_dgp_id=config["legacy_dgp_id"], legacy_error_id=scenario["legacy_error_id"], p=config["p"], n=config["n"], replication=replication, config_hash=config["config_hash"], dataset_seed=seed, crossfit_seed=cfseed, fold_hash=fold_hash, model_seeds=[{"treatment":cfseed+100*k+1,"outcome":cfseed+100*k+2} for k in range(config["n_folds"])], sample_u_mean=float(U.mean()), sample_u_variance=float(U.var(ddof=1)), sample_max_abs_u=float(np.abs(U).max()), sample_max_u=float(U.max()))
    rows, diagnostics = [], []
    arrays = {"fold_ids": fold_ids}
    timings = {}
    def fail(method, error):
        rows.append(dict(base, method=method, status="failed", covered=None, failure_type=getattr(error, "failure_type", type(error).__name__), failure_fold=getattr(error, "fold", None), failure_message=str(error)))
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.perf_counter()
        try:
            g_hat, ell_hat = initial_a(X, D, Y, folds, cfseed)
            dr, yr = D-g_hat, Y-ell_hat
            estimate, se = legacy.fit_mean_dml(dr, yr)
            if not np.isfinite([estimate, se]).all() or se <= 0:
                raise ValueError("Non-finite or degenerate A estimate/SE")
            low, high = estimate-1.96*se, estimate+1.96*se
            rows.append(dict(base, method=METHODS[0], status="ok", theta_hat=estimate, theta_se=se, ci_low=low, ci_high=high, ci_length=high-low, covered=bool(low <= config["theta0"] <= high), legacy_covered=bool(low <= config["theta0"] <= high), coverage_type="wald_coverage"))
            arrays.update(g_hat=g_hat, ell_hat=ell_hat, d_res=dr, y_res=yr)
        except Exception as error:
            fail(METHODS[0], error)
            g_hat = None
        timings["a_seconds"] = time.perf_counter()-t
        t = time.perf_counter()
        initial = None
        try:
            initial = crossfit_initial(X, D, Y, config["theta0"], folds, cfseed, treatment_predictions=g_hat)
            arrays.update(g_hat=initial["g_hat"], m_hat=initial["m_hat"])
            scores_b = (D-initial["g_hat"])*(Y-config["theta0"]*D-initial["m_hat"])
            test_b = score_test(scores_b)
            arrays["scores_B"] = scores_b
            rows.append(dict(base, method=METHODS[1], status="ok", theta_tested=config["theta0"], coverage_type="inverted_test_coverage_at_truth", covered=bool(test_b["accepted"]), **test_b))
        except Exception as error:
            fail(METHODS[1], error)
        timings["bc_initial_seconds"] = time.perf_counter()-t
        t = time.perf_counter()
        try:
            if initial is None:
                raise ValueError("Initial B/C predictions unavailable")
            result = dukes_from_initial(D, Y, config["theta0"], initial["g_hat"], initial["m_hat"], folds, config["kernel"])
            test_c = score_test(result["scores"])
            arrays.update(result["arrays"])
            arrays["scores_C"] = result["scores"]
            diagnostics = [dict(base, **fold) for fold in result["fold_diagnostics"]]
            rows.append(dict(base, method=METHODS[2], status="ok", theta_tested=config["theta0"], coverage_type="inverted_test_coverage_at_truth", covered=bool(test_c["accepted"]), **test_c))
            assert np.array_equal(result["standard_scores"], scores_b)
        except Exception as error:
            # Avoid a duplicate success if the sharing assertion itself fails.
            rows = [row for row in rows if row["method"] != METHODS[2]]
            fail(METHODS[2], error)
        timings["kernel_update_seconds"] = time.perf_counter()-t
    elapsed = time.perf_counter()-started
    for row in rows:
        row.update(elapsed_seconds=elapsed, warnings=[f"{w.category.__name__}: {w.message}" for w in caught], **timings)
    return {"rows": rows, "folds": diagnostics, "timings": timings, "elapsed_seconds": elapsed}, arrays


def persist_task(config, scenario, replication, run_dir):
    result, arrays = compute_task(config, scenario, replication)
    stem = f"{scenario['name']}_{replication:04d}"
    array_path = Path(run_dir) / "arrays" / (stem + ".npz")
    np.savez_compressed(array_path, **arrays)
    result["arrays_sha256"] = sha256(array_path)
    result["arrays_file"] = str(array_path.relative_to(run_dir))
    dump_json(Path(run_dir) / "tasks" / (stem + ".json"), result)
    return stem, result["elapsed_seconds"], sum(r["status"] != "ok" for r in result["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit-only", "smoke", "pilot", "resume"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16; default 4 keeps desktop responsive")
    print(json.dumps(audit_gate()), flush=True)
    if args.mode == "audit-only":
        return
    config = fixed_config()
    # All computational regression and algebra checks must precede model runs.
    marker = ROOT / "audit/validation.json"
    validation = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    if validation.get("status") != "passed" or validation.get("config_hash") != config["config_hash"]:
        raise RuntimeError("Run validate_pilot.py successfully for this exact implementation/config first")
    run_dir = args.run_dir or ROOT / "results" / ("pilot_" + config["config_hash"][:12])
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "results").resolve()) or run_dir == (ROOT / "results").resolve():
        raise RuntimeError("All run outputs must be in a new subdirectory of dukes_pilot/results")
    if args.mode == "resume" and not (run_dir / "config.json").exists():
        raise RuntimeError("resume requires an existing compatible run")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tasks", "arrays"):
        (run_dir / name).mkdir(exist_ok=True)
    if (run_dir / "config.json").exists():
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if saved != config:
            raise RuntimeError("Configuration changed: create a new run, never mix results")
    else:
        dump_json(run_dir / "config.json", config)
    limit = 2 if args.mode == "smoke" else config["replications"]
    pending = []
    for scenario in config["scenarios"]:
        for rep in range(limit):
            path = run_dir / "tasks" / f"{scenario['name']}_{rep:04d}.json"
            if path.exists():
                old = json.loads(path.read_text(encoding="utf-8"))
                assert len(old["rows"]) == 3 and {r["method"] for r in old["rows"]} == set(METHODS)
                assert all(r["config_hash"] == config["config_hash"] and r["scenario"] == scenario["name"] and r["replication"] == rep for r in old["rows"])
                if sha256(run_dir / old["arrays_file"]) != old["arrays_sha256"]:
                    raise RuntimeError(f"Array checkpoint integrity failure: {path}")
            else:
                pending.append((scenario, rep))
    started = time.perf_counter()
    print(f"Run: {run_dir}\nMode={args.mode}; missing datasets={len(pending)}; workers={args.workers}; inner threads=1", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        stream = Parallel(n_jobs=args.workers, return_as="generator_unordered")(
            delayed(persist_task)(config, scenario, rep, str(run_dir)) for scenario, rep in pending)
        for i, (stem, seconds, failures) in enumerate(stream, 1):
            if i <= 6 or i % 20 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} {stem}: {seconds:.2f}s, failed methods={failures}, wall={time.perf_counter()-started:.1f}s", flush=True)
    wall = time.perf_counter()-started
    event = {"timestamp_utc":datetime.now(timezone.utc).isoformat(), "mode":args.mode, "workers":args.workers, "inner_threads":1,"datasets_computed":len(pending),"wall_seconds":wall}
    with (run_dir / "execution_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    from summarize_pilot import summarize_run
    summarize_run(run_dir)
    print(f"Finished in {wall:.1f}s; {run_dir}", flush=True)


if __name__ == "__main__":
    main()

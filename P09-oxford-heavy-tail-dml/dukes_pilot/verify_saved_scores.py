"""Read-only independent verification of saved scores; no model fitting.

Usage: python verify_saved_scores.py results/pilot_CONFIG_HASH
Writes only audit/saved_score_verification.json. Does not import dukes_score or
run_pilot. Original arrays, task records and computational code are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def verify(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    maxima = dict(score_absolute=0.0, T_absolute=0.0, p_absolute=0.0,
                  variance_absolute=0.0, update_moment_absolute=0.0)
    counters = dict(tasks_read=0, C_successes=0, C_failures=0,
                    successful_C_verified=0, scores_checked=0, folds_checked=0)
    boundaries, problems = {}, []
    for path in sorted((run_dir / "tasks").glob("*.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        counters["tasks_read"] += 1
        c_rows = [row for row in task["rows"] if row["method"] == "dukes_dr_gcm"]
        if len(c_rows) != 1:
            problems.append({"task":path.name,"reason":"C record missing or duplicated"})
            continue
        recorded = c_rows[0]
        if recorded["config_hash"] != config["config_hash"]:
            problems.append({"task":path.name,"reason":"Config hash mismatch"})
            continue
        for fold in task.get("folds", []):
            for direction in ("G", "M"):
                key = (recorded["scenario"], direction)
                count = boundaries.setdefault(key, {"total":0,"interior":0,"upper":0,"lower":0,"constant_index":0})
                count["total"] += 1
                if fold["constant_index_"+direction]:
                    count["constant_index"] += 1
                elif fold["boundary_"+direction]:
                    side = fold["boundary_side_"+direction]
                    if side not in ("upper", "lower"):
                        problems.append({"task":path.name,"reason":"Invalid bandwidth boundary direction"})
                    else:
                        count[side] += 1
                else:
                    count["interior"] += 1
        if recorded["status"] != "ok":
            counters["C_failures"] += 1
            continue
        counters["C_successes"] += 1
        array_path = run_dir / task["arrays_file"]
        actual_hash = hashlib.sha256(array_path.read_bytes()).hexdigest()
        if actual_hash != task["arrays_sha256"]:
            problems.append({"task":path.name,"reason":"Array SHA256 mismatch"})
            continue
        try:
            with np.load(array_path, allow_pickle=False) as arrays:
                a, b = arrays["a"], arrays["b"]
                G, M, fold_ids = arrays["G"], arrays["M"], arrays["fold_ids"]
                rD, rY = np.empty_like(a), np.empty_like(b)
                for k in np.unique(fold_ids):
                    ix = fold_ids == k
                    # Direct formulas independent of the normalized projection implementation.
                    den_G, den_M = np.sum(G[ix]**2), np.sum(M[ix]**2)
                    if (den_G == 0 and np.any(G[ix] != 0)) or (den_M == 0 and np.any(M[ix] != 0)):
                        raise ValueError("Direct squared norm underflows; independent check is inconclusive")
                    alpha = np.sum(G[ix]*a[ix])/den_G if den_G > 0 else 0.0
                    beta = np.sum(M[ix]*b[ix])/den_M if den_M > 0 else 0.0
                    rD[ix], rY[ix] = a[ix]-alpha*G[ix], b[ix]-beta*M[ix]
                    moment = max(abs(np.mean(G[ix]*rD[ix])), abs(np.mean(M[ix]*rY[ix])))
                    maxima["update_moment_absolute"] = max(maxima["update_moment_absolute"], float(moment))
                    counters["folds_checked"] += 1
                # In author's R notation, Gs is our M and Ms is our G.
                score = rD*rY - M*rY - G*rD
                n = len(score)
                average = np.sum(score)/n
                variance = np.sum((score-average)**2)/(n-1)
                T = np.sqrt(n)*average/np.sqrt(variance)
                p = chi2.sf(T*T, 1)
                if not np.isfinite(score).all() or not np.isfinite([variance,T,p]).all() or variance <= 0:
                    raise ValueError("Independent reconstruction produced nonfinite or degenerate values")
                differences = {"score_absolute":float(np.max(np.abs(score-arrays["scores_C"]))),
                               "T_absolute":float(abs(T-recorded["T"])),
                               "p_absolute":float(abs(p-recorded["p_value"])),
                               "variance_absolute":float(abs(variance-recorded["score_variance"]))}
                for key,value in differences.items():
                    maxima[key] = max(maxima[key],value)
                np.testing.assert_allclose(score,arrays["scores_C"],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose([T,p,variance],
                                           [recorded["T"],recorded["p_value"],recorded["score_variance"]],
                                           rtol=1e-12,atol=1e-12)
                counters["successful_C_verified"] += 1
                counters["scores_checked"] += n
        except Exception as exc:
            problems.append({"task":path.name,"reason":str(exc),"type":type(exc).__name__})
    planned = len(config["scenarios"])*config["replications"]
    payload = {
        "status":"passed" if not problems else "failed",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "run_dir":str(run_dir),"config_hash":config["config_hash"],
        "scope":"Saved successful C scores, direct fold projections, complete-score variance, T and chi-square p; no bandwidth refitting or R execution",
        "complete_run":counters["tasks_read"] == planned,
        "planned_datasets":planned,"pending_datasets":planned-counters["tasks_read"],
        "tolerance":{"rtol":1e-12,"atol":1e-12},
        "counts":counters,"maximum_absolute_differences":maxima,
        "bandwidth_counts":[dict(scenario=key[0],direction=key[1],**value) for key,value in sorted(boundaries.items())],
        "problems":problems,
    }
    output = Path(__file__).resolve().parent / "audit" / "saved_score_verification.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(output)
    return payload, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    args = parser.parse_args()
    payload, output = verify(args.run_dir)
    print(json.dumps({"status":payload["status"],"complete_run":payload["complete_run"],
                      "counts":payload["counts"],"maximum_absolute_differences":payload["maximum_absolute_differences"],
                      "report":str(output)},ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)

"""Numerical checks and checkpoint inventory. Does not run formal experiments."""
from run import db_path, ROOT, SIZES, METHODS, start_worker
from science import generate, sorted_wilcoxon, random_stream, tail_cutoff
import legacy_core as legacy
import argparse
import csv
import json
import sqlite3
import numpy as np
from scipy.stats import t

def check_rank():
    start_worker()
    cases = []
    for scenario in ("C1", "C2"):
        for rep in range(16):
            n = 500 if rep >= 12 else 100
            _, y, _, _, rare, folds, _, _ = generate("validation", scenario, n, rep)
            yy = y-rare[:, None]*np.array([1., np.sqrt(n)]) if scenario == "C1" else y
            w, z = yy.T
            if rep % 4 == 0:
                w = np.round(w, 1)
                z = np.round(z, 1)
            if rep % 5 == 0:
                folds[:3] = 0
            cases.append((f"{scenario}_{rep}", w, z, folds))
    cases.append(("flat_minimum", np.array([0.,1.,0.,1.]), np.array([0.,2.,0.,4.]),
                  np.array([0,0,1,1])))
    output = []
    for label, w, z, folds in cases:
        point = sorted_wilcoxon(w,z,folds)
        reference = legacy.direct_w(w,z,folds)
        ddw, ddz, weights = [], [], []
        for k in np.unique(folds):
            x, y = w[folds==k], z[folds==k]
            i, j = np.triu_indices(len(x), 1)
            ddw.extend(x[i]-x[j])
            ddz.extend(y[i]-y[j])
            weights.extend(np.full(len(i), 1/(len(w)*(len(x)-1))))
        ddw, ddz, weights = map(np.asarray, (ddw, ddz, weights))
        def objective(value):
            return np.sum(weights*abs(ddz-value*ddw))
        def slope(value):
            return -np.sum(weights*ddw*np.sign(ddz-value*ddw))
        left, right = slope(point-1e-10), slope(point+1e-10)
        assert left <= 1e-12 and right >= -1e-12, (label, left, right)
        assert abs(point-reference) < 2e-10 and abs(objective(point)-objective(reference)) < 1e-10
        output.append(dict(test=label, n=len(w), estimate=point, reference=reference,
            abs_estimate_error=abs(point-reference),
            objective_difference=objective(point)-objective(reference),
            left_subgradient_probe=left, right_subgradient_probe=right))
    target = ROOT/"汇总表_CSV"/"rank_solver_validation.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    for n in SIZES:
        assert abs(t.sf(tail_cutoff(n)*np.sqrt(3), 3)-n**(-.75))<1e-11
    for stream in ("data", "outer-folds", "huber-inner-0"):
        pilot, p = random_stream("pilot","C1",500,0,stream)
        main, m = random_stream("main","C1",500,0,stream)
        assert p != m
        assert not np.array_equal(pilot.normal(size=4),main.normal(size=4))
    print(f"PASSED {len(output)} rank equivalence checks; max slope estimate difference",
          max(x["abs_estimate_error"] for x in output), flush=True)

def inventory(stage):
    cases = [("C1",500,5),("C1",8000,2),("C2",500,5)] if stage=="smoke" else [
        (case,n,100 if stage=="pilot" else 2000) for case in ("C1","C2") for n in SIZES]
    keys = set()
    summaries = []
    for case,n,target in cases:
        with sqlite3.connect(db_path(stage,case,n).as_uri()+"?mode=ro",uri=True) as db:
            assert db.execute("PRAGMA quick_check").fetchone()[0]=="ok"
            rows = list(db.execute("SELECT rep,record,arrays IS NOT NULL FROM results ORDER BY rep"))
        assert [row[0] for row in rows]==list(range(target))
        failures = {}
        for rep, raw, has_cache in rows:
            record = json.loads(raw)
            assert (record["stage"],record["case"],record["n"],record["rep"])==(stage,case,n,rep)
            assert [x["method"] for x in record["methods"]]==list(METHODS)
            if record.get("error"):
                failures["DATA:"+record["error"]] = failures.get("DATA:"+record["error"],0)+1
            else:
                assert has_cache
                for key in ("seed_key", "fold_seed_key"):
                    assert record[key] not in keys
                    _, expected = random_stream(stage,case,n,rep,"data" if key=="seed_key" else "outer-folds")
                    assert record[key]==expected
                    keys.add(record[key])
            for result in record["methods"]:
                if not result["valid_interval"]:
                    label = result["method"]+":"+str(result["failure"])
                    failures[label] = failures.get(label,0)+1
        if stage in ("smoke","pilot"):
            assert not any(key.startswith("DATA:") for key in failures), failures
            assert all(value<target for value in failures.values()), failures
        summaries.append(dict(stage=stage,case=case,n=n,completed=target,failures=failures))
    (ROOT/"logs"/f"{stage}_integrity.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    print(f"PASSED {stage} integrity for {sum(x['completed'] for x in summaries)} replications", flush=True)
    for item in summaries:
        if item["failures"]:
            print(item, flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["math","smoke","pilot","main"])
    a=p.parse_args()
    check_rank() if a.stage=="math" else inventory(a.stage)


"""Time Scenario A's actual GBDT Stage 1 without writing experimental results."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter


WORKERS = 12


def time_replication(key):
    from run import fit_nuisance

    metadata, _ = fit_nuisance(("gbdt", "main", *key))
    failure = metadata.get("failure")
    if metadata["status"] != "ok" and not failure:
        failure = next((fold.get("failure") for fold in metadata["diagnostics"]
                        if fold.get("failure")), "Unknown Stage 1 failure")
    return metadata["nuisance_seconds"], failure, os.getpid()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n", type=int, help="Only time this sample size")
    parser.add_argument("--law", help="Only time this error distribution")
    parser.add_argument("--reps", type=int, default=1, help="Formal replication IDs per condition")
    args = parser.parse_args()

    project = args.project.resolve()
    if not (project / "run.py").is_file():
        parser.error(f"Scenario A run.py was not found in {project}")
    sys.path.insert(0, str(project))
    import run
    from tqdm import tqdm

    sample_sizes = run.MANIFEST["sample_sizes"]
    if args.n is not None and args.n not in sample_sizes:
        parser.error(f"--n must be one of {sample_sizes}")
    if args.law is not None and args.law not in run.methods.LAWS:
        parser.error(f"Unknown --law. Choose one of {', '.join(run.methods.LAWS)}")
    if not 1 <= args.reps <= run.MANIFEST["main_reps"]:
        parser.error(f"--reps must be between 1 and {run.MANIFEST['main_reps']}")
    expected = dict(line.split("==", 1) for line in
                    (project / "requirements.txt").read_text(encoding="utf-8-sig").splitlines()
                    if line.strip() and not line.lstrip().startswith("#"))
    for name, installed in run.PACKAGE_VERSIONS.items():
        if installed != expected[name]:
            parser.error(f"{name} is {installed}, but the project requires {expected[name]}")
    run.science_signature()

    keys = run.keys_for("main", n=args.n, law=args.law, stop=args.reps)
    random.Random(20261125).shuffle(keys)
    workers = min(run.effective_workers(WORKERS), len(keys))
    lock_path = (
        Path("/tmp") / f"rank-dml-scenario-a-{os.getuid()}-{run.digest(os.fsencode(project))[:24]}.lock"
        if sys.platform.startswith("linux") else project / "checkpoints" / "running.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    durations = {n: [] for n in sorted({key[0] for key in keys})}
    worker_pids = set()
    failures = 0
    try:
        with run.FileLock(lock_path).acquire(timeout=0):
            print(f"Scenario A | GBDT Stage 1 | {len(keys)} replications | "
                  f"{workers}/{WORKERS} requested workers | Python {platform.python_version()} | "
                  f"{platform.system()} | scikit-learn {run.PACKAGE_VERSIONS['scikit-learn']}", flush=True)
            started = perf_counter()
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=run.initialize_worker) as pool:
                futures = {pool.submit(time_replication, key): key for key in keys}
                for future in tqdm(as_completed(futures), total=len(keys), unit="rep", desc="GBDT", ascii=True):
                    n, law, rep = futures[future]
                    seconds, failure, pid = future.result()
                    worker_pids.add(pid)
                    if failure:
                        failures += 1
                        tqdm.write(f"Failed n={n} law={law} rep={rep}: {failure}")
                    else:
                        durations[n].append(seconds)
            elapsed = perf_counter() - started
    except run.Timeout as exc:
        raise SystemExit("Another Scenario A run holds the project lock. Wait for it to finish.") from exc

    successful = len(keys) - failures
    rate = 60 * successful / elapsed
    print(f"Wall time: {elapsed:.2f} s | throughput: {rate:.2f} successful reps/min | "
          f"workers used: {len(worker_pids)} | success: {successful} | failed: {failures}")
    for n, seconds in durations.items():
        if seconds:
            print(f"n={n}: {len(seconds)} successful | median {median(seconds):.2f} s/rep")
    if not failures and args.n is None and args.law is None:
        total = len(run.keys_for("main"))
        print(f"Rough Stage 1 estimate for {total:,} replications at this rate: {total / rate / 60:.2f} h")
    return int(failures > 0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

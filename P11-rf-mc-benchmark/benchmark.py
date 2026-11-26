"""Portable RF throughput benchmark. No project imports, no GPU or external data."""
import os

# Must precede numerical imports, including imports in spawned children.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
              'OMP_THREAD_LIMIT'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import platform
import re
import socket
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits, threadpool_info
from tqdm import tqdm

VERSION = '1'


def atomic_text(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, '\ufeff' + out.getvalue())


def replication(index, seed, n, trees):
    start = time.perf_counter()
    row = dict(replication=index, seed_key=f'{seed}:{index}', theta=None,
               elapsed_seconds=None, status='failed', error='', pid=os.getpid(),
               started_perf=start, ended_perf=None)
    try:
        # Replication identity is independent of worker count, order, and run-id.
        streams = np.random.SeedSequence([seed, index]).spawn(12)
        rng = np.random.default_rng(streams[0])
        cov = 0.5 ** np.abs(np.arange(10)[:, None] - np.arange(10)[None, :])
        X = rng.standard_normal((n, 10)) @ np.linalg.cholesky(cov).T
        v, eps = rng.standard_normal((2, n))
        D = .6*np.sin(X[:, 0]) + .4*X[:, 1] + .3*X[:, 2]*X[:, 3] + v
        Y = D + .7*np.cos(X[:, 0]) + .5*X[:, 1]**2 + .3*np.sin(X[:, 2]) + .25*X[:, 3]*X[:, 4] + eps
        folds = KFold(5, shuffle=True, random_state=int(streams[1].generate_state(1)[0]))
        d_hat, y_hat = np.empty(n), np.empty(n)
        with threadpool_limits(limits=1):
            for fold, (train, test) in enumerate(folds.split(X)):
                for target, fitted, offset in ((D, d_hat, 0), (Y, y_hat, 1)):
                    model = RandomForestRegressor(
                        n_estimators=trees, min_samples_leaf=10, max_features=1.0,
                        max_depth=None, bootstrap=True, n_jobs=1,
                        random_state=int(streams[2 + 2*fold + offset].generate_state(1)[0]))
                    model.fit(X[train], target[train])
                    fitted[test] = model.predict(X[test])
            dr, yr = D - d_hat, Y - y_hat
            # OLS with intercept, reporting only the residual-D slope.
            dr, yr = dr - dr.mean(), yr - yr.mean()
            denom = float(dr @ dr)
            if denom <= np.finfo(float).tiny:
                raise ValueError('Degenerate residual D')
            row['theta'] = float(dr @ yr / denom)
            if not np.isfinite(row['theta']):
                raise ValueError('Non-finite theta')
        row['status'] = 'success'
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row['ended_perf'] = time.perf_counter()
    row['elapsed_seconds'] = row['ended_perf'] - start
    return row


def cpu_info():
    model = platform.processor() or 'unknown'
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                model = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    else:
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    model = line.split(':', 1)[1].strip()
                    break
        except OSError:
            pass
    affinity = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        try:
            import psutil
            affinity = len(psutil.Process().cpu_affinity())
        except (ImportError, AttributeError, OSError):
            pass
    quota = ''
    try:
        limit, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        quota = '' if limit == 'max' else int(limit)/int(period)
    except (OSError, ValueError):
        pass
    return dict(cpu_model=model, os=platform.platform(), hostname=socket.gethostname(),
                python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                logical_cpus=os.cpu_count(), affinity_logical_cpus=affinity,
                process_available_cpus=getattr(os, 'process_cpu_count', os.cpu_count)(),
                cgroup_v2_cpu_quota=quota)


class MemorySampler:
    """Sample RSS for parent + descendants; shared pages may be counted repeatedly."""
    def __init__(self):
        self.stop = threading.Event()
        self.peak = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            import psutil
            parent = psutil.Process()
            while not self.stop.is_set():
                total = 0
                for proc in [parent] + parent.children(recursive=True):
                    try:
                        total += proc.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak = max(self.peak or 0, total)
                self.stop.wait(.2)
        except (ImportError, OSError):
            pass


def concurrency(rows):
    events = []
    for row in rows:
        if row['started_perf'] is not None and row['ended_perf'] is not None:
            events.extend([(row['started_perf'], 1), (row['ended_perf'], -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def summarize(state):
    cfg, rows, sessions = state['config'], state['rows'], state['sessions']
    ok = [r for r in rows if r['status'] == 'success']
    durations = [r['elapsed_seconds'] for r in ok]
    total = sum(s['wall_seconds'] for s in sessions)
    complete = len(ok) == cfg['reps']
    eligible = complete and len(sessions) == 1 and sessions[0]['finished']
    return dict(**state['environment'], **cfg, completed_repetitions=len(rows),
                successes=len(ok), failures=len(rows)-len(ok),
                wall_clock_total_seconds=total,
                replication_mean_seconds=float(np.mean(durations)) if durations else '',
                replication_median_seconds=float(np.median(durations)) if durations else '',
                replication_p90_seconds=float(np.percentile(durations, 90)) if durations else '',
                replications_per_minute=len(ok)*60/total if total else 0,
                actual_peak_concurrent_tasks=max((s['peak_concurrent_tasks'] for s in sessions), default=0),
                sampled_peak_process_tree_rss_mib=max((s.get('peak_rss_mib') or 0 for s in sessions), default=0) or '',
                session_count=len(sessions), fair_comparison_eligible=eligible,
                speedup_vs_workers12='', baseline_directory='', comparison_key=state['comparison_key'])


def publish(folder, summary):
    write_csv(folder / 'benchmark_summary.csv', [summary])
    lines = ['# RF Monte Carlo CPU benchmark', '', '| Metric | Value |', '|---|---|']
    lines += [f'| {key} | {value} |' for key, value in summary.items()]
    lines += ['', '## Measurement notes',
              '- Wall time uses perf_counter: pool startup, scheduling, computation, checkpoints and shutdown. Environment setup, final reports and plots are excluded.',
              '- Throughput counts successful replications only. Duration statistics use successful tasks and exclude queue waiting.',
              '- Actual concurrency is the peak overlap of task execution intervals, NOT exclusive cores or CPU utilization.',
              '- RSS is sampled every 0.2 seconds for parent + children. It is an approximate peak, may double-count shared pages, and excludes report plotting.',
              '- Resume totals sum observed session durations. An interrupted session is measured only through its last checkpoint; resume results are excluded from speedup.',
              '- Speedup requires a complete uninterrupted workers=12 baseline, matching n/reps/trees/seed, code, software, machine and run-id.',
              '- theta is a workload sanity check, not a statistical-method evaluation.', '',
              '## Figure index', '- [Replication durations (vector PDF)](replication_durations.pdf)']
    atomic_text(folder / 'benchmark_report.md', '\n'.join(lines) + '\n')


def refresh_comparisons(root):
    entries = []
    for path in root.glob('*/checkpoint.json'):
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            entries.append((path.parent, summarize(state)))
        except (OSError, ValueError, KeyError):
            continue
    for folder, summary in entries:
        baseline = next(((f, s) for f, s in entries if s['comparison_key'] == summary['comparison_key']
                         and s['workers'] == 12 and s['fair_comparison_eligible']), None)
        if baseline and summary['fair_comparison_eligible']:
            summary['speedup_vs_workers12'] = summary['replications_per_minute']/baseline[1]['replications_per_minute']
            summary['baseline_directory'] = str(baseline[0])
        publish(folder, summary)


def plot_rows(folder, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([r['replication'] for r in rows], [r['elapsed_seconds'] for r in rows],
           color=['#2274A5' if r['status'] == 'success' else '#C44536' for r in rows])
    ax.set(xlabel='Replication (zero-based)', ylabel='Task wall time (seconds)',
           title='RF Monte Carlo workload duration (blue: success; red: failed)')
    fig.tight_layout()
    fig.savefig(folder / 'replication_durations.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('n', 1000), ('reps', 40), ('workers', 12), ('trees', 500), ('seed', 20261117)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--run-id', default='default')
    parser.add_argument('--force', action='store_true', help='Archive prior measurement and rerun all tasks')
    args = parser.parse_args()
    if args.n < 5 or min(args.reps, args.workers, args.trees) < 1 or args.seed < 0:
        parser.error('n >= 5; reps/workers/trees >= 1; seed >= 0 required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('run-id must start with a letter/digit and contain only letters, digits, _, ., -')
    env = cpu_info()
    cfg = {k: getattr(args, k) for k in ('n', 'reps', 'workers', 'trees', 'seed', 'run_id')}
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = dict(config=cfg, environment=env, code_hash=code_hash, version=VERSION)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    compare = dict(identity, config={k:v for k,v in cfg.items() if k != 'workers'})
    comparison_key = hashlib.sha256(json.dumps(compare, sort_keys=True).encode()).hexdigest()
    root = args.output.resolve() / args.run_id
    folder = root / f'n{args.n}_r{args.reps}_t{args.trees}_w{args.workers}_s{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive lock protects checkpoints against accidental duplicate launches.
    lock = root / (folder.name + '.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f'Run lock exists: {lock}. If the old process has stopped, remove this lock manually.')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        if args.force and folder.exists():
            import uuid
            archive = root / '_archive'
            archive.mkdir(exist_ok=True)
            folder.rename(archive / (folder.name + '_' + uuid.uuid4().hex[:12]))
        folder.mkdir(exist_ok=True)
        checkpoint = folder / 'checkpoint.json'
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding='utf-8'))
            if state['signature'] != signature:
                parser.error('Environment/code changed. Use a new --run-id or --force.')
        else:
            state = dict(signature=signature, comparison_key=comparison_key, config=cfg,
                         environment=env, rows=[], sessions=[], threadpools=threadpool_info())
        # Both successes and recorded failures are terminal; --force retries the full measurement.
        done = {row['replication'] for row in state['rows']}
        pending = [i for i in range(args.reps) if i not in done]
        if not pending:
            print('All replications already recorded; no measurement repeated. Use --force to remeasure.')
            refresh_comparisons(root)
            plot_rows(folder, state['rows'])
            return 0 if all(r['status'] == 'success' for r in state['rows']) else 1
        session = dict(wall_seconds=0, finished=False, peak_concurrent_tasks=0, peak_rss_mib=None)
        state['sessions'].append(session)
        current_rows = []
        memory = MemorySampler()
        start = time.perf_counter()
        memory.thread.start()

        def save(finished=False):
            session.update(wall_seconds=time.perf_counter()-start, finished=finished,
                           peak_concurrent_tasks=concurrency(current_rows),
                           peak_rss_mib=memory.peak/1024**2 if memory.peak else None)
            state['rows'].sort(key=lambda row: row['replication'])
            write_csv(folder / 'replications.csv', state['rows'])
            atomic_text(checkpoint, json.dumps(state, indent=2, allow_nan=False))

        save()
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
                futures = {pool.submit(replication, i, args.seed, args.n, args.trees): i for i in pending}
                with tqdm(total=args.reps, initial=len(done), desc=f'workers={args.workers}', unit='rep') as bar:
                    for future in as_completed(futures):
                        try:
                            row = future.result()
                        except Exception as exc:
                            row = dict(replication=futures[future], seed_key=f'{args.seed}:{futures[future]}',
                                       theta=None, elapsed_seconds=0.0, status='failed',
                                       error=f'Worker failure: {exc}', pid=None, started_perf=None, ended_perf=None)
                        state['rows'].append(row)
                        current_rows.append(row)
                        save()
                        bar.update(1)
                        bar.set_postfix(ok=sum(r['status']=='success' for r in state['rows']), refresh=False)
        finally:
            memory.stop.set()
            memory.thread.join()
            save(finished=len(state['rows']) == args.reps)
        refresh_comparisons(root)
        plot_rows(folder, state['rows'])
        summary = summarize(state)
        print(f"\nOutput: {folder}\nWall: {summary['wall_clock_total_seconds']:.3f}s; "
              f"success={summary['successes']}; failed={summary['failures']}; "
              f"peak active tasks={summary['actual_peak_concurrent_tasks']}")
        return 0 if summary['failures'] == 0 else 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())

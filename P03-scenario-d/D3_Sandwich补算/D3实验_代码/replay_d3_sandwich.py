"""Replay Scenario D3 rank-score covariance on the original formal samples."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

CODE = Path(__file__).resolve().parent
OUTPUT = CODE.parent
PROJECT = OUTPUT.parent
sys.path.insert(0, str(PROJECT))
import dgp
from estimators import AD_DRAWS, interval
from run import initialize_worker, signature
import numpy as np
import pandas as pd
from scipy.special import eval_legendre
from tqdm import tqdm

METHODS = ("Rank-Ad", "Odd-only")
VARIANTS = ("Naive Wald", "Original max-t", "Sandwich Wald", "Sandwich max-t")


@lru_cache(maxsize=None)
def score_grid(size, degrees):
    u = np.arange(1, size + 1) / (size + 1)
    scores = np.column_stack([
        np.sqrt(2 * degree + 1) * eval_legendre(degree, 2 * u - 1)
        for degree in degrees
    ])
    return scores - scores.mean(axis=0)


def fold_pseudovalues(w, residual, degrees):
    """Centered delete-one pseudo-values of each mean rank score in a fold."""
    order = np.argsort(residual, kind="stable")
    ordered = residual[order]
    tol = 64 * np.finfo(float).eps * np.max(abs(ordered - np.median(ordered)))
    size = len(w)
    weights = w[order, None]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered) > tol) + 1]
    ends = np.r_[starts[1:], size]
    counts = (ends - starts)[:, None]
    group_weights = np.add.reduceat(weights, starts, axis=0)
    short_grid = score_grid(size - 1, degrees)
    zero = np.zeros((1, len(degrees)))
    grid_sum = np.vstack((zero, np.cumsum(short_grid, axis=0)))
    # Each tie group uses the mean score over the group ranks.
    before = group_weights * (grid_sum[ends.clip(max=size-1)] - grid_sum[starts]) / counts
    after = group_weights * (grid_sum[ends - 1] - grid_sum[(starts - 1).clip(min=0)]) / counts
    left = np.vstack((zero, np.cumsum(before[:-1], axis=0)))
    right = np.vstack((np.cumsum(after[:0:-1], axis=0)[::-1], zero))
    retained = (grid_sum[ends - 1] - grid_sum[starts]) / np.maximum(counts - 1, 1)
    remainder = (np.repeat(group_weights, counts.ravel(), axis=0) - weights) * np.repeat(
        retained, counts.ravel(), axis=0
    )
    full_scores = np.add.reduceat(score_grid(size, degrees), starts, axis=0) / counts
    full_sum = (group_weights * full_scores).sum(axis=0)
    pseudo = full_sum - np.repeat(left + right, counts.ravel(), axis=0) - remainder
    centered = np.empty_like(pseudo)
    centered[order] = pseudo - pseudo.mean(axis=0)
    return centered


def score_covariance(w, residual, folds):
    scores = np.empty((len(w), 6))
    for fold in np.unique(folds):
        mask = folds == fold
        values = fold_pseudovalues(w[mask], residual[mask], (1, 2, 3, 4, 5, 6))
        values *= np.sqrt(mask.sum() / (mask.sum() - 1))
        scores[mask] = values
    return scores.T @ scores / len(w)


def replay(row):
    job = tuple(row[key] for key in ("mode", "scenario", "n", "rep"))
    _, _, n, rep = job
    seed_keys = row["seeds"]
    if seed_keys != {s: dgp.seed_key(*job, s) for s in dgp.STREAMS}:
        raise ValueError(f"Seed mismatch at {job}")
    data = dgp.generate(*job)
    w, z, folds = data["d"], data["y"], data["folds"]
    digest = hashlib.sha256(np.column_stack([w, z]).tobytes() + folds.tobytes()).hexdigest()
    if digest != row["input_sha256"] or np.bincount(folds).tolist() != row["fold_sizes"]:
        raise ValueError(f"Original sample hash or fold sizes disagree at {job}")

    by_name = {m["name"]: m for m in row["methods"]}
    pilot = by_name["Rank-W"]["point"]
    meat = score_covariance(w, z - pilot * w, folds)
    results = []
    for name in METHODS:
        method = by_name[name]
        diag = method["diagnostics"]
        degrees = np.asarray(diag["degrees"])
        slopes = np.asarray(diag["slopes"])
        raw_info = np.asarray(diag["raw_information"])
        corrected_info = np.asarray(diag["corrected_information"])
        selected = diag["selected_count"] - 1
        point = method["point"]
        if not method["valid_point"] or point != diag["candidates"][selected]:
            raise ValueError(f"Original point/selection disagree at {job} for {name}")
        loadings = np.tril(np.tile(slopes, (len(slopes), 1)))
        # Match D1's finite-sample G/I adjustment while changing the score covariance.
        loadings /= np.sqrt(raw_info * corrected_info)[:, None]
        cov = loadings @ meat[np.ix_(degrees - 1, degrees - 1)] @ loadings.T / n
        se = np.sqrt(np.diag(cov))
        if not np.all(np.isfinite(se) & (se > 0)):
            raise ValueError(f"Degenerate candidate covariance at {job} for {name}")
        correlation = cov / np.outer(se, se)
        eig, vectors = np.linalg.eigh((correlation + correlation.T) / 2)
        if np.min(eig) < -1e-10:
            raise ValueError(f"Candidate correlation is not PSD at {job} for {name}")
        key = seed_keys["rank_ad_sandwich_max_t"] + [int(name == "Odd-only")]
        rng = np.random.default_rng(np.random.SeedSequence(key))
        draws = rng.normal(size=(AD_DRAWS, len(se))) @ (vectors * np.sqrt(eig.clip(0))).T
        critical = float(np.quantile(np.max(abs(draws), axis=1), 0.95))
        old = method["intervals"]
        new_wald = interval(point, se[selected], row["theta0"])
        new_max_t = interval(point, se[selected], row["theta0"], critical=critical)
        cis = (old["naive"], old["max_t"], new_wald, new_max_t)
        record = dict(n=n, rep=rep, method=name, theta0=row["theta0"], point=point,
                      rank_w_pilot=pilot, selected_L=diag["selected_L"],
                      even_weight_share=diag["even_weight_share"], input_sha256=digest,
                      original_max_t_critical=diag["max_t_critical"],
                      sandwich_max_t_critical=critical,
                      candidate_degrees_json=json.dumps(degrees.tolist()),
                      candidate_covariance_json=json.dumps(cov.tolist()))
        for variant, ci in zip(("naive", "max_t", "sandwich", "sandwich_max_t"), cis):
            record.update({f"{variant}_{k}": v for k, v in ci.items()})
        results.append(record)
    return results


def load_checkpoint():
    path = PROJECT / "checkpoints" / "main.sqlite"
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        original_signature = db.execute("SELECT signature FROM meta WHERE id=1").fetchone()[0]
        if signature() != original_signature:
            raise ValueError("Original source or numerical environment differs from the checkpoint.")
        rows = [
            json.loads(raw) for raw, in db.execute(
                "SELECT json FROM replications WHERE scenario='D3' ORDER BY n,rep"
            )
        ]
    expected = [(n, r) for n in dgp.SIZES for r in range(dgp.COUNTS["main"])]
    if [(r["n"], r["rep"]) for r in rows] != expected:
        raise ValueError("Incomplete or unexpected D3 checkpoint.")
    return rows


def summarize(frame):
    rows = []
    for (method, n), group in frame.groupby(["method", "n"], sort=False):
        sd = group.point.std(ddof=1)
        for label, prefix in zip(VARIANTS, ("naive", "max_t", "sandwich", "sandwich_max_t")):
            coverage = group[f"{prefix}_covered"].mean()
            mcse = np.sqrt(coverage * (1 - coverage) / len(group))
            midse = group[f"{prefix}_se"].median()
            rows.append(dict(method=method, n=n, inference=label, reps=len(group),
                             valid=int(group[f"{prefix}_valid"].sum()), coverage=coverage,
                             coverage_mcse=mcse, coverage_mc95_low=max(0., coverage - 1.96 * mcse),
                             coverage_mc95_high=min(1., coverage + 1.96 * mcse),
                             median_se=midse, empirical_sd=sd, median_se_over_sd=midse / sd,
                             mean_length=group[f"{prefix}_length"].mean(),
                             mean_point=group.point.mean()))
    return pd.DataFrame(rows)


def save_figure(summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, layout="constrained")
    colors = ("#64748b", "#64748b", "#027c75", "#027c75")
    for ax, method in zip(axes, METHODS):
        ax.axhline(.95, c="#b84a39", ls="--", lw=1, label="95% reference")
        for j, (variant, color) in enumerate(zip(VARIANTS, colors)):
            rows = summary[(summary.method == method) & (summary.inference == variant)]
            x = np.arange(len(rows)) + (j - 1.5) * .055
            ax.errorbar(x, rows.coverage, yerr=1.96 * rows.coverage_mcse, label=variant,
                        marker="o" if j % 2 == 0 else "s", lw=1.5, capsize=2.5, ms=4,
                        c=color, ls="-" if j % 2 == 0 else ":")
        ax.set_xticks(range(3), ("500", "1,000", "2,000"))
        ax.set_xlabel("Sample size")
        ax.set_title(method, loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_ylim(min(.85, summary.coverage_mc95_low.min() - .015), 1.0)
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Empirical coverage of zero")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False, fontsize=8)
    fig.suptitle("Scenario D3 | Rank-score covariance", fontweight="bold")
    fig.savefig(folder / "D3_Sandwich_coverage.pdf")
    fig.savefig(folder / "D3_Sandwich_coverage.png", dpi=150)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--verify-only", action="store_true", help="Replay one saved sample per size.")
    args = parser.parse_args()
    start = perf_counter()
    rows = load_checkpoint()
    if args.verify_only:
        initialize_worker()
        verify_pseudovalues()
        for row in rows[::dgp.COUNTS["main"]]:
            print(row["n"], [(r["method"], r["sandwich_se"]) for r in replay(row)])
        return
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as pool:
        batches = list(tqdm(pool.map(replay, rows, chunksize=8), total=len(rows), unit="rep",
                            desc="D3 covariance", ascii=True))
    frame = pd.DataFrame(r for batch in batches for r in batch)
    if len(frame) != 2 * len(rows) or frame.duplicated(["n", "rep", "method"]).any():
        raise ValueError("D3 replay output has missing or duplicate rows.")
    tables = OUTPUT / "汇总表_CSV"
    figures = OUTPUT / "图"
    for folder in (tables, figures):
        folder.mkdir(exist_ok=True)
    frame.to_csv(tables / "D3_Sandwich_replications.csv", index=False, encoding="utf-8-sig")
    summary = summarize(frame)
    summary.to_csv(tables / "D3_Sandwich_summary.csv", index=False, encoding="utf-8-sig")
    save_figure(summary, figures)
    print(f"Verified {len(rows):,} original hashes and wrote {len(frame):,} intervals records "
          f"in {perf_counter() - start:.1f} seconds.")


def verify_pseudovalues():
    from methods import score_grid as original_grid
    w = np.array([-1., 1., 1., -1., 1., -1.])
    residual = np.array([0.1, -1.3, 0.7, -0.1, 0.5, 0.3])
    degrees = (1, 2, 3, 4, 5, 6)
    for values in (residual, np.array([0.1, 0.1, 0.7, -0.1, 0.7, 0.1]), np.ones(len(w))):
        reference = []
        for i in range(len(w)):
            keep = np.arange(len(w)) != i
            reference.append([w @ original_grid(values, d) -
                              w[keep] @ original_grid(values[keep], d) for d in degrees])
        reference = np.asarray(reference)
        np.testing.assert_allclose(fold_pseudovalues(w, values, degrees),
                                   reference - reference.mean(axis=0), atol=1e-12)



if __name__ == "__main__":
    main()

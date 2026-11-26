"""Denominator and pairing checks using synthetic records; no learner training."""
import math
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_pilot import METHODS, diagnostic_counts, paired_records, summarize_records, summarize_run, validate_records, wilson


def row(rep, method, covered, status="ok", scenario="gaussian"):
    result = dict(scenario=scenario, replication=rep, method=method, covered=covered,
                  status=status, config_hash="fixture")
    if method == METHODS[0] and status == "ok":
        result.update(theta_hat=1.0, theta_se=.1, ci_low=.804, ci_high=1.196, ci_length=.392)
    return result


class SummaryTests(unittest.TestCase):
    def test_pending_is_not_failed_and_all_plan_bounds(self):
        config = dict(replications=4, scenarios=["gaussian"])
        records = [row(0, METHODS[1], True), row(1, METHODS[1], None, "failed")]
        b = summarize_records(records, config)[1]
        self.assertEqual((b["attempted"], b["valid"], b["failed"], b["pending"]), (2, 1, 1, 2))
        self.assertEqual(b["coverage_valid"], 1)
        self.assertEqual(b["failure_rate"], .5)
        self.assertEqual((b["planned_coverage_lower"], b["planned_coverage_upper"]), (.25, 1))
        self.assertIsNone(b["bias"])

    def test_pairs_use_common_valid_runs_and_conservative_bounds(self):
        config = dict(replications=4, scenarios=["gaussian"])
        records = [row(0, METHODS[1], False), row(0, METHODS[2], True),
                   row(1, METHODS[1], True), row(1, METHODS[2], False),
                   row(2, METHODS[1], True), row(2, METHODS[2], None, "failed")]
        cb = paired_records(records, config)[0]
        self.assertEqual(cb["valid_pairs"], 2)
        self.assertEqual(cb["mean_difference"], 0)
        self.assertAlmostEqual(cb["mcse"], 1)
        self.assertEqual((cb["count_minus1"], cb["count_zero"], cb["count_plus1"]), (1, 0, 1))
        self.assertEqual((cb["planned_difference_lower"], cb["planned_difference_upper"]), (-.5, .5))
        self.assertEqual(cb["left_failed"], 1)
        self.assertEqual(cb["right_failed"], 0)

    def test_all_failures_cannot_be_coverage(self):
        config = dict(replications=1, scenarios=["gaussian"])
        records = [row(0, METHODS[2], None, "failed")]
        c = summarize_records(records, config)[2]
        self.assertIsNone(c["coverage_valid"])
        self.assertIsNone(c["wilson_low"])
        self.assertEqual(c["failure_rate"], 1)
        self.assertEqual((c["planned_coverage_lower"], c["planned_coverage_upper"]), (0, 1))
        records[0]["covered"] = True
        with self.assertRaises(ValueError):
            summarize_records(records, config)

    def test_duplicate_or_mixed_config_refused(self):
        config = dict(replications=2, scenarios=["gaussian"])
        a = row(0, METHODS[1], True)
        with self.assertRaises(ValueError):
            validate_records([a, a], config)
        b = row(1, METHODS[1], False)
        b["config_hash"] = "other"
        with self.assertRaises(ValueError):
            validate_records([a, b], config)

    def test_bc_estimates_refused(self):
        config = dict(replications=1, scenarios=["gaussian"])
        b = row(0, METHODS[1], True)
        b["theta_hat"] = .1
        with self.assertRaises(ValueError):
            validate_records([b], config)

    def test_wilson_and_mcse_at_nominal(self):
        lo, hi = wilson(190, 200)
        self.assertAlmostEqual(lo, .91042185, places=6)
        self.assertAlmostEqual(hi, .97261735, places=6)
        self.assertAlmostEqual(math.sqrt(.95*.05/200), .015411035, places=8)

    def test_coverage_tail_minus_gaussian_without_fake_pairing(self):
        config = dict(replications=2, scenarios=["gaussian", "t2.1"])
        records = [row(0, METHODS[1], True), row(1, METHODS[1], True),
                   row(0, METHODS[1], True, scenario="t2.1"), row(1, METHODS[1], False, scenario="t2.1")]
        b_tail = summarize_records(records, config)[4]
        self.assertEqual(b_tail["coverage_minus_gaussian"], -.5)

    def test_boundary_side_does_not_duplicate_bandwidth_count(self):
        count = diagnostic_counts([dict(boundary_G=True, boundary_side_G="upper", boundary_M=False, boundary_side_M="", constant_index_G=False, constant_index_M=True)])
        self.assertEqual(count, dict(known=2, boundary=1, constant=1))

    def test_report_artifacts_and_incomplete_note(self):
        with tempfile.TemporaryDirectory(prefix="dukes_summary_test_") as directory:
            run_dir = Path(directory)
            (run_dir / "tasks").mkdir()
            config = dict(replications=2, scenarios=[dict(name=s) for s in ("gaussian", "t2.1", "lognormal2")])
            (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for scenario in ("gaussian", "t2.1", "lognormal2"):
                records = [row(0, method, True, scenario=scenario) for method in METHODS]
                (run_dir / "tasks" / (scenario + "_0000.json")).write_text(json.dumps(dict(rows=records, folds=[])), encoding="utf-8")
            summarize_run(run_dir)
            for artifact in ("raw_results.csv", "fold_diagnostics.csv", "summary.csv", "paired_differences.csv", "failures.csv", "coverage_comparison.png", "decision_note.md"):
                self.assertTrue((run_dir / artifact).exists())
            self.assertIn("筛查尚未完成", (run_dir / "decision_note.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

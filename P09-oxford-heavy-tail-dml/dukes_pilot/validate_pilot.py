"""Audit-gated regression checks; writes the exact-config validation certificate."""
from __future__ import annotations
import run_pilot as runner  # caps inner threads before numerical imports
import json
import time
import unittest
from unittest.mock import patch
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
import legacy_adapter as legacy
from dukes_score import NumericalFailure, _learner


def comparable(result):
    copy = json.loads(json.dumps(runner.clean(result)))
    copy.pop("timings", None)
    copy.pop("elapsed_seconds", None)
    for row in copy["rows"]:
        for key in list(row):
            if key.endswith("seconds"):
                row.pop(key)
    return copy


def main():
    start = time.perf_counter()
    integrity = runner.audit_gate()
    config = runner.fixed_config()
    suite = unittest.defaultTestLoader.discover(str(runner.ROOT / "tests"))
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful() or tested.testsRun < 8:
        raise RuntimeError("Unit validation failed or expected tests are missing")
    records = []
    serial = []
    with threadpool_limits(limits=1):
        for scenario in config["scenarios"]:
            family = "lognormal" if scenario["error_family"] == "lognormal" else "student_t"
            namespace = legacy.original_namespace(family)
            for rep in (0, 1):
                data, folds, seed = runner.make_data(config, scenario, rep)
                reference = namespace["generate_data"](config["n"], scenario["error_case"], config["dgp"], seed, theta0=config["theta0"])
                for left, right in zip(data, reference):
                    np.testing.assert_array_equal(left, right)
                X, D, Y, U = data
                cfseed = seed + config["fold_seed_offset"]
                original_folds = list(namespace["KFold"](5, shuffle=True, random_state=cfseed).split(X))
                for pair, ref_pair in zip(folds, original_folds):
                    for left, right in zip(pair, ref_pair):
                        np.testing.assert_array_equal(left, right)
                dr, yr = namespace["cross_fitted_residuals"](X,D,Y,"mean",5,cfseed)
                expected = namespace["fit_mean_dml"](dr,yr)
                result, arrays = runner.compute_task(config, scenario, rep)
                assert all(row["status"] == "ok" for row in result["rows"]), result["rows"]
                np.testing.assert_allclose(dr, arrays["d_res"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(yr, arrays["y_res"], rtol=1e-12, atol=1e-12)
                row = next(r for r in result["rows"] if r["method"] == "original_mean_dml")
                np.testing.assert_allclose(expected, [row["theta_hat"],row["theta_se"]], rtol=1e-12, atol=1e-12)
                assert legacy.make_nuisance_model("mean",cfseed).get_params() == _learner(cfseed).get_params()
                for diag in result["folds"]:
                    assert abs(diag["mean_G_rD"]) < 1e-10
                    assert abs(diag["mean_M_rY"]) < 1e-10
                records.append(dict(scenario=scenario["name"], replication=rep, seed=seed, theta_A=row["theta_hat"], se_A=row["theta_se"], max_residual_difference=float(max(np.max(np.abs(dr-arrays["d_res"])), np.max(np.abs(yr-arrays["y_res"])))), elapsed_seconds=result["elapsed_seconds"]))
                print(f"Legacy match {scenario['name']} rep={rep}: {result['elapsed_seconds']:.2f}s", flush=True)
                if rep == 0 and len(serial) < 2:
                    serial.append((scenario, result, arrays))
        # Gaussian in the two original notebooks must be identical.
        ns_t, ns_l = legacy.original_namespace("student_t"), legacy.original_namespace("lognormal")
        t = ns_t["generate_data"](1000,"N(0,1)",config["dgp"],20367826)
        l = ns_l["generate_data"](1000,"N(0,1)",config["dgp"],20367826)
        for left, right in zip(t,l):
            np.testing.assert_array_equal(left,right)
        repeat, repeat_arrays = runner.compute_task(config, serial[0][0], 0)
        assert comparable(repeat) == comparable(serial[0][1])
        for key in repeat_arrays:
            np.testing.assert_array_equal(repeat_arrays[key], serial[0][2][key])
        with patch.object(runner, "dukes_from_initial", side_effect=NumericalFailure("injected_validation_failure", "test only", fold=2)):
            failure, _ = runner.compute_task(config, serial[0][0], 0)
        failed = next(r for r in failure["rows"] if r["method"] == "dukes_dr_gcm")
        assert failed["status"] == "failed" and failed["covered"] is None and failed["failure_fold"] == 2
    parallel_start = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        parallel = Parallel(n_jobs=2)(delayed(runner.compute_task)(config, scenario, 0) for scenario, _, _ in serial)
    for (_, before, arrays_before), (after, arrays_after) in zip(serial, parallel):
        assert comparable(before) == comparable(after)
        for key in arrays_before:
            np.testing.assert_array_equal(arrays_before[key], arrays_after[key])
    certificate = dict(status="passed", config_hash=config["config_hash"], unit_tests=tested.testsRun, legacy_checks=records, serial_parallel="bitwise_equal", repeated_task="bitwise_equal", original_gaussian_families="bitwise_equal", failure_recording="passed", source_integrity=runner.audit_gate(), parallel_two_task_wall_seconds=time.perf_counter()-parallel_start, validation_wall_seconds=time.perf_counter()-start, r_author_crosscheck="not_available")
    runner.dump_json(runner.ROOT / "audit/validation.json", certificate)
    text = "# 正确性验证\n\n全部检查通过。配置哈希：`"+config["config_hash"]+"`。\n\n"
    text += f"- 独立数组/统计汇总单元测试：{tested.testsRun} 项通过。\n"
    text += "- 三场景前两次重复：原 notebook 白名单纯函数与新实现的数据、fold 完全一致；OOF 残差、A 估计和 SE 在预定 1e-12 容差内一致。\n"
    text += "- 原两 notebook 的 Gaussian 数据完全相同；仅运行一次 Gaussian。\n- A/B/C 共用处理预测与 folds；B/C 的初始预测与逐点标准 score 完全相同。\n"
    text += "- 更新矩条件绝对值低于 1e-10；完整 score 保留方差修正项。\n- 同次重复重跑和 1/2 进程结果逐位一致。\n- 人工注入 C 第2折失败被记录为 failed、covered=null，未记为覆盖。\n"
    text += "- 作者 R/np 环境不可用，未完成与作者软件的带宽优化交叉核验；Python 公式独立校验通过不能代替该核验。\n- 当前环境的原函数一致性不构成对历史未知软件环境逐位复现的承诺。\n\n固定重复记录见 `validation.json`。验证后所有审计旧文件哈希复核通过。\n"
    (runner.ROOT / "audit/validation.md").write_text(text, encoding="utf-8")
    print(json.dumps(certificate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Mean-DML adapter. Source functions are verbatim in _legacy_*.py.

Only AST-whitelisted pure functions from notebook cells 4/6 were extracted.
No notebook top-level sampling, training, plotting or exporting is executed.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

import _legacy_student as student
import _legacy_lognormal as lognormal

P = student.P
THETA_0 = student.THETA_0
BOOSTING_PARAMS = student.BOOSTING_PARAMS
NORMAL_BASELINE = student.NORMAL_BASELINE
ar1_covariance = student.ar1_covariance
nuisance_functions = student.nuisance_functions
make_nuisance_model = student.make_nuisance_model
cross_fitted_residuals = student.cross_fitted_residuals
fit_mean_dml = student.fit_mean_dml
standardized_student_t = student.standardized_student_t
standardized_centered_lognormal = lognormal.standardized_centered_lognormal


def generate_data(n, error_case, dgp_name, seed, theta0=THETA_0,
                  error_family="student_t"):
    """Dispatch without changing the original random draw order."""
    if error_family == "lognormal":
        module = lognormal
    elif error_family in ("student_t", "gaussian"):
        module = student
        if error_family == "gaussian":
            error_case = NORMAL_BASELINE
    else:
        raise ValueError(f"Unsupported error family: {error_family}")
    return module.generate_data(n, error_case, dgp_name, seed, theta0)


def original_namespace(error_family="student_t", project_root=None):
    """Independently extract audited original definitions for regression tests.

    This deliberately re-reads notebook JSON and compiles only individually
    whitelisted FunctionDef nodes, never whole cells. Constants/imports are
    supplied explicitly. The source SHA-256 must match the pre-run manifest.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    suffix = "lognormal" if error_family == "lognormal" else "student_t"
    path = root / "code" / f"DML_p5_{suffix}_heavytail.ipynb"
    provenance = json.loads((Path(__file__).parent / "audit/extraction_provenance.json").read_text(encoding="utf-8"))
    expected = provenance[suffix]["source_sha256"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"Original notebook changed: {path}")
    notebook = json.loads(raw)
    names = {"ar1_covariance", "nuisance_functions", "generate_data",
             "standardized_student_t", "standardized_centered_lognormal",
             "make_nuisance_model", "cross_fitted_residuals", "fit_mean_dml"}
    namespace = {"np": np, "P": 5, "THETA_0": 1.0,
                 "NORMAL_BASELINE": "N(0,1)",
                 "BOOSTING_PARAMS": dict(BOOSTING_PARAMS),
                 "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
                 "KFold": KFold}
    for cell_index in (4, 6):
        source = "".join(notebook["cells"][cell_index]["source"])
        tree = ast.parse(source)
        defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        selected = ast.Module(body=defs, type_ignores=[])
        exec(compile(selected, f"{path}:cell{cell_index}", "exec"), namespace)
    return namespace

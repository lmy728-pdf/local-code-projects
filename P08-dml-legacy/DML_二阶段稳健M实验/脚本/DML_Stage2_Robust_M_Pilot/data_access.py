"""Read-only reuse of original predictions; reconstruct DGP only for audit."""
from pathlib import Path
import ast, hashlib, json, platform
import numpy as np
import scipy, sklearn, pandas
from legacy_core import make_data, nuisance_functions

ROOT = Path(__file__).resolve().parent
DESKTOP = ROOT.parent
LEGACY = DESKTOP/'DML_Dukes_full_results'/'n1000_R1000_f76477e6a5dc'
ORIGINAL_NB = DESKTOP/'毕设'/'DML_Dukes_n1000_R1000.ipynb'
METHODS = ['A', 'B', 'C', 'D']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, Path): return str(x)
        return x
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def preflight():
    cfg = json.loads((ROOT/'legacy_config.json').read_text(encoding='utf-8'))
    if sha(LEGACY/'config.json') != sha(ROOT/'legacy_config.json'):
        raise RuntimeError('Original configuration changed')
    original = json.loads(ORIGINAL_NB.read_text(encoding='utf-8'))
    src = '\n'.join(''.join(c['source']) for c in original['cells'] if c['cell_type']=='code')
    tree = ast.parse(src)
    defs = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    frozen = ast.parse((ROOT/'legacy_core.py').read_text(encoding='utf-8'))
    checked = []
    for node in frozen.body:
        if isinstance(node, ast.FunctionDef):
            if defs.get(node.name) != ast.dump(node, include_attributes=False):
                raise RuntimeError('Frozen definition mismatch: '+node.name)
            checked.append(node.name)
    for name, actual in [('numpy', np.__version__), ('scipy', scipy.__version__), ('sklearn', sklearn.__version__)]:
        if actual != cfg['environment'][name]: raise RuntimeError('Environment drift: '+name)
    if (cfg['n'], cfg['p'], cfg['n_folds'], cfg['dgp']) != (1000, 5, 5, 'DGP2_nonlinear'):
        raise RuntimeError('Unexpected original design')
    info = dict(original_notebook=str(ORIGINAL_NB), notebook_sha256=sha(ORIGINAL_NB),
                legacy_config_sha256=sha(LEGACY/'config.json'), checked_function_AST=checked,
                frozen_core_sha256=sha(ROOT/'legacy_core.py'), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, pandas=pandas.__version__,
                HGBR_parameters=cfg['hgbr_params_seed_zero'], no_nuisance_training=True)
    return cfg, info


def load_replication(cfg, scene, rep):
    stem = f"{scene['name']}_{rep:04d}"
    taskpath = LEGACY/'tasks'/(stem+'.json')
    task = json.loads(taskpath.read_text(encoding='utf-8'))
    ap = LEGACY/task['arrays_file']
    digest = sha(ap)
    if digest != task['arrays_sha256']: raise RuntimeError('Array hash mismatch: '+stem)
    with np.load(ap, allow_pickle=False) as saved:
        a = {k: saved[k].copy() for k in ['fold_ids', 'g_hat', 'ell_hat', 'd_res', 'y_res']}
    (X, D, Y, eps), folds, seed = make_data(cfg, scene, rep)
    expected_folds = np.full(cfg['n'], -1, dtype=np.int64)
    for k, (_, test) in enumerate(folds): expected_folds[test] = k
    if not np.array_equal(a['fold_ids'], expected_folds): raise RuntimeError('Fold mismatch: '+stem)
    dmax = float(np.max(abs(a['d_res']-(D-a['g_hat']))))
    ymax = float(np.max(abs(a['y_res']-(Y-a['ell_hat']))))
    if max(dmax, ymax) > 1e-12: raise RuntimeError('DGP/residual mismatch: '+stem)
    if not all(np.isfinite(v).all() for v in a.values()): raise RuntimeError('Nonfinite saved array')
    old = next(r for r in task['rows'] if r['method']=='original_mean_dml')
    foldhash = hashlib.sha256(expected_folds.tobytes()).hexdigest()
    assert old['dataset_seed']==seed and old['crossfit_seed']==seed+cfg['fold_seed_offset']
    assert old['fold_hash']==foldhash
    audit = dict(scenario=scene['name'], rep_id=rep, seed=seed, crossfit_seed=seed+cfg['fold_seed_offset'],
                 task_path=str(taskpath), task_sha256=sha(taskpath), array_path=str(ap), array_sha256=digest,
                 fold_hash=foldhash, m_prediction_hash=hashlib.sha256(a['g_hat'].tobytes()).hexdigest(),
                 ell_prediction_hash=hashlib.sha256(a['ell_hat'].tobytes()).hexdigest(),
                 d_res_max_difference=dmax, y_res_max_difference=ymax,
                 original_theta=old['theta_hat'], original_se=old['theta_se'])
    # Truth is exposed ONLY to test_stage2.py diagnostics, never to fit_method.
    m0, g0 = nuisance_functions(X, cfg['dgp'])
    oracle = dict(d=D-m0, y=Y-(cfg['theta0']*m0+g0), epsilon=eps)
    return a['d_res'], a['y_res'], audit, oracle

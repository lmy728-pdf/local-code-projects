"""Check the selected Python environment; install missing packages only."""
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

PACKAGES = {'numpy': 'numpy>=1.24', 'sklearn': 'scikit-learn>=1.3',
            'tqdm': 'tqdm>=4.65', 'threadpoolctl': 'threadpoolctl>=3.1',
            'psutil': 'psutil>=5.9', 'matplotlib': 'matplotlib>=3.7'}

def main():
    if sys.version_info < (3, 10):
        raise SystemExit('Python >= 3.10 required; Python 3.11/3.12 recommended.')
    print('Python:', sys.executable, platform.python_version(), flush=True)
    missing = [spec for module, spec in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    code = '''
import numpy as np, sklearn, tqdm, psutil, matplotlib
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits, threadpool_info
with threadpool_limits(limits=1):
    x = np.arange(40.).reshape(20, 2)
    m = RandomForestRegressor(n_estimators=1, n_jobs=1, random_state=0).fit(x, x[:, 0])
    assert np.isfinite(m.predict(x)).all()
    assert all(p['num_threads'] == 1 for p in threadpool_info())
    print('Environment check OK; numerical thread pools limited to 1.')
    print('NumPy:', np.__version__, 'scikit-learn:', sklearn.__version__)
'''
    subprocess.check_call([sys.executable, '-c', code], env=env)
    snapshot = Path(__file__).resolve().parent / 'environment_snapshot'
    snapshot.mkdir(exist_ok=True)
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (snapshot / 'pip-freeze.txt').write_text(frozen, encoding='utf-8')
    names = ['numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl', 'tqdm', 'psutil', 'matplotlib']
    locked = '\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in names) + '\n'
    (snapshot / 'benchmark-versions.txt').write_text(locked, encoding='utf-8')
    (snapshot / 'environment.json').write_text(json.dumps(dict(python=sys.version,
        executable=sys.executable, platform=platform.platform()), indent=2), encoding='utf-8')
    print('Saved environment snapshot:', snapshot)
    print('For cross-platform comparisons, match Python and benchmark package versions.')

if __name__ == '__main__':
    main()

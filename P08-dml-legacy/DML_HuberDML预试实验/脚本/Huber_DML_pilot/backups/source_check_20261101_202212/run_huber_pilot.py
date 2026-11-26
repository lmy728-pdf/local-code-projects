"""Explicit entry point. Nothing fits at import time; default stage is smoke only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json,pathlib,platform,uuid
from datetime import datetime
import numpy as np
import scipy,sklearn,joblib,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,GradientBoostingRegressor
from joblib import Parallel,delayed
from huber_methods import run_replication,GBR_COMMON
from summarize_pilot import summarize
ROOT=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description='No simulation runs unless this entry point is explicitly invoked.')
    parser.add_argument('--stage',choices=['smoke','pilot'],default='smoke')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=16:parser.error('workers must be between 1 and 16')
    config=json.loads((ROOT/'legacy_config.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    # Refuse silent environment drift. No dependency upgrade is performed.
    for name,actual in [('numpy',np.__version__),('scipy',scipy.__version__),('sklearn',sklearn.__version__),('joblib',joblib.__version__)]:
        if actual!=config['environment'][name]:raise RuntimeError(f'{name} version differs: {actual}; expected {config["environment"][name]}')
    for path,digest in manifest['files_sha256'].items():
        if sha(pathlib.Path(path))!=digest:raise RuntimeError('Source changed since packaging: '+path)
    expected=config['hgbr_params_seed_zero']
    actual=HistGradientBoostingRegressor(loss='squared_error',random_state=0,learning_rate=.05,max_iter=100,
                max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0).get_params()
    if actual!=expected:raise RuntimeError('Original HGBR default parameters have changed')
    b=GradientBoostingRegressor(loss='squared_error',alpha=.9,**GBR_COMMON).get_params()
    c=GradientBoostingRegressor(loss='huber',alpha=.9,**GBR_COMMON).get_params()
    assert {k for k in b if b[k]!=c[k]}=={'loss'}
    output=ROOT/'outputs'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+args.stage+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    for folder in ['tasks','arrays']:(output/folder).mkdir()
    config.update(stage=args.stage,replications=2 if args.stage=='smoke' else 100,workers=args.workers,
                  gbr_B=b,gbr_C=c,run_id=output.name,python=platform.python_version(),
                  source_hashes={p.name:sha(p) for p in ROOT.glob('*.py')},
                  seed_rule='base_seed + 100000*legacy_dgp_id + 1000*legacy_error_id + rep_id',
                  note='New A/B/C definitions; no Dukes correction. Smoke repetitions reused within pilot.')
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    def batch(reps):
        jobs=[(s,r) for s in config['scenarios'] for r in reps]
        result=Parallel(n_jobs=args.workers,backend='loky')(
            delayed(run_replication)(config,s,r,output,manifest['legacy_run']) for s,r in jobs)
        for group in result:rows.extend(group)
        pd.DataFrame(rows).to_csv(output/'replications.csv',index=False)
    batch(range(2))
    # Strict smoke gate. Failures are retained; never replace seeds or omit them silently.
    if any(r['status']!='ok' for r in rows):
        summarize(output)
        raise RuntimeError(f'Smoke failed; pilot not launched. See {output}')
    if args.stage=='pilot':batch(range(2,100))
    summarize(output)
    print('Saved:',output)
if __name__=='__main__':main()

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runner
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def tiny_fit():
    runner.configure('g5_rf')
    cell=runner.cells_for('G5')[1][0]
    runner.CONTEXT.update(cell=cell,rep=0);runner.FORESTS.clear()
    x=np.arange(400,dtype=float).reshape(40,10)/100
    y=np.column_stack([x[:,0],x[:,1]**2])
    first=runner.Forest().fit(x,y);pred=first.predict(x)
    assert pred.shape==(40,2) and np.isfinite(pred).all()
    assert len(runner.FORESTS)==1 and len(set(runner.FORESTS[0]['target_seeds']))==2
    for model in first.models:
        assert model.n_estimators==500 and model.n_jobs==1 and model.min_samples_leaf==10
    seed1=runner.FORESTS[0]['target_seeds'];runner.FORESTS.clear()
    second=runner.Forest().fit(x,y)
    assert seed1==runner.FORESTS[0]['target_seeds']
    assert np.array_equal(pred,second.predict(x))
    for stream in ('data','outer','inner_0','diagnostic'):
        assert np.array_equal(runner.mapped_rng('g5_rf',cell,0,stream).normal(size=8),runner.BASE_RNG(runner.baseline(cell),cell,0,stream).normal(size=8))
    return 'PASS: spawned RF fit/predict, parameters, deterministic target seeds, shared baseline streams; no study replications.'

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=1) as pool:print(pool.submit(tiny_fit).result())

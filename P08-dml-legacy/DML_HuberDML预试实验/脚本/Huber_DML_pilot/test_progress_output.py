"""Validate scheduling with a stub and summarize existing smoke data. No model fits."""
import json,pathlib,tempfile
from unittest.mock import patch
import pandas as pd
import numpy as np
import nbformat
import training_progress as tp
ROOT=pathlib.Path(__file__).resolve().parent
smoke=ROOT/'outputs'/'20261101_202557_smoke_9ef616ec'
rows=pd.read_csv(smoke/'replications.csv').to_dict('records')
config=json.loads((smoke/'config.json').read_text(encoding='utf-8'))
old=pd.read_csv(smoke/'summary.csv').set_index(['scenario','method'])
new=tp.compact_summary(rows,config).set_index(['scenario','method'])
for field in ['coverage','bias','mean_se','mean_T','sd_T','outcome_rmse','outcome_mae','treatment_rmse']:
    np.testing.assert_allclose(new[field],old.loc[new.index,field],atol=1e-12)
class QuietProgress:
    def __init__(self,**kw):self.count=0
    def update(self,n):self.count+=n
    def set_postfix(self,**kw):pass
    def write(self,*a):pass
    def close(self):pass
class FakeParallel:
    def __init__(self,**kw):assert kw['return_as']=='generator_unordered'
    def __call__(self,jobs):
        # Reverse completion order tests that cumulative counts do not rely on submission order.
        for func,args,kwargs in reversed(list(jobs)):yield func(*args,**kwargs)
seen=[];checkpoints=[]
def fake_worker(config,scenario,rep,*rest):
    seen.append((scenario['name'],rep))
    return [dict(scenario=scenario['name'],rep_id=rep,method=m,status='ok') for m in 'ABC']
def fake_publish(rows,config,output,desktop,label,show):
    checkpoints.append((label,len(rows)))
    counts=pd.DataFrame(rows).groupby(['scenario','method']).size()
    assert counts.nunique()==1
cfg=dict(config,replications=200,stage='pilot')
with tempfile.TemporaryDirectory(prefix='huber_progress_check_') as folder:
    out=pathlib.Path(folder)/'run';desk=pathlib.Path(folder)/'desktop';out.mkdir();desk.mkdir()
    with patch.object(tp,'Parallel',FakeParallel),patch.object(tp,'tqdm',QuietProgress),patch.object(tp,'publish',fake_publish),patch.object(tp,'compact_summary',lambda r,c:len(r)):
        assert tp.train_with_progress(cfg,out,'unused',fake_worker,desk,show=False)==1800
    assert len(seen)==len(set(seen))==600
    assert checkpoints==[(f'{n}/200 per scenario',n*9) for n in range(20,201,20)]
    tp.publish(rows,config,out,desk,'verification of existing smoke only',show=False)
    assert {p.name for p in desk.iterdir()}=={'Huber_DML_summary.csv','Huber_DML_coverage.png'}
    assert len(pd.read_csv(desk/'Huber_DML_summary.csv'))==9
    assert (desk/'Huber_DML_coverage.png').stat().st_size>1000
for path in [ROOT.parent/'Huber_DML_pilot.ipynb',ROOT.parent/'毕设'/'Huber_DML_pilot.ipynb']:
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    ns={}
    for i in [1,3,5,7]:exec(compile(nb.cells[i].source,f'cell{i}','exec'),ns)
    exec(nb.cells[9].source.split('output=ROOT/')[0],ns)
    assert ns['PILOT_REPLICATIONS']==200 and ns['HUBER_QUANTILE']==.90
print('PASS: 600 unique jobs; checkpoints 20..200; existing-smoke summaries unchanged; exactly two exports; both notebook preflights. Model fits: 0.')
(ROOT/'progress_validation.json').write_text(json.dumps(dict(unique_stub_jobs=600,method_rows=1800,
    cumulative_checkpoints=list(range(20,201,20)),export_files=['Huber_DML_summary.csv','Huber_DML_coverage.png'],
    checked_existing_smoke=True,notebook_preflights=2,model_fits=0),indent=2),encoding='utf-8')

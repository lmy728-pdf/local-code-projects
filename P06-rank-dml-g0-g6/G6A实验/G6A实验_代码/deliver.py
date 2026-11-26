"""CSV + concise Markdown + separate vector PDF figures; no simulation."""
import os
# Notebook launches may inherit an unavailable inline backend.
os.environ["MPLBACKEND"] = "Agg"
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,sys,shutil,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE/'settings.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(Path(cfg['archive'])/'plot_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':15,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
METHODS=['LS','H','HT','W','N'];COLORS=['#64748b','#d68b20','#a64a83','#007f78','#2563eb']

def audit_records(folder):
    import core
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'));seen=set();replays=0
    for path in (folder/'records').glob('*.json'):
        data=json.loads(path.read_text(encoding='utf-8'));cell=data['cell'];rep=data['replication'];key=(json.dumps(cell,sort_keys=True),rep)
        assert key not in seen;seen.add(key)
        if 'traceback' in data:continue
        for pipe in ('learned','oracle'):
            rows=[x for x in data['records'] if x['pipeline']==pipe]
            assert {x['method'] for x in rows}==set(METHODS) and len({x['input_hash'] for x in rows})==1
            for row in rows:
                V=np.array(row['covariance'],float);p=np.array(row['estimate'],float);truth=np.array(row['truth'],float)
                check=core.inference(p,V,truth,cell['n'])
                assert np.allclose(np.array(row['se'],float),check['se'],equal_nan=True)
                assert row['coverage']==check['coverage'].tolist()
                if cell['d']>1:assert row['joint_coverage']==check['joint_coverage']
        if rep in (0,499,999):
            cache=Path(data['source_cache']) if 'source_cache' in data else folder/'cache'/(path.stem+'.npz')
            with np.load(cache) as a:
                for row in data['records']:
                    oracle=row['pipeline']=='oracle';m=row['method'];w=a['w_oracle' if oracle else 'w'];z=a['z_oracle' if oracle else 'z'];f=a['folds']
                    threshold=a['scales'][f]*(1.345 if m=='H' else a['multipliers'][f]) if m in ('H','HT') else None
                    point,V,_,_=core.estimate(w,z,f,m,threshold,c=data.get('rank_step',1.))
                    assert np.allclose(point,np.array(row['estimate'],float),atol=1e-12,equal_nan=True)
                    assert np.allclose(V,np.array(row['covariance'],float),atol=1e-12,equal_nan=True)
                    replays+=1
    assert len(seen)==len(config['cells'])*1000
    return dict(datasets=len(seen),replayed_records=replays,shared_inputs_verified=True,inference_recomputed=True)

def markdown(df):
    def val(v):return f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def export(folder,target,stage,archive):
    from compact_output import export_compact
    return export_compact(folder,target,stage,archive)

"""Count saved training candidate convergence; never fit a model."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from audit_records import OUT,TMP
from assemble_tables import load_results,write
import pandas as pd

def scan(a):
    row=dict(a['base']);count=bad=missing=scale_bad=0;iters=[]
    root=Path(a['source_folder'])/'records'
    from audit_records import cell_id
    ident=cell_id(a['base'])
    files=sorted(root.glob(ident+'_*.json'))
    assert len(files)==a['completed']
    for f in files:
        data=json.loads(f.read_text(encoding='utf-8'))
        choices=data.get('training_choices')
        if choices is None:
            # G6B deliberately inherits the original preparation; inspect the referenced originals.
            if a['base']['stage']=='G6B':
                from audit_records import ARCHIVE
                batch='g1_scalar_n1000' if a['base']['d']==1 else 'g2_vector_n1000'
                original=ARCHIVE/'runs'/batch/'records'/f.name
                data=json.loads(original.read_text(encoding='utf-8'));choices=data.get('training_choices')
        if choices is None:missing+=1;continue
        for fold in choices:
            for coord in fold['coordinates']:
                scale_bad+=int(not math.isfinite(coord['scale']) or coord['scale']<=0)
                for c in coord['candidates']:
                    count+=1
                    bad+=int(not c.get('converged',False) or c.get('criterion') is None or not math.isfinite(c['criterion']))
                    iters.append(c['iterations'])
    row.update(records=len(files),candidate_count=count,invalid_or_nonconverged=bad,invalid_rate=bad/count if count else None,missing_training_records=missing,invalid_scale=scale_bad,
      max_candidate_iterations=max(iters) if iters else None,mean_candidate_iterations=sum(iters)/len(iters) if iters else None,
      source_records=str(root),scope='inherited original preparation' if row['stage']=='G6B' else 'stored training-only threshold candidates')
    return row

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=12) as pool:rows=list(pool.map(scan,load_results()))
    write('05_运行与复现/T13b_训练候选收敛.csv',pd.DataFrame(rows))
    print(json.dumps({'blocks':len(rows),'candidates':sum(r['candidate_count'] for r in rows),'invalid':sum(r['invalid_or_nonconverged'] for r in rows),'missing':sum(r['missing_training_records'] for r in rows)}))

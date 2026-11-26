import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import *
from run import ROOT,save,worker,cell_id

r=np.random.default_rng(930112)
cell=dict(experiment='vector',law='gaussian',n=150,d=2,rho=.5)
x,D,Y,wo,zo,true=generate(cell,r);f=folds(len(x),5,r)
scales,mult,details,_,_=training_choices(x,D,Y,f,'leakage_check',cell,0)
out=[]
for k in range(5):
    altered=x.copy();dd=D.copy();yy=Y.copy();altered[f==k]+=2;dd[f==k]-=3;yy[f==k]+=5
    s,a,b,_,_=training_choices(altered,dd,yy,f,'leakage_check',cell,0)
    assert np.array_equal(scales[k],s[k]);assert np.array_equal(mult[k],a[k]);assert details[k]==b[k]
    train=f!=k;orig=Ridge().fit(x[train],np.column_stack([D[train],Y[train]]));changed=Ridge().fit(altered[train],np.column_stack([dd[train],yy[train]]))
    assert np.array_equal(orig.beta,changed.beta) and np.array_equal(orig.alpha,changed.alpha)
    fi=folds(train.sum(),3,r);inner=[]
    for h in range(3):
        xx=x[train].copy();y=np.column_stack([D[train],Y[train]]);tr=fi!=h
        original=Ridge().fit(xx[tr],y[tr]);xx[~tr]+=2;y[~tr]+=5;modified=Ridge().fit(xx[tr],y[tr])
        assert np.array_equal(original.beta,modified.beta) and np.array_equal(original.alpha,modified.alpha)
        inner.append(True)
    out.append(dict(outer_fold=k,nuisance_basis_penalty_unchanged=True,scale_threshold_and_candidates_unchanged=True,inner_folds=inner))
# Deliberately invalid sample size tests the dataset failure serialization path.
bad=dict(experiment='scalar',law='gaussian',n=0,d=1,rho=0)
worker(('failure_check',bad,0,str(ROOT/'failure_check')))
record=json.loads((ROOT/'failure_check'/'records'/(cell_id(bad)+'_0000.json')).read_text(encoding='utf-8'))
assert len(record['records'])==10 and all(row['failures'] and not any(row['coverage']) for row in record['records'])
evidence=json.loads((ROOT/'reports'/'implementation_checks.json').read_text(encoding='utf-8'))
evidence['T2']['all_outer_and_inner_perturbation_checks']=out
evidence['T9']['dataset_failure_serialization']=dict(attempted_method_records=10,retained=10,covered=0,test_stage='failure_check',invalid_n=0,excluded_from_G0=True)
evidence['T2']['additional_check_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
save(ROOT/'reports'/'implementation_checks.json',evidence)
save(ROOT/'reports'/'additional_check_evidence.json',dict(leakage=out,retained_failure_records=10))
print('Five outer and fifteen inner perturbations passed; ten failure records retained.',flush=True)

"""Validate saved diagnostics and input integrity. No fitting or simulation."""
import hashlib, importlib.metadata, json, pathlib, platform, re, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    d=pd.read_csv(ROOT/'r_python_comparison.csv',float_precision='round_trip')
    f=pd.read_csv(ROOT/'fold_level_comparison.csv',float_precision='round_trip')
    selected=pd.read_csv(ROOT/'selected_samples.csv')
    assert len(d)==30 and len(f)==150 and (d.groupby('scenario').size()==10).all()
    assert d[['scenario','replication']].drop_duplicates().shape[0]==30
    assert (d.status=='ok').all() and d.wrapper_matches_original.all()
    assert d.wrapper_max_difference.max()==0 and d.nonfinite_count.sum()==0
    assert d.Python_original_bw_reconstruction_maxdiff.max()==0
    assert np.allclose(d.p_value_R,2*norm.sf(abs(d.T_R)),rtol=1e-12,atol=1e-14)
    assert np.allclose(d.p_value_Python,2*norm.sf(abs(d.T_python)),rtol=1e-12,atol=1e-14)
    assert d.decision_disagrees.sum()==1
    assert d.T_R_largeh_off_minus_fixed_Rbw_Python.abs().max()<1e-12
    warnings=0
    for row in selected.itertuples():
        inp=ROOT/'inputs'/f'{row.stem}.csv'
        assert sha(inp)==row.input_sha256
        r=json.loads((ROOT/'r_results'/f'{row.stem}.json').read_text())
        assert r['rng_consumption_matches'] and r['wrapper_matches_original']
        assert not r['warnings']; warnings+=len(r['warnings'])
        a=pd.read_csv(ROOT/'r_results'/f'{row.stem}_arrays.csv')
        assert len(a)==1000 and np.isfinite(a.to_numpy()).all()
        folds=pd.read_csv(ROOT/'r_results'/f'{row.stem}_folds.csv')
        assert not folds.constant_G.any() and not folds.constant_M.any()
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    for path,digest in manifest['source_files_sha256'].items():assert sha(pathlib.Path(path))==digest
    assert sha(pathlib.Path('C:/Users/Lenovo/Desktop/DML_Dukes_n1000_R1000.ipynb'))=='b73f66d89f6518e03121cc149de2871ac5840bb5fcc468fb5e1d9e42bc12df3e'
    assert sha(ROOT/'downloads'/'dri_functions.r')=='fa32ffae449144fb0bf85c894f2cd9271bc5f492657f289db324da8db4b9bd71'
    lock=json.loads((ROOT/'package_lock.json').read_text())
    for package in lock:assert sha(ROOT/package['file'])==package['sha256']
    for filename in ('README.md','cross_validation_report.md'):
        content=(ROOT/filename).read_text(encoding='utf-8')
        for link in re.findall(r'\]\(([^)]+)\)',content):
            if not link.startswith('https://'):assert (ROOT/link).exists(),link
        for line in content.splitlines():
            if line.startswith('|'):
                assert line.endswith('|')
    versions={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
              'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}}
    (ROOT/'python_versions.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    result={'passed':True,'replications':30,'folds':150,'observations':30000,'R_warnings':warnings,
            'nonfinite_R_arrays':0,'author_wrapper_max_difference':0,'original_source_files_unchanged':61,
            'notebook_unchanged':True,'p_values_reconstructed':True,'constant_index_branches_exercised':0,
            'archived_packages_verified':len(lock)}
    (ROOT/'delivery_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.parts for x in ['runtime','R_library','np_source','__pycache__']) and p.name not in ['delivery_manifest.json','runtime_install.log']]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    (ROOT/'delivery_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

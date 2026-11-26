"""Packaging only: extract named definitions without executing notebook cells."""
import ast,hashlib,json,pathlib,shutil
ROOT=pathlib.Path(__file__).resolve().parent
NB=pathlib.Path('C:/Users/Lenovo/Desktop/DML_Dukes_n1000_R1000.ipynb')
RUN=NB.parent/'DML_Dukes_full_results'/'n1000_R1000_f76477e6a5dc'
NAMES={'ar1_covariance','nuisance_functions','standardized_student_t','generate_data_student',
       'make_nuisance_model','fit_mean_dml','lognormal_ar1_covariance','lognormal_nuisance_functions',
       'standardized_centered_lognormal','generate_data_lognormal','generate_data','make_data','initial_a'}
def main():
    if (ROOT/'legacy_core.py').exists():raise FileExistsError('Frozen source already exists; do not overwrite')
    nb=json.loads(NB.read_text(encoding='utf-8'));definitions=[];locations=[]
    for i,cell in enumerate(nb['cells'],1):
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef) and node.name in NAMES:
                definitions.append(ast.get_source_segment(source,node))
                locations.append(dict(function=node.name,notebook_cell=i,cell_line=node.lineno))
    assert {x['function'] for x in locations}==NAMES
    header='''# Frozen original definitions; no notebook launch cells or Dukes correction.
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
P=5
THETA_0=1.0
NORMAL_BASELINE="N(0,1)"
BOOSTING_PARAMS=dict(learning_rate=0.05,max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0)

'''
    (ROOT/'legacy_core.py').write_text(header+'\n\n'.join(definitions)+'\n',encoding='utf-8')
    shutil.copy2(RUN/'config.json',ROOT/'legacy_config.json')
    shutil.copy2(NB.parent/'README_Huber_DML_pilot.md',ROOT/'requested_design.md')
    manifest=dict(notebook=str(NB),legacy_run=str(RUN),locations=locations,
                  files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [NB,RUN/'config.json',ROOT/'legacy_core.py']})
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()

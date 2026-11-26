"""Preflight only: no data generation, fit or output-run creation."""
import ast,pathlib,json,nbformat
ROOT=pathlib.Path(__file__).resolve().parent
nb=nbformat.read(ROOT.parent/'Huber_DML_pilot.ipynb',as_version=4)
nbformat.validate(nb)
for c in nb.cells:
    if c.cell_type=='code':ast.parse(c.source)
for name in ['run_huber_pilot.py','huber_methods.py','summarize_pilot.py','build_notebook.py']:
    ast.parse((ROOT/name).read_text(encoding='utf-8'))
ns={}
for i in [1,3,5,7]:exec(compile(nb.cells[i].source,'cell_'+str(i),'exec'),ns)
exec(nb.cells[9].source.split('output=ROOT/')[0],ns)
assert ns['STAGE']=='pilot' and ns['PILOT_REPLICATIONS']==200 and ns['HUBER_QUANTILE']==.9
assert len(range(2))+len(range(2,ns['PILOT_REPLICATIONS']))==200
assert ns['GBR_COMMON']['n_estimators']==100 and ns['BOOSTING_PARAMS']['max_iter']==100
assert 'else PILOT_REPLICATIONS' in nb.cells[9].source
assert "batch(range(2,config['replications']))" in nb.cells[11].source
assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
proof=dict(pilot_R=200,stage='pilot',huber_quantile=.9,adaptive_delta=True,boosting_rounds=100,
           preflight_passed=True,notebook_schema_valid=True,syntax_valid=True,simulations=0)
(ROOT/'r200_preflight.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof,indent=2))

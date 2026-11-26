"""Check startup dependencies without running an experiment."""
import os,sys,json,subprocess,importlib,io
from pathlib import Path
HERE=Path(__file__).resolve().parent
SETTINGS=json.loads((HERE/'settings.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(HERE/'dependencies'))
PACKAGES=['scikit-learn==1.9.1','joblib==1.6.0','threadpoolctl==3.8.0','cloudpickle==3.1.2','narwhals==2.28.0']

def install_if_missing():
    if SETTINGS['stage']!='G5':return
    try:
        import sklearn,joblib,threadpoolctl,cloudpickle,narwhals
        return
    except ImportError:pass
    cmd=[sys.executable,'-m','pip','install','--index-url','https://pypi.org/simple','--timeout','20','--retries','1','--only-binary=:all:','--no-deps','--target',str(HERE/'dependencies'),*PACKAGES]
    # Retry a failed proxy connection directly, without changing system proxy settings.
    attempt=subprocess.run(cmd)
    if attempt.returncode:
        env=os.environ.copy()
        for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
        env['NO_PROXY']='*';env['no_proxy']='*'
        subprocess.run(cmd,env=env,check=True)
    importlib.invalidate_caches()

def check():
    os.environ['MPLBACKEND']='Agg'
    import numpy,scipy,pandas
    versions=dict(numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__)
    if SETTINGS['stage']=='G5':
        try:
            import sklearn,joblib,threadpoolctl,cloudpickle,narwhals
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as ex:
            raise RuntimeError('G5 dependency unavailable: run the notebook environment preparation cell. Details: '+str(ex)) from ex
        versions.update(sklearn=sklearn.__version__,joblib=joblib.__version__,threadpoolctl=threadpoolctl.__version__,cloudpickle=cloudpickle.__version__,narwhals=narwhals.__version__)
        model=RandomForestRegressor(n_estimators=500,criterion='squared_error',max_features=1.0,min_samples_leaf=10,bootstrap=True,max_depth=None,n_jobs=1)
        assert model.n_jobs==1
    import deliver
    from matplotlib.figure import Figure
    stream=io.BytesIO();fig=Figure(figsize=(2,2));fig.subplots().plot([0,1],[0,1]);fig.savefig(stream,format='pdf')
    assert stream.getvalue().startswith(b'%PDF-')
    archive=Path(SETTINGS['archive'])
    for stage in ('g1_scalar_n1000','g2_vector_n1000','g3_n500'):
        assert (archive/'runs'/stage/'summary/coordinates.csv').is_file(),'Missing baseline summary: '+stage
    if SETTINGS['stage'] in ('G5','G6B'):
        for stage in ('g1_scalar_n1000','g2_vector_n1000'):
            assert (archive/'runs'/stage/'records').is_dir() and (archive/'runs'/stage/'cache').is_dir(),'Missing baseline records/cache: '+stage
    print(SETTINGS['stage']+' startup checks passed: '+json.dumps(versions)+'. PDF backend=Agg; no experiment executed.',flush=True)
    return versions

if __name__=='__main__':
    if '--install' in sys.argv:install_if_missing()
    check()

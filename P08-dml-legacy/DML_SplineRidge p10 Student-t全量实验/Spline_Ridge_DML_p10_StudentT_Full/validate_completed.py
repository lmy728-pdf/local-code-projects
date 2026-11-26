# -*- coding: utf-8 -*-
"""只读核对本次正式结果；不拟合模型、不修改任何既有实验。"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import json
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import nbformat

ROOT=Path(__file__).resolve().parent

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            h.update(block)
    return h.hexdigest()

def close(a,b,label):
    assert np.allclose(a,b,rtol=1e-10,atol=1e-11,equal_nan=True),label

def validate():
    cfg=json.loads((ROOT/'configuration.json').read_text(encoding='utf-8'))
    done=json.loads((ROOT/'completed.json').read_text(encoding='utf-8'))
    assert cfg['p']==10 and cfg['n']==1000 and cfg['R']==1000
    assert cfg['errors']==list(range(8)) and cfg['workers']==8 and cfg['threads']==1
    assert cfg['config_hash']==done['config_hash']
    assert sha_file(ROOT/'spline_p10_full_worker.py')==cfg['worker_sha256']
    assert done['data_n']==16000
    raw=pd.read_csv(ROOT/'full_replication_results.csv')
    summary=pd.read_csv(ROOT/'full_summary.csv')
    pairs=pd.read_csv(ROOT/'full_paired_comparisons.csv')
    nuisance=pd.read_csv(ROOT/'full_nuisance_diagnostics.csv')
    fails=pd.read_csv(ROOT/'full_failures.csv')
    dimensions=pd.read_csv(ROOT/'p10_vs_p5_comparison.csv')
    assert len(raw)==64000 and len(summary)==len(nuisance)==len(dimensions)==64 and len(pairs)==48
    keys=['dgp_id','error_id','replication_id']
    assert not raw.duplicated(keys+['method']).any()
    groups=raw.groupby(keys)
    assert len(groups)==16000 and groups.size().eq(4).all()
    assert groups.data_hash.nunique().eq(1).all() and groups.fold_hash.nunique().eq(1).all()
    assert raw.data_hash.nunique()==16000
    assert set(raw.method)=={'M1','M2','M3','S'}
    assert raw.point_status.eq('ok').equals(raw.theta_hat.notna())
    assert raw.test_status.eq('ok').equals(raw['T'].notna())
    close(raw.estimation_error,raw.theta_hat-1,'estimation_error')
    has_score=raw['T'].notna()
    close(raw.loc[has_score,'T'],np.sqrt(1000)*raw.loc[has_score,'score_mean']/raw.loc[has_score,'score_sd'],'score standardization')
    assert raw.loc[has_score,'score_ddof'].eq(1).all()
    good=raw.delta_product_net.notna()
    close(raw.loc[good,'delta_product_net'],
          raw.loc[good,'delta_m_delta_ell']-raw.loc[good,'delta_m2'],'net product identity')
    for (d,e,m),part in raw.groupby(['dgp_id','error_id','method']):
        assert len(part)==1000 and set(part.replication_id)==set(range(1000))
        row=summary[(summary.dgp_id==d)&(summary.error_id==e)&(summary.method==m)].iloc[0]
        T=part['T'].dropna();err=part.estimation_error.dropna()
        expected=dict(point_n=len(err),test_n=len(T),point_fail=1000-len(err),test_fail=1000-len(T),
            accept_n=int((T.abs()<=1.96).sum()),coverage_pct=100*(T.abs()<=1.96).mean(),
            bias=err.mean(),rmse=np.sqrt((err**2).mean()),mean_T=T.mean(),sd_T=T.std(ddof=1),
            left_reject_pct=100*(T<-1.96).mean(),right_reject_pct=100*(T>1.96).mean(),
            y_nuisance_mse=part.y_nuisance_mse.mean(),d_nuisance_mse=part.d_nuisance_mse.mean())
        for k,v in expected.items():close(row[k],v,(d,e,m,k))
        gaussian=summary[(summary.dgp_id==d)&(summary.error_id==0)&(summary.method==m)].iloc[0].coverage_pct
        close(row.deterioration_pp,gaussian-row.coverage_pct,(d,e,m,'deterioration'))
        diag=nuisance[(nuisance.dgp_id==d)&(nuisance.error_id==e)&(nuisance.method==m)].iloc[0]
        for metric in ('delta_m_delta_ell','delta_m2','delta_product_net'):
            vals=part[metric].dropna()
            close(diag[metric+'_mean'],vals.mean(),(d,e,m,metric,'mean'))
            close(diag[metric+'_sd'],vals.std(ddof=1),(d,e,m,metric,'sd'))
            for q in (10,25,50,75,90):
                close(diag[f'{metric}_q{q}'],vals.quantile(q/100),(d,e,m,metric,q))
    for _,row in pairs.iterrows():
        part=raw[(raw.dgp_id==row.dgp_id)&(raw.error_id==row.error_id)].set_index(['method','replication_id'])
        combined=part.loc['S'][['T','estimation_error']].join(
            part.loc[row.reference][['T','estimation_error']],lsuffix='_S',rsuffix='_ref')
        T=combined[['T_S','T_ref']].dropna()
        point=combined[['estimation_error_S','estimation_error_ref']].dropna()
        dif=100*((T.T_S.abs()<=1.96).astype(float)-(T.T_ref.abs()<=1.96).astype(float))
        rs,rb=np.sqrt((point**2).mean())
        close(row.coverage_diff_pp,dif.mean(),'pair coverage')
        close(row.coverage_diff_MCSE_pp,dif.std(ddof=1)/np.sqrt(len(dif)),'paired MCSE')
        close(row.bias_diff,(point.estimation_error_S-point.estimation_error_ref).mean(),'paired bias')
        close(row.rmse_ratio,rs/rb,'paired RMSE ratio')
        assert row.test_pairs==len(T) and row.point_pairs==len(point)
    p5root=ROOT.parent.parent/'DML_SplineRidge Student-t全量实验'/'脚本'/'Spline_Ridge_DML_Full_Results'
    p5=pd.read_csv(p5root/'full_summary.csv').set_index(['dgp_id','error_id','method'])
    now=summary.set_index(['dgp_id','error_id','method'])
    for _,row in dimensions.iterrows():
        key=(row.dgp_id,row.error_id,row.method)
        for metric in ('coverage_pct','bias','rmse','mean_T','sd_T','deterioration_pp','y_nuisance_mse','d_nuisance_mse'):
            close(row[metric+'_p10_minus_p5'],now.loc[key,metric]-p5.loc[key,metric],('dimensions',key,metric))
    manifest=json.loads((ROOT/'p5_source_manifest.json').read_text(encoding='utf-8'))
    for path,item in manifest.items():
        assert sha_file(path)==item['sha256'],('p5 changed',path)
    db=sqlite3.connect((ROOT/'checkpoints'/'replications.sqlite3').as_uri()+'?mode=ro',uri=True)
    assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    assert db.execute('SELECT COUNT(*) FROM packets').fetchone()[0]==16000
    attempt_count=db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    db.close()
    assert summary.point_n.sum()==done['point_n'] and summary.test_n.sum()==done['test_n']
    assert len(fails)==done['failure_events']
    for _,r in raw.loc[~raw.point_status.eq('ok')|~raw.test_status.eq('ok')].iterrows():
        evidence=fails[(fails.dgp_id==r.dgp_id)&(fails.error_id==r.error_id)&
                       (fails.replication_id==r.replication_id)&(fails.method==r.method)]
        assert len(evidence)>0,'unreported failure'
    images={}
    for filename in ('coverage_student_t.png','deterioration_student_t.png','bias_rmse_student_t.png',
                     'score_diagnostics.png','nuisance_diagnostics.png','p10_vs_p5.png'):
        with Image.open(ROOT/filename) as im:
            images[filename]=list(im.size);im.verify()
    notebook=ROOT.parent/'DML_SplineRidge p10 Student-t全量实验_代码.ipynb'
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None,'Notebook 尚未执行完'
            assert not any(x.output_type=='error' for x in cell.get('outputs',[]))
    report=dict(passed=True,datasets=16000,method_rows=len(raw),summary_rows=len(summary),
        paired_rows=len(pairs),p10_vs_p5_rows=len(dimensions),point_n=int(summary.point_n.sum()),
        test_n=int(summary.test_n.sum()),failure_events=len(fails),attempts=attempt_count,
        p5_original_files_unchanged=len(manifest),sqlite_integrity='ok',images=images,
        notebook_executed=True,independent_recalculation=True)
    temporary=ROOT/'final_validation.json.tmp'
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,ROOT/'final_validation.json')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    import sys
    if '--wait' in sys.argv:
        # 等待本次 Notebook 进程退出后独立核对，断网也可完成。
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        launcher=json.loads((ROOT/'launch.json').read_text(encoding='utf-8-sig'))
        handle=kernel.OpenProcess(0x00100000,False,launcher['pid'])
        if handle:
            try:
                while True:
                    status=kernel.WaitForSingleObject(handle,60000)
                    if status==0:break
                    if status!=258:raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(handle)
    validate()

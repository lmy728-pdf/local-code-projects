# -*- coding: utf-8 -*-
"""2026-11-10：从已核验断点续跑原 Full，完成后核验并提供桌面结果。"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import ctypes
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sqlite3
import sys
import traceback
from contextlib import closing
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import spline_p10_lognormal_worker as m
    start=datetime.now().astimezone().isoformat()
    control=ROOT/'运行控制'
    error_file=control/'resume_20261110_error.json'
    assert not error_file.exists(), '已有运行错误记录，需检查后再启动'
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        # 原 Full 自动加载并跳过所有已有 replication，不启用额外重试或改变方法。
        result=m.full_run(ROOT)
        with closing(sqlite3.connect((ROOT/'checkpoints'/'replications.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
            assert db.execute('PRAGMA integrity_check').fetchone()==('ok',)
            assert db.execute('SELECT COUNT(*) FROM packets').fetchone()[0]==16000
            counts=db.execute('SELECT dgp,error,count(*) FROM packets GROUP BY dgp,error').fetchall()
            assert len(counts)==16 and all(x[2]==1000 for x in counts)
            receipt=json.loads((control/'pause_receipt.json').read_text(encoding='utf-8'))
            with closing(sqlite3.connect(Path(receipt['backup']).as_uri()+'?mode=ro',uri=True)) as old:
                preserved=0
                for d,e,r,attempt,payload in old.execute('SELECT * FROM packets'):
                    assert db.execute('SELECT attempt,payload FROM packets WHERE dgp=? AND error=? AND rep=?',(d,e,r)).fetchone()==(attempt,payload)
                    preserved+=1
                assert preserved==9637
        import pandas as pd
        raw=pd.read_csv(ROOT/'full_replication_results.csv')
        assert len(raw)==64000
        assert raw.groupby(['dgp_id','error_id','method']).size().eq(1000).all()
        summary,diag=m.summarize_results(raw)
        saved=pd.read_csv(ROOT/'full_summary.csv')
        keys=['dgp_id','error_id','method']
        for metric in ('coverage_pct','bias','rmse','mean_T','sd_T','point_n','test_n'):
            import numpy as np
            assert np.allclose(summary.sort_values(keys)[metric],saved.sort_values(keys)[metric],equal_nan=True)
        required=['full_summary.csv','full_replication_results.csv','full_paired_comparisons.csv',
            'full_nuisance_diagnostics.csv','full_failures.csv','full_report.html',
            'coverage_lognormal.png','deterioration_lognormal.png','bias_rmse_lognormal.png',
            'score_diagnostics.png','nuisance_diagnostics.png','p10_vs_p5.png','completed.json']
        assert all((ROOT/name).is_file() and (ROOT/name).stat().st_size for name in required)
        # 桌面直接提供图表副本；全部原始记录仍保存在正式运行目录。
        desktop=Path.home()/'Desktop'
        target_html=desktop/'DML_p10_Lognormal_R1000_结果表.html'
        target_figure=desktop/'DML_p10_Lognormal_R1000_对照图.png'
        assert not target_html.exists() and not target_figure.exists(), '已有同名桌面结果，未覆盖'
        m.figure_panels(desktop,saved,target_figure.name,'Lognormal 全量结果 · 每场景 R=1000',[
            ('Lognormal','coverage_pct','Score-test Coverage (%)',95),
            ('Lognormal','bias','Bias',0),('Lognormal','rmse','RMSE',None),
            ('Lognormal','deterioration_pp','相对 Gaussian 覆盖下降 (百分点)',0)])
        shutil.copy2(ROOT/'full_report.html',target_html)
        report=dict(passed=True,start=start,finished=datetime.now().astimezone().isoformat(),
            data_n=16000,method_records=64000,preserved_previous_datasets=preserved,
            point_fail=int(saved.point_fail.sum()),test_fail=int(saved.test_fail.sum()),
            desktop_outputs=[str(target_html),str(target_figure)],output_directory=str(ROOT))
        m.atomic_json(control/'resume_20261110_verified.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except BaseException:
        m.atomic_json(error_file,dict(error=traceback.format_exc(),time=datetime.now().astimezone().isoformat()))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()


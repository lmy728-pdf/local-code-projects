# -*- coding: utf-8 -*-
"""隔离验证：直接使用原 collect/持久化函数，以快速假任务核验收尾及续跑。"""
import ast, collections, concurrent.futures, contextlib, importlib.util, json
import multiprocessing, os, pathlib, sqlite3, sys, tempfile, threading, time, types, uuid
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent))
import safe_pause_guard as guard

def main():
    source=ast.parse((ROOT/'spline_p10_lognormal_worker.py').read_text(encoding='utf-8'))
    names={'checkpoint_db','record_batch','load_packets','append_log','save_progress','collect','atomic_json'}
    chosen=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    fake=types.ModuleType('isolated_pause_test')
    fake.__dict__.update(sqlite3=sqlite3,contextmanager=contextlib.contextmanager,
        json=json,Path=Path,datetime=datetime,time=time,Counter=collections.Counter,
        multiprocessing=multiprocessing,WORKERS=8,METHODS=('M1','M2','M3','S'),
        FLUSH_EVERY=32,ERROR_LABELS={0:'Gaussian',1:'LN(2)'},tqdm=tqdm,
        FIRST_COMPLETED=concurrent.futures.FIRST_COMPLETED,wait=concurrent.futures.wait,
        os=os,uuid=uuid,clean_json=lambda value:value,init_worker=lambda barrier:None)
    calls=[]
    def worker(config,task,methods):
        calls.append(task)
        time.sleep(.02)
        return dict(task=list(task),rows={m:dict(theta=1.0,T=0.0) for m in methods})
    def validate(packet,config,task,methods=None):
        assert packet['task']==list(task)
        assert set(packet['rows'])==set(fake.METHODS)
    def pool(**kwargs):
        return concurrent.futures.ThreadPoolExecutor(max_workers=kwargs['max_workers'])
    fake.worker_packet=worker
    fake.validate_packet=validate
    fake.ProcessPoolExecutor=pool
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(ROOT/'spline_p10_lognormal_worker.py'),'exec'),fake.__dict__)
    config=dict(mode='full',R=100,errors=[0,1],config_hash='isolated_test')
    with tempfile.TemporaryDirectory(prefix='dml_pause_selftest_') as directory:
        destination=Path(directory)
        guard.install(fake,destination,time.time()+.12,config['config_hash'],
                      require_active=False,export_tables=False)
        try:
            with fake.checkpoint_db(destination,config) as db:
                fake.collect(destination,db,config)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('定时暂停未触发')
        state=guard.status(destination)
        assert state['pause_guard_status.json']['status']=='paused_saved_verified',state
        paused=state['pause_receipt.json']['completed']
        with fake.checkpoint_db(destination,config) as db:
            saved=fake.load_packets(db,config)
        assert set(saved)==set(calls), '有已执行任务未保存'
        assert len(calls)==len(set(calls))==paused
        assert 0<paused<400
        # 原 collect 自动跳过所有已保存任务；验证恢复后既不漏算也不重复算。
        with fake.checkpoint_db(destination,config) as db:
            restored=fake.collect(destination,db,config)
        assert len(restored)==400
        assert len(calls)==len(set(calls))==400
        print(json.dumps(dict(passed=True,paused_datasets=paused,resumed_datasets=400,
                              executed_equals_saved=True,no_duplicate_on_resume=True,
                              backup_integrity_passed=True,original_collect_used=True)))
        return dict(passed=True,paused_datasets=paused,resumed_datasets=400,
                    executed_equals_saved=True,no_duplicate_on_resume=True,
                    backup_integrity_passed=True,original_collect_used=True)

if __name__=='__main__':
    receipt=main()
    guard.atomic_json(Path(__file__).parent/'pause_guard_validation.json',receipt)


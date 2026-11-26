"""Update code/text only; do not execute notebook or fit a model."""
import ast,hashlib,inspect,json,pathlib,shutil
from datetime import datetime
import sklearn.ensemble._gb as gb
ROOT=pathlib.Path(__file__).resolve().parent
NB=ROOT.parent/'Huber_DML_pilot.ipynb'
def main():
    backup=ROOT/'backups'/('r200_quantile_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False)
    files=[NB]+[ROOT/name for name in ['huber_methods.py','run_huber_pilot.py','summarize_pilot.py','build_notebook.py','README.md','implementation_notes.md']]
    for p in files:shutil.copy2(p,backup/p.name)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    notebook=json.loads(NB.read_text(encoding='utf-8'))
    for c in notebook['cells']:
        s=''.join(c['source'])
        if c['cell_type']=='code':
            if 'STAGE = ' in s:
                s=s.replace('每场景100次','每场景200次').replace('WORKERS = 4','PILOT_REPLICATIONS = 200\nWORKERS = 4')
            if 'GBR_COMMON=dict(' in s:
                s=s.replace('GBR_COMMON=dict(', '# Fixed quantile level; sklearn updates delta on TRAIN residuals at EVERY boosting stage.\nHUBER_QUANTILE = 0.90\nGBR_COMMON=dict(')
                s=s.replace('alpha=.90','alpha=HUBER_QUANTILE')
            if "config.update(" in s:
                s=s.replace('replications=2 if args.stage==\'smoke\' else 100','replications=2 if args.stage==\'smoke\' else PILOT_REPLICATIONS')
                s=s.replace('alpha=.9,','alpha=HUBER_QUANTILE,')
                s=s.replace('config.update(',"config.update(huber_threshold_rule='per_stage_training_absolute_residual_quantile',\n              huber_quantile=HUBER_QUANTILE,\n              huber_delta_is_adaptive=True,\n              ")
            s=s.replace('batch(range(2,100))',"batch(range(2,config['replications']))")
            s=s.replace('R=100最坏coverage MCSE约5个百分点','R=200最坏coverage MCSE约3.54个百分点')
            s=s.replace("'C':'C: GBR Huber'","'C':'C: GBR Huber (adaptive delta, q=0.90)'")
            c['execution_count']=None;c['outputs']=[]
            c['metadata'].pop('execution',None)
            ast.parse(s)
        else:
            s=s.replace('每场景100次','每场景200次').replace('余下98次','余下198次').replace('默认 `STAGE="smoke"`（每场景2次）；改为 `"pilot"` 才会每场景200次','当前 `STAGE="pilot"`（每场景200次）；改为 `"smoke"` 则只检查每场景2次')
            if s.startswith('# Huber-DML'):
                s+='''\n\n### 2026-11-02 更新：R=200，分位数自适应Huber阈值

C使用sklearn原生的分位数自适应Huber：第k折第t轮，delta[k,t]取训练集绝对残差abs(Y−F[t−1](X))的90%分位数。delta随折和轮次变化；alpha=0.90是固定分位水平，不是固定delta，也不是自动搜索alpha。只使用外层train，不读取held-out残差作阈值校准。

原版本已采用同一机制，本次不声称换成了另一种估计量。正式pilot每场景200组，共600组数据、1800条A/B/C记录；先验收前2组，再运行余下198组。boosting仍为100轮，seed中的100/1000倍数不改。代码已更新，未运行；旧输出保存在修改前备份，旧实验目录不变。
'''
        c['source']=s.splitlines(keepends=True)
    # Make the parameter value explicit even if the user had switched back to smoke.
    for c in notebook['cells']:
        if c['cell_type']=='code' and 'PILOT_REPLICATIONS = 200' in ''.join(c['source']):
            c['source']=''.join(c['source']).replace('STAGE = "smoke"','STAGE = "pilot"').splitlines(keepends=True)
    NB.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    # Keep CLI implementation aligned with notebook definitions.
    methods=ROOT/'huber_methods.py';s=methods.read_text(encoding='utf-8')
    s=s.replace('GBR_COMMON=dict(', '# sklearn recomputes delta at every stage using TRAIN residuals.\nHUBER_QUANTILE = 0.90\nGBR_COMMON=dict(').replace('alpha=.90','alpha=HUBER_QUANTILE')
    methods.write_text(s,encoding='utf-8')
    runner=ROOT/'run_huber_pilot.py';s=runner.read_text(encoding='utf-8')
    s=s.replace('run_replication,GBR_COMMON','run_replication,GBR_COMMON,HUBER_QUANTILE')
    s=s.replace("else 100,workers=", "else 200,workers=").replace('batch(range(2,100))',"batch(range(2,config['replications']))").replace('alpha=.9,','alpha=HUBER_QUANTILE,')
    s=s.replace('config.update(stage=',"config.update(huber_threshold_rule='per_stage_training_absolute_residual_quantile',\n                  huber_quantile=HUBER_QUANTILE,huber_delta_is_adaptive=True,stage=")
    runner.write_text(s,encoding='utf-8')
    reporter=ROOT/'summarize_pilot.py';s=reporter.read_text(encoding='utf-8').replace('R=100最坏coverage MCSE约5个百分点','R=200最坏coverage MCSE约3.54个百分点').replace("'C':'C: GBR Huber'","'C':'C: GBR Huber (adaptive delta, q=0.90)'")
    reporter.write_text(s,encoding='utf-8')
    for name in ['README.md','implementation_notes.md','build_notebook.py']:
        p=ROOT/name;s=p.read_text(encoding='utf-8').replace('每场景100次','每场景200次').replace('余下98次','余下198次').replace('每场景100组','每场景200组')
        if name=='build_notebook.py':s=s.replace('STAGE = "smoke"','STAGE = "pilot"')
        if name.endswith('.md'):
            s+='\n\n2026-11-02：正式pilot更新为每场景200次；C保留sklearn原生逐轮训练绝对残差90%分位数自适应delta（alpha仍为0.90，不是搜索alpha）。notebook默认pilot，CLI默认smoke。本次没有运行实验。\n'
        p.write_text(s,encoding='utf-8')
    evidence={'date':'2026-11-02','backup':str(backup),'old_hashes':before,'new_replications_per_scenario':200,
              'algorithm_changed':False,'delta_adaptive':True,'alpha':.9,'model_fits':0,
              'installed_source_file':inspect.getsourcefile(gb.set_huber_delta),
              'set_huber_delta':inspect.getsource(gb.set_huber_delta),
              'fit_stage_huber_call':inspect.getsource(gb.GradientBoostingRegressor._fit_stage).split('# TODO:')[0]}
    (ROOT/'r200_quantile_update.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Notebook updated; no fits. Backup:',backup)
if __name__=='__main__':main()

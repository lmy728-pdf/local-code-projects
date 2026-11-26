# -*- coding: utf-8 -*-
"""只读提取全量实验前 R 次重复，生成阶段报告；不训练模型、不改正式断点。"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import spline_p10_lognormal_worker as m

def extract(r):
    config=json.loads((ROOT/'configuration.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'spline_p10_lognormal_worker.py').read_bytes()).hexdigest()==config['worker_sha256']
    expected={(d,e,k) for d in range(2) for e in config['errors'] for k in range(r)}
    path=ROOT/'checkpoints'/'replications.sqlite3'
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30)) as source:
        source.execute('BEGIN')
        assert source.execute('SELECT config_hash FROM manifest').fetchall()==[(config['config_hash'],)]
        packets=source.execute('SELECT dgp,error,rep,attempt,payload FROM packets WHERE rep < ?',(r,)).fetchall()
        tasks={tuple(row[:3]) for row in packets}
        assert tasks<=expected
        counts={f'DGP{d+1} {m.ERROR_LABELS[e]}':sum((d,e,k) in tasks for k in range(r))
                for d in range(2) for e in config['errors']}
        if tasks!=expected:
            return None,counts
        attempts=source.execute('SELECT dgp,error,rep,attempt,payload FROM attempts WHERE rep < ?',(r,)).fetchall()
        manifest=source.execute('SELECT * FROM manifest').fetchall()
    # 内存数据库只包含所选重复；失败表也严格限于该批，避免混入后续 replication。
    memory=sqlite3.connect(':memory:')
    memory.executescript('''
        CREATE TABLE manifest(config_hash TEXT PRIMARY KEY,config TEXT NOT NULL);
        CREATE TABLE packets(dgp INTEGER,error INTEGER,rep INTEGER,attempt INTEGER,payload TEXT,
                             PRIMARY KEY(dgp,error,rep));
        CREATE TABLE attempts(dgp INTEGER,error INTEGER,rep INTEGER,attempt INTEGER,payload TEXT,
                              PRIMARY KEY(dgp,error,rep,attempt));
    ''')
    memory.executemany('INSERT INTO manifest VALUES(?,?)',manifest)
    memory.executemany('INSERT INTO packets VALUES(?,?,?,?,?)',packets)
    memory.executemany('INSERT INTO attempts VALUES(?,?,?,?,?)',attempts)
    memory.commit()
    try:
        loaded=m.load_packets(memory,config)
        assert set(loaded)==expected
        raw,failures=m.flatten_results(memory,loaded,config)
    finally:
        memory.close()
    assert len(raw)==len(expected)*4 and raw.replication_id.between(0,r-1).all()
    summary,nuisance=m.summarize_results(raw)
    paired=m.pair_results(raw,summary)
    assert len(summary)==64 and len(paired)==48
    assert summary.planned_n.eq(r).all()
    assert (raw.groupby(['dgp_id','error_id','replication_id'])[['data_hash','fold_hash']].nunique()==1).all().all()
    provenance=dict(snapshot_R=r,source_R=config['R'],source_config_hash=config['config_hash'],
                    data_n=len(expected),method_records=len(raw),replication_ids=f'0–{r-1}',
                    snapshot_at=datetime.now().astimezone().isoformat(),
                    source_checkpoint=str(path),read_only_source=True,
                    estimation_rerun=False,original_summary_and_pairing_functions=True)
    return (raw,failures,summary,nuisance,paired,provenance),counts

def export(result,destination,r):
    raw,failures,summary,nuisance,paired,provenance=result
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name,frame in [('replication_results',raw),('failures',failures),('summary',summary),
                       ('nuisance_diagnostics',nuisance),('paired_comparisons',paired)]:
        m.atomic_csv(destination/f'pilot_R{r}_{name}.csv',frame)
    title=f'Lognormal 前 {r} 次阶段结果 · 每场景 R={r}'
    figure=f'Pilot_R{r}_对照图.png'
    m.figure_panels(destination,summary,figure,title,[
        ('Lognormal','coverage_pct','Score-test Coverage (%)',95),
        ('Lognormal','bias','Bias',0),
        ('Lognormal','rmse','RMSE',None),
        ('Lognormal','deterioration_pp','相对 Gaussian 覆盖下降 (百分点)',0)])
    focus=summary[summary.error_id.eq(14)]
    focused_pair=paired[paired.error_id.eq(14)]
    cover=summary.pivot(index=['dgp_id','dgp','error_id','error'],columns='method',values='coverage_pct').reset_index()
    cover=cover[['dgp','error',*m.METHODS]]
    rename={'dgp':'场景','error':'误差','method':'方法','reference':'比较基准',
        'coverage_pct':'Coverage (%)','bias':'Bias','rmse':'RMSE','deterioration_pp':'覆盖下降 (pp)',
        'point_n':'有效点估计','test_n':'有效检验','point_fail':'点估计失败','test_fail':'检验失败',
        'mean_T':'Mean(T)','sd_T':'SD(T)','left_reject_pct':'左尾拒绝 (%)','right_reject_pct':'右尾拒绝 (%)',
        'y_nuisance_mse':'Y nuisance MSE','d_nuisance_mse':'D nuisance MSE',
        'coverage_diff_pp':'S覆盖差 (pp)','coverage_diff_MCSE_pp':'配对MCSE (pp)',
        'rmse_ratio':'S/基准 RMSE','bias_diff':'S Bias差','rmse_diff':'S RMSE差',
        'test_pairs':'共同有效检验','point_pairs':'共同有效点估计'}
    short=['dgp','method','coverage_pct','bias','rmse','deterioration_pp','mean_T','sd_T','point_n','test_n','point_fail','test_fail']
    ps=['dgp','reference','test_pairs','point_pairs','coverage_diff_pp','coverage_diff_MCSE_pp','bias_diff','rmse_diff','rmse_ratio']
    tables=[('全部场景覆盖率',cover),('LN(2) 极端偏斜重尾',focus[short]),
            ('LN(2)：S 与三个基准的配对比较',focused_pair[ps]),('完整指标',summary),
            ('全部配对比较',paired),('第一阶段误差诊断',nuisance),('失败记录',failures)]
    table_html=''.join('<h2>'+html.escape(name)+'</h2><div class="scroll">'+
        frame.rename(columns=rename).to_html(index=False,na_rep='—',float_format=lambda x:f'{x:.5g}')+'</div>'
        for name,frame in tables)
    intro=(f'固定提取每个场景 replication 0–{r-1}，共 {provenance["data_n"]:,} 组数据、'
           f'{len(raw):,} 条方法记录。包含失败，未按结果筛样。'
           '每项指标使用其实际有效分母；配对比较核对同一数据和 folds。'
           '原全量实验继续运行，下午五点的安全暂停设置不变。'
           f'这是全量模拟的 R={r} 阶段切片，不是另一次独立验证实验；微小覆盖率差异需要结合 MCSE 判断。'
           '所有结果均为原 Score-test，未重新训练或调整统计方法。'
           'M2 学习条件中位数，Y nuisance MSE 用条件均值作为共同诊断参照；M3 使用 transformed-outcome 目标。')
    document=('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'''+title+'''</title>
<style>body{font:15px/1.65 "Microsoft YaHei",sans-serif;margin:32px;background:#f3f6fa;color:#18344a}
h1,h2{color:#165c70}.scroll{overflow:auto}table{border-collapse:collapse;background:white;white-space:nowrap}
th,td{border:1px solid #d0dce3;padding:6px 9px;text-align:right}th{background:#17576c;color:white}
tbody tr:nth-child(even){background:#eaf3f8}img{max-width:100%;background:white}</style>
<h1>p=10 · '''+html.escape(title)+'</h1><p>'+html.escape(intro)+'</p><p>快照时间：'+provenance['snapshot_at']+
        '</p><img src="'+figure+'">'+table_html+'</html>')
    m.atomic_text(destination/f'Pilot_R{r}_结果表.html',document)
    m.atomic_json(destination/'snapshot_validation.json',dict(provenance,passed=True,
        summary_rows=len(summary),paired_rows=len(paired),point_fail=int(summary.point_fail.sum()),
        test_fail=int(summary.test_fail.sum()),failure_events=len(failures),completed=True))
    print('\nCOVERAGE_TABLE\n'+cover.to_string(index=False),flush=True)
    print('\nLN2_FOCUS\n'+focus[short].to_string(index=False),flush=True)
    print('\nLN2_PAIRED\n'+focused_pair[ps].to_string(index=False),flush=True)
    print('REPORT_READY '+str(destination),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--r',type=int,default=200)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--destination')
    args=parser.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    destination=Path(args.destination) if args.destination else ROOT.parent/f'R{args.r}阶段结果'
    last=-1
    while True:
        result,counts=extract(args.r)
        current=min(counts.values())
        if current//20!=last or result is not None:
            print(json.dumps(dict(time=datetime.now().isoformat(timespec='seconds'),
                  min_per_scene=current,max_per_scene=max(counts.values()),target=args.r),ensure_ascii=False),flush=True)
            last=current//20
        if result is not None:
            export(result,destination,args.r)
            return
        progress=json.loads((ROOT/'progress.json').read_text(encoding='utf-8'))
        if not args.watch or progress['status'] not in ('running','exporting'):
            raise SystemExit(f'尚未达到每场景 {args.r}：{current}–{max(counts.values())}；状态 {progress["status"]}')
        time.sleep(30)

if __name__=='__main__':
    main()


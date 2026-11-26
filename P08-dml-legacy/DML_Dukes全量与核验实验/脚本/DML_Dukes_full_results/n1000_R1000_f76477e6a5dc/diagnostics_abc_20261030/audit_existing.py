"""Read saved runs; no training, sampling, notebook execution or input mutation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import ast, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
NOTEBOOK=Path('C:/Users/Lenovo/Desktop/DML_Dukes_n1000_R1000.ipynb')
METHODS=['original_mean_dml','standard_score','dukes_dr_gcm']

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,data): (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def source_snapshot(nb):
    lines=['# READ-ONLY SOURCE SNAPSHOT. Do not execute: contains the original launch cell.\n']
    functions={}; locations=[]; names=None; code_cell=0
    for cell_id,cell in enumerate(nb['cells'],1):
        if cell['cell_type']!='code': continue
        code_cell+=1; source=''.join(cell['source'])
        lines.extend(['\n',f'# %% Notebook cell {cell_id}; code cell {code_cell}\n'])
        start=len(lines)+1
        for node in ast.parse(source).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                functions[node.name]=(node,source)
                locations.append(dict(function=node.name,notebook_cell_1based=cell_id,code_cell_1based=code_cell,
                                      cell_line_start=node.lineno,cell_line_end=node.end_lineno,snapshot_line=start+node.lineno-1))
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COMPUTATIONAL_FUNCTIONS' for t in node.targets):
                names=ast.literal_eval(node.value)
        lines.extend(source.splitlines(keepends=True))
        if lines[-1] and not lines[-1].endswith('\n'): lines[-1]+='\n'
    (OUT/'notebook_source_snapshot.py').write_text(''.join(lines),encoding='utf-8')
    csvsave('code_locations.csv',locations)
    normalized={k:ast.dump(ast.parse(ast.get_source_segment(functions[k][1],functions[k][0])),include_attributes=False) for k in names}
    init=next(x for x in functions['NumericalFailure'][0].body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    normalized['NumericalFailure.__init__']=ast.dump(ast.Module(body=[init],type_ignores=[]),include_attributes=False)
    return hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest(),locations

def nw(x,y,h,loo=False):
    if h is None:
        return (y.sum()-y)/(len(y)-1) if loo else np.full(len(y),y.mean())
    distances=(x[:,None]-x[None,:])/h
    log_weights=-0.5*distances**2
    if loo: np.fill_diagonal(log_weights,-np.inf)
    log_weights-=np.max(log_weights,axis=1,keepdims=True)
    weights=np.exp(log_weights)
    return (weights@y)/weights.sum(axis=1)

def main():
    started=time.perf_counter()
    files=sorted(p for p in RUN.rglob('*') if p.is_file() and not p.is_relative_to(OUT)) + [NOTEBOOK]
    before={str(p):digest(p) for p in files}
    cfg=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    nb=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code_hash,locations=source_snapshot(nb)
    assert code_hash==cfg['computational_code_hash'],'Notebook differs from executed computational code'
    cfg_copy=dict(cfg); expected=cfg_copy.pop('config_hash')
    assert hashlib.sha256(json.dumps(cfg_copy,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==expected
    maxima={}; problems=[]; stats=[]; preds=[]; folds_out=[]; seen=set()
    def compare(key,a,b,task):
        a,b=np.asarray(a),np.asarray(b)
        error=float(np.max(np.abs(a-b)))
        maxima[key]=max(maxima.get(key,0),error)
        if not np.allclose(a,b,rtol=1e-11,atol=1e-10,equal_nan=False):
            problems.append(dict(task=task,check=key,max_absolute_difference=error))
    tasks=sorted((RUN/'tasks').glob('*.json'))
    with threadpool_limits(limits=1):
        for count,path in enumerate(tasks,1):
            task=json.loads(path.read_text(encoding='utf-8'))
            rows={r['method']:r for r in task['rows']}
            assert set(rows)==set(METHODS)
            A,B,C=[rows[m] for m in METHODS]
            scenario=A['scenario']; rep=A['replication']; key=(scenario,rep)
            assert key not in seen; seen.add(key)
            assert all(r['config_hash']==cfg['config_hash'] and r['scenario']==scenario and r['replication']==rep for r in rows.values())
            assert all(r['dataset_seed']==A['dataset_seed'] and r['crossfit_seed']==A['crossfit_seed'] and r['fold_hash']==A['fold_hash'] for r in rows.values())
            assert all(r['n']==1000 for r in rows.values())
            array_path=RUN/task['arrays_file']; assert before[str(array_path)]==task['arrays_sha256']
            with np.load(array_path,allow_pickle=False) as arr:
                v={k:arr[k] for k in arr.files}
            nonfinite=sum(int((~np.isfinite(value)).sum()) for value in v.values())
            if nonfinite: problems.append(dict(task=path.name,check='nonfinite_arrays',count=nonfinite))
            a,b,g,m,ell,dr,yr=[v[k] for k in ['a','b','g_hat','m_hat','ell_hat','d_res','y_res']]
            n=len(a); theta=cfg['theta0']; ids=v['fold_ids']; indices=np.empty(n,dtype=np.int64)
            for k,(train,test) in enumerate(KFold(5,shuffle=True,random_state=A['crossfit_seed']).split(np.arange(n))):
                assert not np.intersect1d(train,test).size
                indices[test]=k
            compare('fold_ids',ids,indices,path.name)
            assert hashlib.sha256(ids.tobytes()).hexdigest()==A['fold_hash']
            compare('A_B_treatment_residual',dr,a,path.name)
            DA,DB=dr+g,a+g
            YA,YB=yr+ell,b+theta*DB+m
            compare('A_B_reconstructed_D',DA,DB,path.name)
            compare('A_B_reconstructed_Y',YA,YB,path.name)
            difference=m-(ell-theta*g)
            pred=dict(scenario=scenario,replication=rep,treatment_residual_maxdiff=float(np.max(abs(dr-a))),
                      outcome_prediction_mean_difference=float(difference.mean()),
                      outcome_prediction_rmse=float(np.sqrt(np.mean(difference**2))),
                      outcome_prediction_max_abs_difference=float(np.max(abs(difference))),
                      outcome_predictions_equal_at_1e_12=bool(np.allclose(m,ell-theta*g,rtol=1e-12,atol=1e-12)))
            preds.append(pred)
            J=np.mean(dr**2); estimate=np.mean(dr*yr)/J
            se=np.sqrt(np.var(dr*(yr-estimate*dr)/J,ddof=1)/n)
            compare('A_theta',estimate,A['theta_hat'],path.name); compare('A_SE',se,A['theta_se'],path.name)
            TB=math.sqrt(n)*np.mean(a*b)/np.std(a*b,ddof=1)
            compare('B_score',a*b,v['scores_B'],path.name); compare('B_T',TB,B['T'],path.name)
            rD=np.empty(n); rY=np.empty(n)
            for diag in task['folds']:
                k=diag['fold']; ix=ids==k; G,M=v['G'][ix],v['M'][ix]
                Gdirect=nw(g[ix],b[ix],diag['bandwidth_G']); Mdirect=nw(m[ix],a[ix],diag['bandwidth_M'])
                compare('kernel_G',Gdirect,G,path.name); compare('kernel_M',Mdirect,M,path.name)
                compare('G_LOOCV_at_saved_h',np.mean((b[ix]-nw(g[ix],b[ix],diag['bandwidth_G'],True))**2),diag['cv_loss_G'],path.name)
                compare('M_LOOCV_at_saved_h',np.mean((a[ix]-nw(m[ix],a[ix],diag['bandwidth_M'],True))**2),diag['cv_loss_M'],path.name)
                alpha=np.dot(G,a[ix])/np.dot(G,G) if np.any(G) else 0.
                beta=np.dot(M,b[ix])/np.dot(M,M) if np.any(M) else 0.
                rD[ix]=a[ix]-alpha*G; rY[ix]=b[ix]-beta*M
                compare('alpha',alpha,diag['alpha'],path.name); compare('beta',beta,diag['beta'],path.name)
                moment_G=float(np.mean(G*rD[ix])); moment_M=float(np.mean(M*rY[ix]))
                compare('mean_G_rD',moment_G,0,path.name); compare('mean_M_rY',moment_M,0,path.name)
                folds_out.append(dict(scenario=scenario,replication=rep,fold=k,alpha=alpha,beta=beta,
                                      mean_G_rD=moment_G,mean_M_rY=moment_M,
                                      G_subterm_rms=float(np.sqrt(np.mean((G*rD[ix])**2))),
                                      M_subterm_rms=float(np.sqrt(np.mean((M*rY[ix])**2))),
                                      bandwidth_G=diag['bandwidth_G'],bandwidth_M=diag['bandwidth_M'],
                                      boundary_G=diag['boundary_G'],boundary_M=diag['boundary_M'],
                                      boundary_side_G=diag['boundary_side_G'],boundary_side_M=diag['boundary_side_M']))
            compare('rD',rD,v['rD'],path.name); compare('rY',rY,v['rY'],path.name)
            product=rD*rY; full=product-v['G']*rD-v['M']*rY
            variance=np.var(full,ddof=1); TC=np.sqrt(n)*full.mean()/np.sqrt(variance)
            compare('C_full_score',full,v['scores_C'],path.name)
            compare('C_variance',variance,C['score_variance'],path.name)
            compare('C_T',TC,C['T'],path.name)
            compare('C_p',2*norm.sf(abs(TC)),C['p_value'],path.name)
            fold_mean=np.mean([full[ids==k].mean() for k in range(5)])
            compare('fold_mean_vs_pooled',fold_mean,full.mean(),path.name)
            compare('subterms_mean_only_zero',product.mean(),full.mean(),path.name)
            pred['full_variance_over_product_only']=float(variance/np.var(product,ddof=1))
            pred['C_minus_product_only_T']=float(TC-np.sqrt(n)*product.mean()/np.std(product,ddof=1))
            values=[(estimate-theta)/se,TB,TC]
            for label,method,value in zip('ABC',METHODS,values):
                original=rows[method]; finite=bool(np.isfinite(value))
                cutoff=1.96 if label=='A' else float(norm.ppf(.975))
                assert bool(abs(value)<=cutoff)==original['covered']
                stats.append(dict(scenario=scenario,replication=rep,method=label,method_id=method,
                                  theta_tested=theta,T=float(value),finite=finite,status=original['status'],
                                  negative_tail=bool(value < -1.96),positive_tail=bool(value > 1.96),
                                  within_1_96=bool(abs(value)<=1.96),original_covered=original['covered'],
                                  original_cutoff=cutoff,warning_count=len(original.get('warnings',[])),
                                  config_hash=cfg['config_hash'],task_file=path.name,
                                  array_sha256=task['arrays_sha256']))
            if count%500==0: print(f'Checked {count}/{len(tasks)} saved datasets; {time.perf_counter()-started:.1f}s',flush=True)
    assert len(tasks)==3000 and all((s['name'],r) in seen for s in cfg['scenarios'] for r in range(1000))
    summaries=[]; predictions=[]
    for s in cfg['scenarios']:
        scenario=s['name']; pp=[p for p in preds if p['scenario']==scenario]
        predictions.append(dict(scenario=scenario,datasets=len(pp),
                                D_prediction_maxdiff=max(p['treatment_residual_maxdiff'] for p in pp),
                                mean_outcome_prediction_rmse=float(np.mean([p['outcome_prediction_rmse'] for p in pp])),
                                pooled_outcome_prediction_rmse=float(np.sqrt(np.mean([p['outcome_prediction_rmse']**2 for p in pp]))),
                                max_outcome_prediction_difference=max(p['outcome_prediction_max_abs_difference'] for p in pp),
                                identical_outcome_prediction_datasets=sum(p['outcome_predictions_equal_at_1e_12'] for p in pp),
                                mean_full_to_product_variance_ratio=float(np.mean([p['full_variance_over_product_only'] for p in pp]))))
        for method in 'ABC':
            subset=[r for r in stats if r['scenario']==scenario and r['method']==method]
            valid=[r for r in subset if r['finite'] and r['status']=='ok']; t=np.array([r['T'] for r in valid])
            summaries.append(dict(scenario=scenario,method=method,planned=1000,attempted=len(subset),valid=len(valid),
                                  nonfinite=sum(not r['finite'] for r in subset),failed=sum(r['status']!='ok' for r in subset),
                                  warning_records=sum(r['warning_count']>0 for r in subset),
                                  mean_T=float(t.mean()),sd_T=float(t.std(ddof=1)),mean_T_mcse=float(t.std(ddof=1)/np.sqrt(len(t))),
                                  p_T_lt_minus1_96=float(np.mean(t < -1.96)),p_T_gt_1_96=float(np.mean(t > 1.96)),
                                  p_abs_T_le_1_96=float(np.mean(abs(t)<=1.96)),
                                  critical_value_difference_count=sum(r['within_1_96']!=r['original_covered'] for r in subset)))
    unchanged=all(digest(Path(p))==h for p,h in before.items())
    assert unchanged,'An input file changed during the audit'
    csvsave('standardized_statistics.csv',stats); csvsave('standardized_statistics_summary.csv',summaries)
    csvsave('prediction_differences.csv',preds); csvsave('prediction_differences_summary.csv',predictions)
    csvsave('fold_verification.csv',folds_out)
    save('input_manifest.json',dict(notebook=str(NOTEBOOK),run=str(RUN),files_sha256=before,unchanged_after_audit=unchanged))
    save('verification.json',dict(code_hash_matches=True,computational_code_hash=code_hash,datasets=len(tasks),
                                  observations=len(tasks)*1000,folds=len(folds_out),kernel_fits_verified=2*len(folds_out),
                                  input_files_unchanged=unchanged,max_absolute_errors=maxima,problems=problems,
                                  elapsed_seconds=time.perf_counter()-started,
                                  scope='Independent fixed-bandwidth NW, saved-h LOO loss, projections, full score, aggregation and statistics; no refitting or bandwidth reoptimization'))
    print(json.dumps(dict(statistics=summaries,predictions=predictions,max_errors=maxima,problems=problems),ensure_ascii=False,indent=2))

if __name__=='__main__': main()

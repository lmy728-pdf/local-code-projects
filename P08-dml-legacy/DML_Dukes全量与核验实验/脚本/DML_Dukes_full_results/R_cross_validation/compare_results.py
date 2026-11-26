"""Compare saved R results; controlled Python calculation at R-selected bandwidths."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import ast,hashlib,json,pathlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits
ROOT=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def main():
    samples=pd.read_csv(ROOT/'selected_samples.csv',float_precision='round_trip')
    nb=pathlib.Path('C:/Users/Lenovo/Desktop/DML_Dukes_n1000_R1000.ipynb')
    notebook=json.loads(nb.read_text(encoding='utf-8'))
    required={'NumericalFailure','_finite_vector','nw_predict','_nw_from_distances','stable_projection','corrected_score','score_test'}
    definitions=[]
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        definitions.extend(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in required)
    frozen='import numpy as np\nfrom scipy.stats import norm\n\n'+'\n\n'.join(definitions)+'\n'
    (ROOT/'python_correction_frozen.py').write_text(frozen,encoding='utf-8')
    namespace={};exec(compile(frozen,'python_correction_frozen.py','exec'),namespace)
    nw,corrected,score_test=[namespace[k] for k in ('nw_predict','corrected_score','score_test')]
    records=[]; fold_rows=[]; controls=[]
    with threadpool_limits(limits=1):
        for meta in samples.to_dict('records'):
            stem=meta['stem']; path=ROOT/'r_results'/f'{stem}.json'
            if not path.exists():raise RuntimeError(f'R verification is incomplete: {stem}')
            rr=json.loads(path.read_text(encoding='utf-8'))
            if rr['status']!='ok':
                records.append({**meta,'status':'R_failed','R_error':rr.get('error')});continue
            inp=ROOT/'inputs'/f'{stem}.csv'; assert sha(inp)==meta['input_sha256']
            x=pd.read_csv(inp,float_precision='round_trip')
            rf=pd.read_csv(ROOT/'r_results'/f'{stem}_folds.csv',float_precision='round_trip')
            pf=pd.read_csv(ROOT/'inputs'/f'{stem}_folds.csv',float_precision='round_trip')
            ra=pd.read_csv(ROOT/'r_results'/f'{stem}_arrays.csv',float_precision='round_trip')
            assert np.array_equal(x.observation_id,ra.observation_id) and np.array_equal(x.fold,ra.fold)
            a=x.D.to_numpy()-x.g_initial.to_numpy(); b=x.y_transformed.to_numpy()-x.m_initial.to_numpy()
            Gcontrol=np.empty(1000); Mcontrol=np.empty(1000); scontrol=np.empty(1000)
            original_bw_control=np.empty(1000); clipped_Rbw_control=np.empty(1000)
            for row in rf.to_dict('records'):
                k=int(row['fold']); ix=x.fold.to_numpy()==k; old=pf[pf['fold']==k-1].iloc[0]
                settings_match=all(row[f'{key}_{direction}']==value for direction in ('G','M') for key,value in
                                   [('regtype','lc'),('method','cv.ls'),('type','fixed'),('ckertype','gaussian'),('ckerorder',2)])
                if not settings_match:raise RuntimeError('Kernel configuration differs; do not attribute everything to bandwidth')
                Gi=nw(x.g_initial.to_numpy()[ix],b[ix],row['bandwidth_G'])
                Mi=nw(x.m_initial.to_numpy()[ix],a[ix],row['bandwidth_M'])
                fixed=corrected(a[ix],b[ix],Gi,Mi)
                Gcontrol[ix]=Gi;Mcontrol[ix]=Mi;scontrol[ix]=fixed['scores']
                Goriginal=nw(x.g_initial.to_numpy()[ix],b[ix],old['bandwidth_G'])
                Moriginal=nw(x.m_initial.to_numpy()[ix],a[ix],old['bandwidth_M'])
                original_bw_control[ix]=corrected(a[ix],b[ix],Goriginal,Moriginal)['scores']
                Gclipped=nw(x.g_initial.to_numpy()[ix],b[ix],np.clip(row['bandwidth_G'],old['bandwidth_low_G'],old['bandwidth_high_G']))
                Mclipped=nw(x.m_initial.to_numpy()[ix],a[ix],np.clip(row['bandwidth_M'],old['bandwidth_low_M'],old['bandwidth_high_M']))
                clipped_Rbw_control[ix]=corrected(a[ix],b[ix],Gclipped,Mclipped)['scores']
                item={'scenario':meta['scenario'],'replication':meta['replication'],'fold_python':k-1,'fold_R':k,
                      'selection_reason':meta['selection_reason'],'kernel_configuration_matches':settings_match,
                      'alpha_python':old['alpha'],'alpha_R':row['alpha'],'alpha_difference':row['alpha']-old['alpha'],
                      'beta_python':old['beta'],'beta_R':row['beta'],'beta_difference':row['beta']-old['beta'],
                      'alpha_fixed_Rbw_Python':fixed['diagnostics']['alpha'],'beta_fixed_Rbw_Python':fixed['diagnostics']['beta'],
                      'score_rmse_R_vs_Python':rms(ra.score.to_numpy()[ix]-x.score_python.to_numpy()[ix]),
                      'score_rmse_R_vs_fixed_Rbw_Python':rms(ra.score.to_numpy()[ix]-fixed['scores'])}
                for direction,predictor,response,control in [('G',x.g_initial.to_numpy()[ix],b[ix],Gi),('M',x.m_initial.to_numpy()[ix],a[ix],Mi)]:
                    pr=x[f'{direction}_python'].to_numpy()[ix]; rprediction=ra[f'{direction}_aux'].to_numpy()[ix]
                    hp,hr=old[f'bandwidth_{direction}'],row[f'bandwidth_{direction}']
                    cvp=np.mean((response-nw(predictor,response,hp,leave_one_out=True))**2)
                    cvr=np.mean((response-nw(predictor,response,hr,leave_one_out=True))**2)
                    item.update({f'bandwidth_{direction}_Python':hp,f'bandwidth_{direction}_R':hr,
                                 f'bandwidth_{direction}_difference':hr-hp,f'bandwidth_{direction}_R_over_Python':hr/hp,
                                 f'Python_boundary_{direction}':old[f'boundary_{direction}'],
                                 f'Python_boundary_side_{direction}':old[f'boundary_side_{direction}'],
                                 f'Python_search_lower_{direction}':old[f'bandwidth_low_{direction}'],
                                 f'Python_search_upper_{direction}':old[f'bandwidth_high_{direction}'],
                                 f'R_outside_Python_range_{direction}':bool(hr<old[f'bandwidth_low_{direction}'] or hr>old[f'bandwidth_high_{direction}']),
                                 f'aux_{direction}_rmse_R_vs_Python':rms(rprediction-pr),
                                 f'aux_{direction}_maxdiff_R_vs_fixed_Rbw_Python':float(np.max(abs(rprediction-control))),
                                 f'LOOCV_at_Python_h_{direction}':float(cvp),f'LOOCV_at_R_h_{direction}':float(cvr),
                                 f'R_reported_LOOCV_{direction}':row[f'cv_loss_{direction}'],
                                 f'LOOCV_gain_R_relative_{direction}':float((cvp-cvr)/max(cvp,1e-300)),
                                 f'R_LOOCV_reconstruction_error_{direction}':float(cvr-row[f'cv_loss_{direction}'])})
                fold_rows.append(item)
            original=score_test(x.score_python.to_numpy()); controlled=score_test(scontrol)
            clipped=score_test(clipped_Rbw_control)
            rsummary=score_test(ra.score.to_numpy()); T_R=rr['T_original']; T_py=original['T']
            assert abs(T_py-meta['T_python'])<1e-10
            diff=ra.score.to_numpy()-x.score_python.to_numpy()
            records.append({**meta,'status':'ok','T_R':T_R,'T_difference_R_minus_Python':T_R-T_py,
                            'T_Python_at_R_bandwidth':controlled['T'],'T_R_minus_Python_at_R_bandwidth':T_R-controlled['T'],
                            'T_Python_at_R_bandwidth_clipped_to_old_range':clipped['T'],
                            'T_change_due_to_allowing_R_bandwidth_outside_old_range':controlled['T']-clipped['T'],
                            'T_change_remaining_inside_old_range':clipped['T']-T_py,
                            'p_value_Python':original['p_value'],'p_value_R':rr['diagnostics']['p_value'],
                            'reject_Python':not original['accepted'],'reject_R':bool(rr['diagnostics']['p_value']<.05),
                            'decision_disagrees':bool((not original['accepted'])!=(rr['diagnostics']['p_value']<.05)),
                            'score_mean_Python':original['score_mean'],'score_mean_R':rr['diagnostics']['mean_score'],
                            'score_sd_Python':original['score_sd'],'score_sd_R':rr['diagnostics']['sd_score'],
                            'score_R_vs_Python_rmse':rms(diff),'score_R_vs_Python_max_abs_diff':float(np.max(abs(diff))),
                            'G_aux_rmse':rms(ra.G_aux-x.G_python),'M_aux_rmse':rms(ra.M_aux-x.M_python),
                            'score_R_vs_fixed_Rbw_Python_rmse':rms(ra.score-scontrol),
                            'score_R_vs_fixed_Rbw_Python_maxdiff':float(np.max(abs(ra.score-scontrol))),
                            'Python_original_bw_reconstruction_maxdiff':float(np.max(abs(original_bw_control-x.score_python))),
                            'R_variance_reconstruction_difference':rsummary['score_variance']-rr['diagnostics']['variance'],
                            'wrapper_matches_original':rr['wrapper_matches_original'],'wrapper_max_difference':rr['wrapper_max_difference'],
                            'R_warnings':json.dumps(rr['warnings']), 'nonfinite_count':int((~np.isfinite(np.r_[scontrol,ra.score,x.score_python])).sum()),
                            'R_version':rr['R_version'],'np_version':rr['np_version']})
            control_df=pd.DataFrame({'observation_id':x.observation_id,'G_at_R_bandwidth':Gcontrol,'M_at_R_bandwidth':Mcontrol,
                                     'score_at_R_bandwidth':scontrol,'score_R':ra.score,'score_Python':x.score_python})
            (ROOT/'controlled_python').mkdir(exist_ok=True)
            control_df.to_csv(ROOT/'controlled_python'/f'{stem}.csv',index=False,float_format='%.17g')
    df=pd.DataFrame(records);ff=pd.DataFrame(fold_rows)
    for i, row in df.iterrows():
        group=ff[(ff.scenario==row.scenario)&(ff.replication==row.replication)].sort_values('fold_R')
        for direction in ('G','M'):
            for implementation in ('Python','R'):
                df.loc[i,f'fold_bandwidths_{direction}_{implementation}']=json.dumps(group[f'bandwidth_{direction}_{implementation}'].tolist())
            df.loc[i,f'max_abs_bandwidth_difference_{direction}']=group[f'bandwidth_{direction}_difference'].abs().max()
        for coefficient in ('alpha','beta'):
            df.loc[i,f'max_abs_{coefficient}_difference']=group[f'{coefficient}_difference'].abs().max()
    exact_path=ROOT/'fixed_bandwidth_numeric_summary.csv'
    if exact_path.exists():
        exact=pd.read_csv(exact_path,float_precision='round_trip')
        df=df.merge(exact[['scenario','replication','T_R_largeh_off']],on=['scenario','replication'],validate='one_to_one')
        df['T_R_largeh_off_minus_fixed_Rbw_Python']=df.T_R_largeh_off-df.T_Python_at_R_bandwidth
        for i,row in df.iterrows():
            rr=pd.read_csv(ROOT/'r_fixed_bandwidth_exact'/f"{row.stem}.csv",float_precision='round_trip')
            pp=pd.read_csv(ROOT/'controlled_python'/f"{row.stem}.csv",float_precision='round_trip')
            df.loc[i,'score_maxdiff_largeh_off_vs_fixed_Rbw_Python']=float(np.max(abs(rr.score-pp.score_at_R_bandwidth)))
    df.to_csv(ROOT/'r_python_comparison.csv',index=False,float_format='%.17g')
    ff.to_csv(ROOT/'fold_level_comparison.csv',index=False,float_format='%.17g')
    aggregates=[]
    for scenario,ss in df.groupby('scenario',sort=False):
        valid=ss[ss.status=='ok'];d=valid.T_difference_R_minus_Python
        aggregates.append(dict(scenario=scenario,selected=len(ss),valid=len(valid),failed=len(ss)-len(valid),
                               mean_T_difference=float(d.mean()),rmse_T_difference=rms(d),max_abs_T_difference=float(abs(d).max()),
                               decision_disagreements=int(valid.decision_disagrees.sum()),
                               R_rejections=int(valid.reject_R.sum()),Python_rejections=int(valid.reject_Python.sum()),
                               max_abs_T_difference_same_R_bandwidth=float(abs(valid.T_R_minus_Python_at_R_bandwidth).max())))
    pd.DataFrame(aggregates).to_csv(ROOT/'comparison_summary.csv',index=False,float_format='%.17g')
    manifest=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
    unchanged=all(sha(pathlib.Path(p))==h for p,h in manifest['source_files_sha256'].items())
    assert unchanged
    evidence={'source_files_unchanged':unchanged,'checked_source_files':len(manifest['source_files_sha256']),
              'notebook_sha256':sha(nb),'frozen_python_sha256':sha(ROOT/'python_correction_frozen.py'),
              'selected':len(df),'valid':int((df.status=='ok').sum()),'decision_disagreements':int(df.decision_disagrees.sum()),
              'all_wrappers_match':bool(df.wrapper_matches_original.all()),
              'maximum_same_bandwidth_T_difference':float(abs(df.T_R_minus_Python_at_R_bandwidth).max()),
              'maximum_same_bandwidth_score_difference':float(df.score_R_vs_fixed_Rbw_Python_maxdiff.max()),
              'maximum_same_bandwidth_T_difference_largeh_off':float(abs(df.T_R_largeh_off_minus_fixed_Rbw_Python).max()) if 'T_R_largeh_off' in df else None,
              'maximum_same_bandwidth_score_difference_largeh_off':float(df.score_maxdiff_largeh_off_vs_fixed_Rbw_Python.max()) if 'T_R_largeh_off' in df else None,
              'nonfinite_count':int(df.nonfinite_count.sum())}
    (ROOT/'comparison_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(pd.DataFrame(aggregates).to_string(index=False));print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()

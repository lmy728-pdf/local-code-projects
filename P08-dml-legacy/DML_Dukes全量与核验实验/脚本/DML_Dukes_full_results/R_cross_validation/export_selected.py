"""Fix a diagnostic sample and export only existing saved predictions; no training."""
import csv,hashlib,json,pathlib
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
RUN=ROOT.parent/'n1000_R1000_f76477e6a5dc'
SCENARIOS=['gaussian','t2.1','lognormal2']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'inputs'; out.mkdir(exist_ok=True)
    config=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    raw=pd.read_csv(RUN/'raw_results.csv')
    folds=pd.read_csv(RUN/'fold_diagnostics.csv')
    c=raw[raw.method=='dukes_dr_gcm']
    rules={'seed_base':20261030,'per_scenario':10,'ordinary_random':4,'boundary_ranked':3,'large_abs_T_ranked':3,
           'order':'random first; then descending count of G/M boundary flags; then descending abs(T); remove earlier picks; ties use ascending replication',
           'purpose':'Implementation diagnostics only; selected sample cannot estimate population coverage',
           'scenario_order':SCENARIOS,'configured_before_R_results':True}
    if (ROOT/'selection_rules.json').exists():
        assert json.loads((ROOT/'selection_rules.json').read_text())==rules
    else:(ROOT/'selection_rules.json').write_text(json.dumps(rules,indent=2),encoding='utf-8')
    selected=[]; manifests={str(RUN/'config.json'):sha(RUN/'config.json')}
    for i,scenario in enumerate(SCENARIOS):
        cf=c[c.scenario==scenario].set_index('replication'); ff=folds[folds.scenario==scenario]
        counts=ff.assign(boundaries=ff.boundary_G.astype(int)+ff.boundary_M.astype(int)).groupby('replication').boundaries.sum()
        rng=np.random.default_rng(20261030+100*i)
        random_ids=sorted(int(x) for x in rng.choice(np.arange(1000),4,replace=False))
        picks=[(r,'random') for r in random_ids]; used=set(random_ids)
        boundary_ids=sorted((r for r in range(1000) if r not in used and counts[r]>0),key=lambda r:(-counts[r],r))[:3]
        picks += [(r,'boundary_ranked') for r in boundary_ids]; used.update(boundary_ids)
        tail_ids=sorted((r for r in range(1000) if r not in used),key=lambda r:(-abs(cf.loc[r,'T']),r))[:3]
        picks += [(r,'large_abs_T') for r in tail_ids]
        for rep,reason in picks:
            stem=f'{scenario}_{rep:04d}'
            path=RUN/'tasks'/f'{stem}.json'; task=json.loads(path.read_text(encoding='utf-8'))
            array=RUN/task['arrays_file']; assert sha(array)==task['arrays_sha256']
            manifests[str(path)]=sha(path);manifests[str(array)]=sha(array)
            with np.load(array,allow_pickle=False) as z:
                data={k:z[k].copy() for k in z.files}
            D=data['d_res']+data['g_hat']; transformed=data['b']+data['m_hat']
            frame=pd.DataFrame({'observation_id':np.arange(1,1001),'D':D,'y_transformed':transformed,
                                'g_initial':data['g_hat'],'m_initial':data['m_hat'],
                                'fold':data['fold_ids']+1,'G_python':data['G'],'M_python':data['M'],
                                'rD_python':data['rD'],'rY_python':data['rY'],'score_python':data['scores_C']})
            frame.to_csv(out/f'{stem}.csv',index=False,float_format='%.17g')
            pd.DataFrame(task['folds']).to_csv(out/f'{stem}_folds.csv',index=False,float_format='%.17g')
            selected.append({'scenario':scenario,'replication':rep,'selection_reason':reason,'boundary_count':int(counts[rep]),
                             'T_python':float(cf.loc[rep,'T']),'p_python':float(cf.loc[rep,'p_value']),
                             'dataset_seed':int(cf.loc[rep,'dataset_seed']),'crossfit_seed':int(cf.loc[rep,'crossfit_seed']),
                             'r_auxiliary_seed':20261030+100000*i+rep,'stem':stem,
                             'D_residual_roundtrip_maxdiff':float(np.max(abs((D-data['g_hat'])-data['a']))),
                             'Ytheta_residual_roundtrip_maxdiff':float(np.max(abs((transformed-data['m_hat'])-data['b']))),
                             'input_sha256':sha(out/f'{stem}.csv'),'array_sha256':sha(array)})
    pd.DataFrame(selected).to_csv(ROOT/'selected_samples.csv',index=False,float_format='%.17g')
    manifest={'run':str(RUN),'config_hash':config['config_hash'],'author_commit':'e7dfb038117aea3d95ca1371e2e85557ff9f25d8',
              'source_files_sha256':manifests,'data_reconstruction':'D=d_res+g_hat; Y-thetaD=b+m_hat; no HGBR refit, no new random sample',
              'python_fold_ids':'0..4','R_fold_ids':'1..5','observation_order':'unchanged; observation_id 1..1000'}
    (ROOT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(selected)[['scenario','replication','selection_reason','boundary_count','T_python']].to_string(index=False))
if __name__=='__main__':main()

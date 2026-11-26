# -*- coding: utf-8 -*-
"""只完善结果展示与交付核验，不重新估计任何方法。"""
from pathlib import Path
import sys,html,json,sqlite3,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import rank_worker as m
def main():
    comparison=pd.read_csv(ROOT/'comparison_historical.csv')
    m.module(0).configure_plot_style()
    colors={'M1':'#52759b','M2':'#d48629','M3':'#8866ab','S':'#00847d','Rank':'#c73350'}
    panels=[('coverage_pct','Coverage (%)',95),('bias','Bias',0),('rmse','RMSE',None),
            ('mean_T','Mean(T)',0),('sd_T','SD(T)',1),('deterioration_pp','覆盖下降 (pp)',0)]
    for d in range(2):
        fig,axes=m.plt.subplots(3,2,figsize=(12,11),layout='constrained')
        for ax,(metric,label,target) in zip(axes.ravel(),panels):
            for method in m.METHODS:
                a=comparison[(comparison.dgp_id==d)&comparison.method.eq(method)].set_index('error_id').reindex([0,7,14])
                ax.plot(range(3),a[metric],marker='o',label=method+(' Pilot R200' if method=='Rank' else ' Historical R1000'),color=colors[method])
                if method=='Rank':
                    for x,y in enumerate(a[metric]):
                        ax.annotate(f'{y:.3g}',(x,y),xytext=(-5 if x==2 else 4,7),textcoords='offset points',
                                    ha='right' if x==2 else 'left',fontsize=9)
            if target is not None:ax.axhline(target,color='gray',ls=':',lw=1)
            ax.set_xticks(range(3),list(m.ERRORS.values()));ax.set_ylabel(label);ax.grid(alpha=.2);ax.margins(x=.08,y=.2)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle(f'DGP{d+1} '+('线性' if d==0 else '非线性')+' · Wilcoxon Rank-DML 独立 Pilot')
        fig.savefig(ROOT/f'DGP{d+1}_对照图.png',dpi=180);m.plt.close(fig)
    notes=(ROOT/'Pilot结论.md').read_text(encoding='utf-8')
    report=(ROOT/'中文报告.html').read_text(encoding='utf-8')
    marker='<h2>本次 Pilot 结论</h2>'
    if marker not in report:
        report=report.replace('<h2>结果解读</h2>',marker+'<pre>'+html.escape(notes)+'</pre><h2>结果解读</h2>')
        (ROOT/'中文报告.html').write_text(report,encoding='utf-8')
    config=json.loads((ROOT/'configuration.json').read_text(encoding='utf-8'))
    assert m.sha((ROOT/'rank_worker.py').read_bytes())==config['worker_hash']
    for path,h in config['source_hashes'].items():assert m.sha(Path(path).read_bytes())==h
    with sqlite3.connect(ROOT/'checkpoint'/'replications.sqlite3') as db:
        assert db.execute('PRAGMA integrity_check').fetchone()==('ok',)
        counts=db.execute('SELECT d,e,COUNT(*) FROM results GROUP BY d,e').fetchall()
        assert len(counts)==6 and all(x[2]==200 for x in counts)
    raw=pd.read_csv(ROOT/'replication_results.csv');s=pd.read_csv(ROOT/'summary.csv')
    assert len(raw)==1200 and not raw.duplicated(['dgp_id','error_id','replication_id']).any()
    for _,r in s.iterrows():
        a=raw[(raw.dgp_id==r.dgp_id)&(raw.error_id==r.error_id)]
        assert np.isclose(r.coverage_pct,100*(abs(a['T'])<=1.96).mean())
        assert np.isclose(r.bias,(a.theta_hat-1).mean())
        assert np.isclose(r.rmse,np.sqrt(np.mean((a.theta_hat-1)**2)))
    assert raw.matched_historical.all() and raw.optimizer_converged.all()
    receipt=dict(passed=True,checkpoint_integrity=True,scenarios=6,replications=1200,point_success=1200,
                 test_success=1200,matched_data_folds_seeds_predictions=1200,old_sources_unchanged=True,
                 summary_independently_recomputed=True)
    m.atomic_json(ROOT/'delivery_validation.json',receipt)
    files='\n'.join(str(f) for f in ROOT.rglob('*') if f.is_file() and '__pycache__' not in str(f))
    (ROOT/'文件清单.txt').write_text(files,encoding='utf-8')
    print(json.dumps(receipt))
if __name__=='__main__':main()


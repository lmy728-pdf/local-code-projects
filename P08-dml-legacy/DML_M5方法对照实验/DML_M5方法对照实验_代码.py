"""M4 / M5-A / M5-B 对比：直接运行 python M5_comparison.py。
依赖 numpy, scipy, scikit-learn (原环境 1.7.2), matplotlib, tqdm。
来源：桌面 DML_HeavyTail_Benchmark/DML_p5_student_t.ipynb 的 HGBR_PARAMS、
DGP、seed、generate_observed、learner、ols_point、m4；复制到此以便独立运行。
"""
import os
# 单个进程只使用一个计算线程；可用环境变量 M5_WORKERS 降低 CPU 占用。
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import hashlib
from pathlib import Path
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import brentq
from tqdm import tqdm

HGBR_PARAMS = {'categorical_features': 'from_dtype',
 'early_stopping': 'auto',
 'interaction_cst': None,
 'l2_regularization': 1,
 'learning_rate': 0.05,
 'loss': 'squared_error',
 'max_bins': 255,
 'max_depth': None,
 'max_features': 1.0,
 'max_iter': 100,
 'max_leaf_nodes': 15,
 'min_samples_leaf': 20,
 'monotonic_cst': None,
 'n_iter_no_change': 10,
 'quantile': None,
 'random_state': 0,
 'scoring': 'loss',
 'tol': 1e-07,
 'validation_fraction': 0.1,
 'verbose': 0,
 'warm_start': False}
DGP_NAMES = ('DGP1_linear', 'DGP2_nonlinear')
T_GRID = (20.,10.,5.,3.,2.6,2.3,2.1)  # 保留原 error_id；本实验只运行 0 和 7
SOLVER = {'min_denominator': 1e-10, 'kappa':1.345}
CONFIG = dict(n=1000, p=5, mode='full', base_seed=20261103, spec={'learner':HGBR_PARAMS})
R = 200
METHODS = ('M4','M5-A','M5-B')
NumericalFailure = ValueError

def ar1_covariance(p: int, rho: float = 0.5) -> np.ndarray:
    """构造 Sigma[j,k] = rho**|j-k| 的 AR(1) 型协方差矩阵。"""
    index = np.arange(p)
    return rho ** np.abs(index[:, None] - index[None, :])


def nuisance_functions(X: np.ndarray, dgp_name: str):
    """返回指定 DGP 的 m0(X) 与 g0(X)。"""
    x1, x2, x3, x4, x5 = X.T

    if dgp_name == "DGP1_linear":
        m0 = 0.80 * x1 - 0.50 * x2 + 0.30 * x3 + 0.20 * x4 - 0.10 * x5
        g0 = 1.00 * x1 + 0.50 * x2 - 0.50 * x3 + 0.25 * x4 + 0.10 * x5
    elif dgp_name == "DGP2_nonlinear":
        # x2**2 - 1 做中心化，因为标准正态变量满足 E[X^2]=1
        m0 = (
            0.60 * np.sin(x1)
            + 0.35 * (x2**2 - 1.0)
            + 0.25 * x3 * x4
            - 0.20 * np.tanh(x5)
        )
        g0 = (
            1.20 * np.sin(x1)
            + 0.70 * (x2**2 - 1.0)
            + 0.50 * x3 * x4
            + 0.30 * np.tanh(x5)
        )
    else:
        raise ValueError(f"未知 DGP: {dgp_name}")
    return m0, g0


def standardized_student_t(rng: np.random.Generator, df: float, size: int) -> np.ndarray:
    """生成均值 0、方差 1 的 Student-t 误差；要求 df > 2。"""
    if df <= 2:
        raise ValueError("Student-t 自由度必须大于 2，才能标准化为单位方差。")
    return rng.standard_t(df, size=size) * np.sqrt((df - 2.0) / df)


def seed(config, dgp, error, rep, purpose, fold=0, learner=0):
    key = [config['base_seed'], {'smoke': 1, 'full': 2}[config['mode']],
           config['p'], dgp, config['n'], error, rep, purpose, fold, learner]
    return int(np.random.SeedSequence(key).generate_state(1)[0])


def generate_observed(config, dgp, error, rep):
    n = config['n']
    seeds = {name: seed(config, dgp, error, rep, purpose)
             for name, purpose in [('X', 1), ('v', 2), ('epsilon', 3), ('folds', 4)]}
    X = np.random.default_rng(seeds['X']).multivariate_normal(
        np.zeros(5), ar1_covariance(5), n)
    m, g = nuisance_functions(X, DGP_NAMES[dgp])
    v = np.random.default_rng(seeds['v']).normal(size=n)
    rng = np.random.default_rng(seeds['epsilon'])
    epsilon = rng.normal(size=n) if error == 0 else standardized_student_t(rng, T_GRID[error-1], n)
    D = m + v
    Y = D + g + epsilon  # theta_true=1 is confined to generation/evaluation.
    assert X.shape == (n, 5) and np.isfinite(np.r_[X.ravel(), D, Y]).all()
    folds = list(KFold(5, shuffle=True, random_state=seeds['folds']).split(X))
    validate_folds(folds, n)
    assert n % 5 == 0
    fold_ids = np.empty(n, dtype=np.int64)
    for k, (_, test) in enumerate(folds):
        fold_ids[test] = k
    audit = dict(seeds=seeds, data_hash=sha(X.tobytes()+D.tobytes()+Y.tobytes()),
                 fold_hash=sha(fold_ids.tobytes()),
                 epsilon_sample_variance=float(np.var(epsilon, ddof=1)))
    return (X, D, Y), folds, fold_ids, audit


def validate_folds(folds, n):
    """Validate full complementary train/test partitions with equal test sizes."""
    clean, seen, sizes = [], np.zeros(n, dtype=int), []
    for k, (train, test) in enumerate(folds):
        train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
        if (train.ndim != 1 or test.ndim != 1 or len(test) < 2 or
                len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test) or
                np.any(train < 0) or np.any(train >= n) or np.any(test < 0) or np.any(test >= n) or
                np.intersect1d(train, test).size or
                not np.array_equal(np.sort(np.r_[train, test]), np.arange(n))):
            raise NumericalFailure("invalid_folds", "Each training set must be the test complement", k)
        seen[test] += 1
        sizes.append(len(test))
        clean.append((train, test))
    if not clean or np.any(seen != 1) or len(set(sizes)) != 1:
        raise NumericalFailure("invalid_folds", "Equal-sized test folds must partition observations")
    return clean


def sha(value):
    return hashlib.sha256(value).hexdigest()


def learner(config, random_state, median=False):
    params = config['spec']['learner'].copy()
    # sklearn 1.7 requires a float here; JSON conversions can erase .0.
    params['max_features'] = float(params['max_features'])
    params.update(random_state=random_state, loss='quantile' if median else 'squared_error',
                  quantile=.5 if median else None)
    return HistGradientBoostingRegressor(**params)


def ols_point(w, u):
    J = float(np.mean(w*w))
    if J <= SOLVER['min_denominator']:
        raise ArithmeticError('degenerate_OLS_denominator')
    return float(np.mean(w*u)/J)


def m4(w, u):
    initial = ols_point(w, u)
    r0 = u-initial*w
    s = float(1.4826*np.median(abs(r0-np.median(r0))))
    if not np.isfinite(s) or s <= 1e-12:
        raise ArithmeticError('degenerate_Huber_MAD')
    tau = SOLVER['kappa']*s
    def equation(t):
        return float(np.mean(w*np.clip(u-t*w, -tau, tau)))
    width = 1.
    for _ in range(40):
        if equation(initial-width) >= 0 >= equation(initial+width):
            break
        width *= 2
    else:
        raise ArithmeticError('Huber_bracket_failed')
    t = float(brentq(equation, initial-width, initial+width, xtol=1e-12, rtol=1e-14))
    r = u-t*w
    q = w*np.clip(r, -tau, tau)
    A = float(np.mean(w*w*(abs(r) < tau)))
    if abs(q.mean()) > 1e-9:
        raise ArithmeticError('Huber_score_residual')
    out = dict(theta_hat=t, se_type='Huber_plugin_ddof1_fixed_tau', ci_type='Huber_plugin_Wald',
               diagnostics=dict(scale=s, tau=tau, A=A, score_error=float(abs(q.mean())),
                                clipping=float(np.mean(abs(r)>tau)),
                                boundary_count=int(np.isclose(abs(r), tau, atol=1e-10, rtol=0).sum()),
                                objective=float(np.mean(np.where(abs(r)<=tau, .5*r*r, tau*abs(r)-.5*tau*tau)))))
    out['se'] = float(np.std(q/A, ddof=1)/np.sqrt(len(w))) if A > 1e-12 else None
    return out


def replication(job):
    d,e,r = job
    (X,D,Y),folds,_,_ = generate_observed(CONFIG,d,e,r)
    # 四组折外预测；每个目标在两种 loss 下的 seed 完全相同。
    pred = np.full((2,2,len(Y)),np.nan)
    for k,(tr,te) in enumerate(folds):
        for target_j,target in enumerate((D,Y)):
            model_seed = seed(CONFIG,d,e,r,5,k,target_j)
            for median in (0,1):
                model = learner(CONFIG,model_seed,bool(median))
                pred[target_j,median,te] = model.fit(X[tr],target[tr]).predict(X[te])
    if not np.isfinite(pred).all():
        raise ArithmeticError(f"非有限折外预测 {job}")
    out=[]
    for method,(dy,yy) in zip(METHODS,((0,0),(0,1),(1,1))):
        w,u = D-pred[0,dy], Y-pred[1,yy]
        fit = m4(w,u)  # 估计函数只接收残差，不接收真值
        t,se = fit['theta_hat'],fit['se']
        if se is None or not np.isfinite([t,se]).all() or se<=0:
            raise ArithmeticError(f"{job} {method}: invalid Huber estimate/SE")
        out.append((t,se,t-1.96*se,t+1.96*se))
    return job,out


def summarize(raw):
    # raw 的轴是 DGP、分布、重复、方法、估计量字段；仅存在内存中。
    t,se,lo,hi=np.moveaxis(raw,-1,0)
    err=t-1.0
    cov=((lo<=1.0)&(1.0<=hi)).mean(axis=2)*100
    return (err.mean(axis=2), np.sqrt((err**2).mean(axis=2)), cov,
            se.mean(axis=2),(hi-lo).mean(axis=2),cov[:,0]-cov[:,1])


def render(stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ('Microsoft YaHei','SimHei','Noto Sans CJK SC'):
        if name in font_manager.get_font_names():
            plt.rcParams['font.family']=name
            break
    plt.rcParams.update({'axes.unicode_minus':False,'savefig.facecolor':'#fafaf8'})
    folder=Path(__file__).resolve().parent
    bias,rmse,cov,se,length,loss=stats
    colors=['#365572','#0a766f','#aa5132']
    dgps=['DGP1 线性','DGP2 非线性']
    errors=['Gaussian','t(2.1)']
    headers=['DGP','误差','方法','Bias','RMSE','Coverage (%)','Mean SE','平均区间长度','退化 (pp)']
    rows=[]
    for d in range(2):
        for m in range(3):
            for e in range(2):
                rows.append([dgps[d],errors[e],METHODS[m],f'{bias[d,e,m]:+.4f}',
                    f'{rmse[d,e,m]:.4f}',f'{cov[d,e,m]:.1f}',f'{se[d,e,m]:.4f}',
                    f'{length[d,e,m]:.4f}',f'{loss[d,m]:+.1f}' if e else '—'])
    print('\n'+' | '.join(headers),flush=True)
    for row in rows: print(' | '.join(row),flush=True)
    fig,ax=plt.subplots(figsize=(13,7.6),facecolor='#fafaf8'); ax.axis('off')
    fig.subplots_adjust(left=.032,right=.968,bottom=.08,top=.87)
    fig.text(.035,.936,'第一阶段损失与 Huber-DML  |  指标对照',size=23,weight='bold',color='#18334a')
    fig.text(.036,.887,'n = 1,000    ·    p = 5    ·    5 折交叉拟合    ·    每组 200 次重复    ·    95% Wald 区间',size=12,color='#526473')
    table=ax.table(cellText=rows,colLabels=headers,cellLoc='center',bbox=[0,.03,1,.95],
                   colWidths=[.136,.105,.075,.10,.10,.13,.10,.14,.114])
    table.auto_set_font_size(False);table.set_fontsize(10.5)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#25435b' if i==0 else ('#edf2f4' if ((i-1)//2)%2==0 else '#fbfbfa'))
        if not i: cell.set_text_props(color='white',weight='bold')
        elif j==2: cell.set_text_props(color=colors[((i-1)//2)%3],weight='bold')
    fig.text(.036,.04,'退化 = Gaussian 覆盖率 − t(2.1) 覆盖率；单位为百分点。正值表示重尾下覆盖率下降。',size=11,color='#526473')
    fig.savefig(folder/'M5_对照表.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),facecolor='#fafaf8')
    fig.subplots_adjust(left=.068,right=.985,bottom=.08,top=.79,hspace=.50,wspace=.27)
    fig.text(.055,.934,'均值与中位数第一阶段：覆盖率、误差与退化',size=23,weight='bold',color='#18334a')
    fig.text(.056,.878,'方法仅改变 Y / D 的第一阶段损失，第二阶段固定为无截距对称 Huber。',size=12,color='#526473')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(colors,['M4 · 均值 / 均值','M5-A · 中位数 / 均值','M5-B · 中位数 / 中位数'])],
               loc='upper left',bbox_to_anchor=(.05,.867),ncol=3,frameon=False)
    for d in range(2):
        for j,(title,values) in enumerate([('Coverage (%)',cov),('RMSE',rmse),('覆盖率退化 (pp)',loss)]):
            ax=axes[d,j]; ax.set_facecolor('#fafaf8');ax.set_axisbelow(True)
            ax.grid(axis='y',color='#dce3e5',linewidth=.7); ax.spines[['top','right','left']].set_visible(False)
            ax.set_title(dgps[d]+'   /   '+title,loc='left',fontsize=12,weight='bold')
            for m in range(3):
                if j<2:
                    bars=ax.bar(np.arange(2)+(m-1)*.26,values[d,:,m],.24,color=colors[m])
                else: bars=ax.bar([m],[values[d,m]],.64,color=colors[m])
                ax.bar_label(bars,fmt='%.1f' if j!=1 else '%.4f',padding=2,fontsize=8)
            if j==0:
                ax.set_ylim(0,107);ax.axhline(95,ls='--',lw=.9,color='#465c63')
            elif j==1: ax.set_ylim(0,rmse.max()*1.23)
            else:
                span=max(np.ptp(loss),abs(loss).max(),1)
                ax.set_ylim(min(0,loss.min())-.12*span,max(0,loss.max())+.18*span)
                ax.axhline(0,color='#839198',lw=.7)
            ax.set_xticks(range(2) if j<2 else range(3),errors if j<2 else METHODS)
    fig.savefig(folder/'M5_对照图.png',dpi=170);plt.close(fig)


def main():
    jobs=[(d,e,r) for r in range(R) for d in range(2) for e in (0,7)]
    raw=np.full((2,2,R,3,4),np.nan)
    workers=max(1,min(int(os.environ.get('M5_WORKERS',16)),os.cpu_count() or 1))
    print(f'800 份数据，2400 次方法评价；{workers} 个进程，每个进程一个计算线程。',flush=True)
    with mp.get_context('spawn').Pool(workers) as pool:
        for (d,e,r),row in tqdm(pool.imap_unordered(replication,jobs,chunksize=1),
                total=len(jobs),desc='M4/M5 配对实验',unit='rep',mininterval=8,ascii=True):
            raw[d,int(e==7),r]=row
    if not np.isfinite(raw).all(): raise RuntimeError('实验不完整，不导出结果')
    render(summarize(raw))

if __name__=='__main__':
    mp.freeze_support()
    main()


"""Format the completed read-only audit; never touches experiment inputs."""
import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
def read(name):
    with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
v=json.loads((P/'verification.json').read_text(encoding='utf-8'))
stats=read('standardized_statistics_summary.csv'); pred=read('prediction_differences_summary.csv'); folds=read('fold_verification.csv')
locations={r['function']:r for r in read('code_locations.csv')}
def loc(name):
    r=locations[name]
    return f"第{r['notebook_cell_1based']}单元（第{r['code_cell_1based']}个代码单元）`{name}`，单元行{r['cell_line_start']}–{r['cell_line_end']}；[快照L{r['snapshot_line']}]({(P/'notebook_source_snapshot.py').as_posix()}:{r['snapshot_line']})"
e=v['max_absolute_errors']
lines=['# A/B/C 实现审计与标准化统计量诊断','',
       '审计对象：桌面 `DML_Dukes_n1000_R1000.ipynb`；结果 `n1000_R1000_f76477e6a5dc`。三场景各1000次、每次n=1000，共3000数据集、9000方法记录。',
       '', '本轮只读取已有结果并重算诊断，未训练模型、未生成新数据、未重新选择带宽，未修改notebook、DGP、HGBR、种子或原结果。磁盘notebook无保存单元输出，但其计算函数AST哈希与本次run记录完全相同；全部输入文件在审计前后SHA-256不变，见 `input_manifest.json`。',
       '',f"结论：A/B的处理预测相同，但变换结果预测不同；B/C初始预测相同。C没有发现遗漏两个减项、折内更新或合并方差的错误。独立核验覆盖{v['folds']}折、{v['kernel_fits_verified']}个固定带宽核拟合；超出预定容差的问题数为{len(v['problems'])}。这不等于复现作者R/np带宽优化。",
       '', '## 1. A/B/C 的数据、fold 与 nuisance predictions','',
       '记 `g_hat=预测D|X`，`ell_hat=预测Y|X`，`m_hat=直接预测(Y−θD)|X`。A用于同目标比较的预测为 `m_A=ell_hat−θ*g_hat`，B用 `m_B=m_hat`。总体条件期望具有线性恒等式，但重新训练的有限样本HGBR预测不必满足该恒等式。',
       '', '| 核查 | 代码位置 | 结果 |','|---|---|---|',
       f"| 同一数据、五折、种子 | {loc('make_data')}；{loc('compute_task')} | 每份数据只生成一次，统一传入A/B/C；全部3000份fold标签与同seed重新构造的KFold逐元素一致，方法记录的seed/fold_hash一致 |",
       f"| A的初始预测 | {loc('initial_a')} | 在train上分别拟合D和Y，在test上预测；未训练内预测冒充OOF |",
       f"| B的初始预测 | {loc('crossfit_initial')} | `response=Y-theta*D`；复用A传入的`treatment_predictions=g_hat`，仅另训变换结果模型；超参、fold、模型seed相同 |",
       f"| B/C共享预测 | {loc('compute_task')} | B使用initial[g_hat/m_hat]，同一对象传入C；保存Bscore与C初始a*b完全一致 |",'',
       f"数据证据：处理残差 `d_res−a` 最大差{e['A_B_treatment_residual']:.3g}；从残差和预测重建D的最大差{e['A_B_reconstructed_D']:.3g}，重建Y的最大差{e['A_B_reconstructed_Y']:.3g}。X未单独保存在数组里，其共用及训练折隔离由已匹配的计算代码确认，本轮没有重新拟合来验证预测值。",'',
       '下表差值定义为 `m_B−(ell_hat−g_hat)`，θ=1。RMSE对每场景1000×1000个保存预测值合并计算。', '',
       '| 场景 | D侧最大差 | 结果侧合并RMSE | 结果侧最大绝对差 | 结果侧完全一致的数据集 |','|---|---:|---:|---:|---:|']
for r in pred:
    lines.append(f"| {r['scenario']} | {float(r['D_prediction_maxdiff']):.3g} | {float(r['pooled_outcome_prediction_rmse']):.6f} | {float(r['max_outcome_prediction_difference']):.6f} | {r['identical_outcome_prediction_datasets']}/1000 |")
lines += ['', '因此A→B同时改变结果学习目标和检验标准化方式，不能将A、B当作同一有限样本方法。B→C才隔离新增修正步骤。本轮报告此差异，没有把B改成A。', '',
          '## 2. Dukes Section 3.3（纸面第10—12页）逐项核验','',
          '[正式论文](https://jmlr.org/papers/volume25/22-1233/22-1233.pdf)；[作者rdml_cross.test](https://github.com/ordukes/DRInference/blob/e7dfb038117aea3d95ca1371e2e85557ff9f25d8/dri_functions.r)。以下a=D−g_hat，b=Y−θD−m_hat。', '',
          '| 步骤与公式 | 代码位置 | 结论及数值证据 |','|---|---|---|',
          f"| 初始拟合在I_k补集；辅助拟合在I_k | {loc('validate_folds')}；{loc('crossfit_initial')}；{loc('dukes_from_initial')} | 五个200人验证折及800人训练补集；初始仅train训练；核回归及更新仅test |",
          f"| G_i=Σ_j K((g_i−g_j)/h_G)b_j / Σ_j K；M_i=Σ_j K((m_i−m_j)/h_M)a_j / Σ_j K，j∈I_k | {loc('dukes_from_initial')}；{loc('_nw_from_distances')} | 方向正确。独立Gaussian权重重算G/M最大差{e['kernel_G']:.3g}/{e['kernel_M']:.3g} |",
          f"| 带宽CV与最终拟合 | {loc('select_nw')} | LOOCV排自身，最终拟合含自身；已存带宽处LOOCV损失独立重算最大差G={e['G_LOOCV_at_saved_h']:.3g}、M={e['M_LOOCV_at_saved_h']:.3g}。未重新优化带宽 |",
          f"| α_k=ΣGa/ΣG²；β_k=ΣMb/ΣM²；rD=a−αG，rY=b−βM | {loc('stable_projection')}；{loc('corrected_score')} | 折内无截距更新。系数最大差α={e['alpha']:.3g}、β={e['beta']:.3g}；max绝对更新矩G={e['mean_G_rD']:.3g}、M={e['mean_M_rY']:.3g} |",
          f"| ψ*=rD·rY−G·rD−M·rY | {loc('corrected_score')} | 两个减项均在逐点score中。完整score独立重算最大差{e['C_full_score']:.3g} |",
          f"| 按原观测次序回填，先合并再标准化 | {loc('dukes_from_initial')}；{loc('score_test')} | 等大折均值的平均与全样本均值最大差{e['fold_mean_vs_pooled']:.3g}；未平均各折T或p |",
          f"| s²=Σ(ψ*−均值)²/(n−1)，T=√n·均值/s | {loc('score_test')} | 方差保留全部修正项；方差最大差{e['C_variance']:.3g}，T最大差{e['C_T']:.3g}，p最大差{e['C_p']:.3g} |",'',
          '两项必须披露的差异：①论文第11页Step 6分母印作方差/√n，与该节标准差标准化说明不一致；代码使用标准差，与作者R检验一致，按疑似排版问题处理。②论文第12页方差分母为n，当前按作者R `var` 使用n−1，T相对ddof=0缩小√(999/1000)，约0.05%。均未在此次修改。', '',
          '带宽优化并非作者npregbw的逐项复现：当前为固定Gaussian核、41点有限网格加局部细化，作者采用R/np优化器；常数索引的退化分支也存在实现约定差异。核公式与更新核验通过不能消除这些软件差异。','',
          '| 场景 | G/M带宽命中边界 / 总数 | full-score方差 / 仅乘积方差（每次比值的均值） |','|---|---:|---:|']
for r in pred:
    fs=[f for f in folds if f['scenario']==r['scenario']]
    boundary=sum(f[k]=='True' for f in fs for k in ['boundary_G','boundary_M'])
    lines.append(f"| {r['scenario']} | {boundary}/{2*len(fs)} | {float(r['mean_full_to_product_variance_ratio']):.6f} |")
lines += ['', '虽然两个减项的折内样本均值接近零，逐点值不为零，删除它们会改变方差。每折减项RMS、更新矩与带宽边界见fold_verification.csv。','',
          '## 3. θ₀=1 的标准化检验统计量','',
          f"A：T_A=(theta_hat−1)/SE_A，与原1.96 Wald区间严格对应，SE在theta_hat处按旧影响函数计算，不替换成原假设处score SE。代码：{loc('fit_mean_dml')}。B/C：T=√n·mean(score)/SD(score,ddof=1)，代码：{loc('score_test')}。",'',
          'A原记录未单独存T，本轮从已保存估计和SE重建，并独立由残差核验；B/C原记录已存T，本轮从score重算再核对。原始CSV与JSON不改写，9000条诊断T另存standardized_statistics.csv。SD(T)是1000次重复之间的样本标准差（ddof=1），不是单次score的标准差。', '',
          '| 场景 | 方法 | Mean(T) | SD(T) | P(T<−1.96) | P(T>1.96) | P(绝对T≤1.96) | 有效/非有限/失败 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
for r in stats:
    lines.append(f"| {r['scenario']} | {r['method']} | {float(r['mean_T']):.6f} | {float(r['sd_T']):.6f} | {float(r['p_T_lt_minus1_96']):.3%} | {float(r['p_T_gt_1_96']):.3%} | {float(r['p_abs_T_le_1_96']):.3%} | {r['valid']}/{r['nonfinite']}/{r['failed']} |")
lines += ['',f"原记录含警告的方法数：{sum(int(r['warning_records']) for r in stats)}。本轮独立重算超出容差的问题数：{len(v['problems'])}。统一按用户要求用1.96；B/C原临界值norm.ppf(.975)=1.9599639845，与统一阈值产生不同判定的记录数为{sum(int(r['critical_value_difference_count']) for r in stats)}。",'',
          '这批结果的九组Mean(T)均明显大于0，左尾拒绝均为0，拒绝全部来自右尾；SD约0.88至1.08。欠覆盖主要表现为统计量整体向右偏移，不能单用方差估计偏小解释，也不能仅凭此确定偏差的具体来源。C相对B减小均值右移，但幅度不足，且C的均值仍高于A。单凭均值/SD也不能证明正态性。A/B差异不是只换一个SE，C与B比较才对应修正步骤。当前代码的结构核验及保存结果重算没有发现漏项，但作者R带宽交叉核验仍未完成；本轮不据此宣称原论文方法失效，也不追加其他实验。', '',
          '## 可追溯文件与复核','',
          '- `standardized_statistics.csv`：逐次T、左右尾指示、覆盖指示、有效状态、配置/数组哈希和来源任务名。',
          '- `standardized_statistics_summary.csv`：按场景/方法汇总、有效数、非有限值、异常及Mean(T)的MCSE。',
          '- `prediction_differences.csv`、`prediction_differences_summary.csv`：每次与按场景的预测差异。',
          '- `fold_verification.csv`：15000折的更新矩、减项RMS、系数、带宽与边界。',
          '- `verification.json`、`input_manifest.json`：数值误差、代码对应性和输入完整性证据。',
          '- `notebook_source_snapshot.py`、`code_locations.csv`：原代码只读快照和单元/行号映射；快照含原启动单元，勿将其作为脚本运行。',
          '- `audit_existing.py`：独立诊断脚本，不导入/执行notebook或原模拟入口。', '',
          '复核命令（只诊断已保存数组）：', '```powershell',
          f"& 'D:\\Anoconda\\envs\\environment2\\python.exe' '{P / 'audit_existing.py'}'", '```','']
(P/'audit_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(P/'audit_report.md')

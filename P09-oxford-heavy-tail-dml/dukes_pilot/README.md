# 给 Codex 的任务书：在现有 DML 实验中加入 Dukes DR inference，先做小规模覆盖率筛查

> 本文件可直接交给 Codex 执行。先审计原项目，再新增独立实现，最后运行最小筛查。不要直接执行原 notebook 的全部单元，不要覆盖旧代码和结果。
>
> 编写日期：2026-10-30。本任务书已只读核对用户的原论文、两个 `p=5` 主实验 notebook、旧 README，以及 Dukes 等人的 JMLR 正式论文和作者代码。**本次交付仅为任务书，尚未实现或运行 Dukes 模拟，不含新实验结果。**

## 1. 研究目的与本轮范围

判断 Dukes, Vansteelandt & Whitney (2024) 的 doubly robust inference 是否已明显缓解原实验中的 coverage 问题，为是否继续当前 DML 毕设选题提供依据。

- 如果现有方法已经在关注场景中表现良好，用户倾向于更换选题，考虑导师的 M-estimation 方向；不要自行将任务转为“继续研究 Dukes 并投稿”。
- 如果仍有明显欠覆盖，先排查实现与比较口径，再报告剩余问题；不要自动开始 Huber。
- 如果结果不明确，说明不确定性以及最小的后续验证需求，不要自动扩为完整研究。

**默认只做：DGP2，`n=1000`，`p=5`，Gaussian、Student-t(2.1)、centered lognormal(σ=2)，每场景 `R=200`，原五折交叉拟合，原 HGBR。**先做每场景 2 次的冒烟检查，`n` 仍为 1000；检查通过后继续到预先固定的 200 次，不根据中间结果提前挑选停止点。冒烟重复可复用为正式筛查的前 2 次。

本轮不扩 `p`，不更换 learner，不做 Huber/Median-DML/RF 大比较，不运行完整误差网格，不默认求 Dukes 完整置信集或点估计。`n` 控制每次样本量，`R` 控制 Monte Carlo 测量精度；渐近理论本身不要求先把 `n` 改成 5000。

## 2. 已定位材料与依据

### 2.1 用户原项目

原项目根目录：

```text
C:/Users/Lenovo/Desktop/项目文件/2026夏牛津项目/DML论文
```

| 材料 | 原项目中的位置 | 本任务用途 |
|---|---|---|
| 用户论文 | `A Study on the Robustness of Double Machine Learning under Heavy-Tailed Errors (2).docx` | Section 2 的 DGP、误差、估计式；Appendix A.1/A.2 的主基准与文末实现参数 |
| 项目说明 | `README.md` | 实验组织与旧输出行为；以实际代码和论文交叉核对 |
| Student-t 主实验 | `code/DML_p5_student_t_heavytail.ipynb` | 复用 DGP、Mean-DML、seed 和 folds |
| Lognormal 主实验 | `code/DML_p5_lognormal_heavytail.ipynb` | 复用总体中心化、标准化和场景编号 |
| Gaussian learner 比较 | `code/DML_p5_gaussian_RF_vs_HGBR.ipynb` | 仅作参考；不是本轮主 Gaussian 的 seed 来源 |
| 旧结果 | `resaults/`（原目录就是这个拼写）与 notebook 已保存输出 | 只读验证，不覆盖 |

如果在另一台电脑运行，请先定位同名材料并更新配置中的路径，不要假设上述路径仍存在。

### 2.2 Dukes 原始来源

1. Dukes, O., Vansteelandt, S., & Whitney, D. (2024). *On Doubly Robust Inference for Double Machine Learning in Semiparametric Regression*. JMLR, 25(279), 1–46。[期刊页面](https://jmlr.org/papers/v25/22-1233.html)，[正式 PDF](https://jmlr.org/papers/volume25/22-1233/22-1233.pdf)。
2. 期刊页面链接的[作者代码仓库](https://github.com/ordukes/DRInference)。本次核对版本：`e7dfb038117aea3d95ca1371e2e85557ff9f25d8`；重点为 [dri_functions.r](https://github.com/ordukes/DRInference/blob/e7dfb038117aea3d95ca1371e2e85557ff9f25d8/dri_functions.r) 中 `rdml_cross.test`，同时参考 `dml_cross.test`。

实现前阅读：Section 2 的 PLR 记号，Section 3.2 的错误一步修正讨论，Section 3.3（纸面第 9–12 页）的核回归、更新和交叉拟合算法，Section 4.1/4.2/4.4 的条件与局限，Section 5 的实现选择。第 6 节以下的公式是对原算法的重新表述；原论文采用 CC-BY 4.0，保留以上作者与来源归属。

**优先级：用户本轮限定范围 → 用户论文和实际旧代码的核对结果 → JMLR 正式论文及作者实现 → 旧对话建议。**发现论文、代码、作者实现之间不一致时，列出差异和处理理由；不要悄悄修改原实验来消除差异。

## 3. 必须先做的只读审计

在任何模型训练或 Monte Carlo 运行之前，生成 `audit/legacy_audit.md` 和 `audit/source_manifest.json`。

1. 查找适用的 `AGENTS.md`、项目说明和依赖信息；确认原项目是否受 Git 管理。如有 Git，记录 commit 和工作区状态，不 reset、clean、checkout 覆盖用户改动。如无 Git，用清单和 SHA-256 保护旧文件，不必为了本任务强制重建仓库。
2. 记录论文、两个主 notebook、旧 README、将用于核验的旧结果文件的绝对路径、大小、修改时间和 SHA-256。
3. **将 `.ipynb` 当 JSON 读取，静态检查单元和保存输出。禁止用 `Run All`、`nbconvert --execute` 或直接执行整份 notebook 来完成审计。**两个主 notebook 有顶层 `run_simulation()`；lognormal 版本还会向桌面写汇总 CSV。
4. 建立“论文设定 → notebook 文件/单元 → 函数/参数 → 新实现引用位置”的映射。明确区分论文 Section 2 的主实验、Appendix A.5 的 learner 敏感性实验和 oracle 实验。
5. 记录 Python、NumPy、SciPy、scikit-learn、joblib 版本；记录 HGBR 的完整 `get_params()`，包括旧代码未显式指定的默认参数。旧环境版本无法恢复时如实标注，不声称逐位复现；不要全局升级旧项目依赖。
6. 核实输出路径与副作用，确保新增程序所有输出写入新目录。
7. 审计全部通过且无实质歧义后，继续实现与本轮小规模模拟，不为常规可逆步骤反复请求确认。只有缺失关键材料、冲突会改变实验定义或基准无法复现且原因未明时，暂停相应运行并说明具体缺口。

### 3.1 隔离方式

建议在原项目内**新建** `dukes_pilot/`，该目录若已存在则先审计并复用合适内容，禁止覆盖已有同名结果。本任务书应保存为 `dukes_pilot/README.md`，**不要替换原项目根目录的 README**。原 `code/`、论文、`resaults/` 和 `output image/` 保持不变。

优先复用可导入的纯函数。如只有 notebook，可将已核对的纯函数机械提取到新目录中的 `legacy_adapter.py`，保留来源文件、cell 索引、来源哈希；不要同时执行演示抽样、完整模拟、绘图或导出单元。不要把不受控的整段 notebook 当模块执行。

提取后必须与原函数在相同固定输入下比较，证明 DGP、残差、估计值和标准误未发生变化。不要为了代码整洁改写随机数调用或向量化次序。

## 4. 必须保持一致的原实验定义

### 4.1 模型与 DGP

统一沿用用户论文记号：

$$
Y=\theta_0D+g_0(X)+U,\qquad D=m_0(X)+V,\qquad\theta_0=1.
$$

`X ~ N(0,Σ)`，`Σ[j,k]=0.5**abs(j-k)`；`V ~ N(0,1)`；`X,V,U` 相互独立。只改变 outcome error `U`，不把 treatment disturbance 改成重尾。

首轮用 `DGP2_nonlinear`：

$$
\begin{aligned}
m_0(X)&=0.60\sin X_1+0.35(X_2^2-1)+0.25X_3X_4-0.20\tanh X_5,\\
g_0(X)&=1.20\sin X_1+0.70(X_2^2-1)+0.50X_3X_4+0.30\tanh X_5.
\end{aligned}
$$

仅在后续需要 DGP1 时，继续使用旧函数：

$$
\begin{aligned}
m_0(X)&=0.80X_1-0.50X_2+0.30X_3+0.20X_4-0.10X_5,\\
g_0(X)&=1.00X_1+0.50X_2-0.50X_3+0.25X_4+0.10X_5.
\end{aligned}
$$

这些公式用于审计，程序优先复用原 `ar1_covariance`、`nuisance_functions`、`generate_data`，不要重新设计 DGP。首轮不运行 `p=10`；若未来扩展，后五维仍为相关但结构系数为零的变量。

### 4.2 三种 outcome error

$$
\begin{aligned}
\text{Gaussian: }&U\sim N(0,1),\\
\text{Student-t: }&U=T_{2.1}\sqrt{(2.1-2)/2.1},\quad T_{2.1}\sim t_{2.1},\\
\text{Lognormal: }&U=\frac{\exp(2Z)-\exp(2)}{\sqrt{\{\exp(4)-1\}\exp(4)}},\quad Z\sim N(0,1).
\end{aligned}
$$

必须使用**总体矩**中心化和标准化，不能按本次样本或验证折再中心化/归一化，不能缩尾、截断、删极端观测。总体方差为 1 不意味着每次重尾样本的样本方差应接近 1。

### 4.3 随机数：保留原编号，不对子集重新编号

原规则为：

```python
dataset_seed = 20260826 + 100_000 * legacy_dgp_id + 1_000 * legacy_error_id + replication
crossfit_seed = dataset_seed + 500_000
# replication 从 0 开始；首轮取 0,...,199
```

| 本轮场景 | `legacy_dgp_id` | `legacy_error_id` | `replication=0` 的数据 seed |
|---|---:|---:|---:|
| DGP2 / Gaussian | 1 | 7 | 20367826 |
| DGP2 / Student-t(2.1) | 1 | 0 | 20360826 |
| DGP2 / lognormal(2.0) | 1 | 0 | 20360826 |

原 DGP 列表是 `['DGP1_linear','DGP2_nonlinear']`。Student-t 原网格为 `[2.1,2.3,2.6,3,5,10,20,'N(0,1)']`；lognormal 原网格为 `[2.0,1.5,1.25,1.0,0.75,0.5,0.25,'N(0,1)']`。

**最常见错误：**把筛查列表改成 `[Gaussian,t2.1,lognormal2]` 后用 `enumerate` 生成新 seed；或只保留 DGP2 后把其 id 改为 0。必须使用固定的原编号映射，筛选场景和决定 seed 分开。

沿用 `np.random.default_rng(dataset_seed)` 及原调用顺序：先 `multivariate_normal` 生成 X，再生成 V，最后生成 U。不得用数学上等价的其他采样器替换原调用。Student-t 和 lognormal 首场景 seed 相同是旧设计的一部分，不要擅自添加 family seed 偏移；结果主键必须包含误差 family，防止错误合并。

两个主 notebook 的 Gaussian 相同，核实后只运行一次并共享结果，不计成两份独立证据。`DML_p5_gaussian_RF_vs_HGBR.ipynb` 使用不同 seed 方案，不得拿其 Gaussian 当本轮主 baseline。

新增核回归调参如用随机算法，采用单独、记录在配置中的辅助随机流，不能消耗或改变旧数据、fold 和 HGBR 的随机流。并行调度顺序不得影响结果。

### 4.4 原 HGBR 与五折

```python
HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=model_seed,
)
KFold(n_splits=5, shuffle=True, random_state=crossfit_seed)
# k 从 0 开始
treatment_model_seed = crossfit_seed + 100*k + 1
outcome_model_seed   = crossfit_seed + 100*k + 2
```

保持其他参数的有效默认值与基准运行环境一致，特别记录 `early_stopping` 等隐式设定。不新增标准化、调参搜索、特征选择或训练内预测替代 OOF 预测。

每个数据集只生成一份 fold 索引，A/B/C 共享。每个观测恰好属于一个验证折，训练和验证索引互斥。初始 HGBR 在其余四折训练；不要将 Dukes 辅助核回归的折内拟合误改为“所有环节必须再做一次交叉拟合”。

## 5. 三组方法，分别回答不同问题

| ID | 方法 | 本轮输出与解释 |
|---|---|---|
| A `original_mean_dml` | 完全保留旧 Mean-DML：学习 `Y|X` 和 `D|X`，残差 OLS + 原 Wald SE | 原目标的估计值、SE、Wald coverage；主要基准 |
| B `standard_score` | 用原 HGBR 学习 `D|X` 和 `(Y-θD)|X`，计算未修正 score test | 与 C 共用初始预测；用于区分学习目标变化与修正效果 |
| C `dukes_dr_gcm` | 在 B 的初始预测上按 Section 3.3 增加核回归、更新与完整修正 score | 在真值处检验，以检验反演覆盖率作为筛查指标 |

A 对 C 是总体流程比较；B 对 C 才更直接反映新增修正步骤的影响。不要把 B 与旧 A 混称为同一有限样本实现。

对 A 保留原函数 `fit_mean_dml`：

$$
\widehat\theta_A=\frac{\operatorname{mean}(\widetilde D\widetilde Y)}{\operatorname{mean}(\widetilde D^2)},\quad
\widehat J=\operatorname{mean}(\widetilde D^2),\quad
\widehat{SE}_A=\sqrt{\operatorname{var}_{n-1}\!\left[\widetilde D(\widetilde Y-\widehat\theta_A\widetilde D)/\widehat J\right]/n}.
$$

区间仍为 `theta_hat ± 1.96*se_hat`。不要为了让 A、B、C 的数值看起来一致而替换 A 的 SE 或统计量。

记号映射必须写进实现说明：用户论文的 `m0=E[D|X]` 对应 Dukes 的 `g`；Dukes 的 `m_theta=E[Y-theta*D|X]` 在真值处对应用户论文的结构函数 `g0`。总体恒等式 `m_theta=ell-theta*m0` **不意味着**分别训练的 HGBR 预测相减等于对 `Y-theta*D` 重新训练。B/C 默认直接学习该变换响应，不得未经说明用 A 的预测差代替。

A 与 B/C 可复用完全相同的 treatment OOF 预测，避免重复训练；B/C 的 outcome 初始预测必须逐元素相同。

## 6. Dukes DR-GCM 的实施规格

下面 `theta` 是被检验的候选值；程序函数必须接收它为参数，不能硬编码真值。筛查 runner 在模拟中传入 `theta=1.0`。

### 6.1 初始学习：训练折

对每个验证折 `I_k`，在补集 `I_k^c` 上用第 4.4 节完全相同的 HGBR 设置拟合：

$$
\widehat g^{(-k)}:D\sim X,\qquad
\widehat m_\theta^{(-k)}:(Y-\theta D)\sim X.
$$

在验证折预测并定义：

$$
a_i=D_i-\widehat g_i,\qquad
b_i=Y_i-\theta D_i-\widehat m_{\theta i},\qquad i\in I_k.
$$

B 的逐点 score 为 `psi_B[i]=a_i*b_i`。

### 6.2 辅助学习：验证折内的一维 Nadaraya–Watson 核回归

按论文 Section 3.3，在同一个 `I_k` 内进行：

$$
\widehat G_i=
\frac{\sum_{j\in I_k}K((\widehat g_i-\widehat g_j)/h_G)b_j}
{\sum_{j\in I_k}K((\widehat g_i-\widehat g_j)/h_G)},
\qquad
\widehat M_i=
\frac{\sum_{j\in I_k}K((\widehat m_{\theta i}-\widehat m_{\theta j})/h_M)a_j}
{\sum_{j\in I_k}K((\widehat m_{\theta i}-\widehat m_{\theta j})/h_M)}.
$$

即 G 是“结果残差对处理预测”，M 是“处理残差对变换结果预测”。索引各为一维，不是对全部 X 回归，不是将两种预测联合做二维核回归，也不能把它们替换成 HGBR。

核回归拟合、带宽选择、alpha/beta 更新都在该验证折内；不跨折混合初始预测来拟合一套全局辅助回归。`n=1000,K=5` 时，每个辅助回归实际使用 200 条记录。

**带宽实现要求：**论文使用 Nadaraya–Watson 与交叉验证；作者代码调用 R `np` 包的 `npregbw` 和 `npreg`。优先将作者测试函数作为参考实现，记录 R/`np` 版本、核函数、回归类型、带宽类型、CV 目标、优化初始化/随机种子，以及最终带宽。官方 `npregbw` 文档所列相关默认值为 `regtype="lc"`、`bwmethod="cv.ls"`、`bwtype="fixed"`、`ckertype="gaussian"`、`ckerorder=2`；以实际锁定版本再核验一次。参见 [npregbw 官方文档](https://search.r-project.org/CRAN/refmans/np/html/np.regression.bw.html) 与 [npksum Example 4 的 leave-one-out 对照](https://search.r-project.org/CRAN/refmans/np/html/np.kernelsum.html)。若实现 Python 版本，先固定 R 选出的带宽核对逐点辅助预测、更新、score 和 T，再单独检查带宽优化；不能将“同样用了 CV”写成逐项复现作者软件。

作者环境不可用时，可继续实现明确命名的 Python 核回归移植版本，但先形成 `method_spec.md`：固定核函数、局部常数回归、CV 方式及搜索规则，说明与作者软件的差异。可用 Gaussian kernel + leave-one-out squared-error CV 作为透明的工程选择；这不是声称论文规定了唯一带宽网格。所有这些选择必须在观察 pilot coverage 前固定，不以 0.95 coverage 调带宽。若带宽最优值频繁落在搜索边界，先标记实现敏感性，不据此下选题结论。

区分“LOOCV 选择带宽时排除自身”和“最终在折内训练位置求拟合值”：原公式最终求和包含自身；不能无说明把最终拟合也改成 leave-one-out。

### 6.3 无截距的一维更新

对每折分别求：

$$
\widehat\alpha_k=\frac{\sum_{i\in I_k}\widehat G_i a_i}{\sum_{i\in I_k}\widehat G_i^2},\qquad
\widehat\beta_k=\frac{\sum_{i\in I_k}\widehat M_i b_i}{\sum_{i\in I_k}\widehat M_i^2}.
$$

然后：

$$
r_{D,i}=a_i-\widehat\alpha_k\widehat G_i,\qquad
r_{Y,i}=b_i-\widehat\beta_k\widehat M_i.
$$

G 用于更新处理预测，M 用于更新变换结果预测，不能交换。按该无截距版本实现；增加截距或惩罚项应另标变体。

### 6.4 完整修正 score 与标准化

$$
\psi_{C,i}=r_{D,i}r_{Y,i}-\widehat G_i r_{D,i}-\widehat M_i r_{Y,i}.
$$

折内更新方程会使最后两项各自的样本均值接近零，**但逐点不为零，计算方差绝不能删去**。保存完整逐点 score 或足以重算它的折内数组。

将每折结果按原观测顺序放回长度为 n 的向量。等大五折时，均值为所有观测 score 的均值。本轮统一 B/C 采用作者 R 测试函数中的样本方差约定：

$$
\bar\psi=n^{-1}\sum_i\psi_i,\quad
s_\psi^2=(n-1)^{-1}\sum_i(\psi_i-\bar\psi)^2,\quad
T(\theta)=\frac{\sqrt n\,\bar\psi}{s_\psi},\quad
p=2\{1-\Phi(|T|)\}.
$$

不要平均五个 fold 的 t 值、p 值或 coverage；先合并 score 再标准化。若未来允许不等大折，需要另行明确权重；本轮 `n=1000` 可整除 5，不引入该差异。

**原文与代码的两个核对点：**

- 纸面第 11 页 Step 6 分母印为 `sigma_hat^2/sqrt(n)`，第 12 页又把 `sigma_hat^2` 定义为方差。结合 score-test 的标准化及作者 `rdml_cross.test` 的 `mean(u)^2/(var(u)/N)`，本实现明确使用**标准差**，不机械除以方差；将此记录为原文疑似排版问题。
- 论文方差表达式对应除以 n；作者 R `var(u)` 除以 n−1。本轮 B/C 默认 `ddof=1`，并记录差异，不在两种方法之间混用。作者代码的命名与正文也不同：代码初始 `G/M` 对应正文初始 `g/m`，代码辅助 `Gs/Ms` 分别对应正文辅助 `M/G`。

不要直接调用作者 `rdml_cross.est` 代替上述 score-test 流程。本轮不需要该点估计辅助函数。

### 6.5 数值退化与失败

- 检查所有预测、带宽、更新和 score 是否有限，记录异常及所属折。
- 辅助方向严格为零向量时，更新量定义为零是自然约定；近零方向优先用归一化投影或稳定最小二乘计算实际更新量，并记录系数与更新量范数，不默默加 ridge 或截断系数。
- 辅助索引为常数、CV 无可用带宽或分母退化时，按参考实现核对并预先规定处理。采用常数均值回归等退化延拓时要明确标记；未经验证不能声称与作者特殊分支相同。
- `s_psi=0`、非有限 score 或不可解退化情况记为失败，不能记为“不拒绝真值”。禁止换 seed 重抽一份“更好”的样本。
- 不把原代码的全局 `warnings.filterwarnings('ignore')` 原样搬到新运行入口；保留并汇总新方法的数值警告。

## 7. 只在真值处检验，得到反演置信集的覆盖率

完整反演置信集定义为：

$$
\mathcal C_r=\{\theta:|T_r(\theta)|\le z_{0.975}\}.
$$

因此只需在每次模拟的真值 `theta0=1` 上计算一次 B/C：

$$
I_r=\mathbf1\{|T_r(1)|\le z_{0.975}\},\qquad
\widehat c=R^{-1}\sum_r I_r.
$$

这正是相应反演置信集包含真值的判定，不需要先求出区间端点。默认 B/C 用 `scipy.stats.norm.ppf(0.975)`；A 保留旧 `1.96`，在元数据中记录这个极小差别。

在候选原假设下学习 `Y-theta*D` 是检验的正常步骤，不是把真实 nuisance 函数泄露给估计器。`theta0` 只由模拟驱动用于评价；方法函数不能读取真实 `m0/g0/U/V` 或用其选择带宽。

输出名称要准确：A 为 `wald_coverage`，B/C 为 `inverted_test_coverage_at_truth`。B/C 的 `theta_hat`、`theta_se`、`ci_low`、`ci_high`、`ci_length` 本轮均为空或未计算，不能把 `mean(score)` 当 theta 估计，不能把 score 的 SE 当 theta 的 SE，不能伪造区间长度、bias 或 RMSE。

## 8. 验证顺序与运行预算

### 8.1 原基准检查

先对三个固定场景的前 1–2 个 replication，比较旧纯函数与新 adapter 的数据、fold、OOF 残差、A 点估计及 SE。数据和 folds 应一致；同环境模型结果应在预先规定的严格浮点容差内一致。

论文 Appendix A.1/A.2 与保存输出中的主 Mean-DML coverage（原 `R=1000`）如下，仅作背景核对：

| DGP / p=5 | Gaussian | t(2.1) | lognormal(2) |
|---|---:|---:|---:|
| DGP1 | 0.838 | 0.713 | 0.705 |
| DGP2 | 0.602 | 0.314 | 0.332 |

本轮 `R=200` 是旧 seed 序列的前缀，不能要求数值精确等于原 1000 次汇总，更不能靠调 learner 参数追这些数字。同一前缀属于旧完整样本的一部分，不是独立模拟；如计算两者差异的不确定性，应考虑重叠。主要正确性证据是固定重复的程序结果一致，而不是“coverage 大约 0.60”。

### 8.2 方法检查

至少完成以下有实际意义的检查，并把结果写入 `audit/validation.md`：

1. B 与 C 的数据、fold 和初始预测一致；初始预测仅来自训练折；辅助回归按第 6 节在验证折完成。
2. 把 G/M 强制设为零，仅作为代数测试时，C 退化为 B。
3. 检查每折 `mean(G*rD)` 和 `mean(M*rY)` 在适当数值容差内为零，且均值等价不导致错误删减方差项。
4. 在一组固定小数组上，与独立的直接核加权计算比较；如果可以运行作者 R 函数，对同一折输入验证带宽、更新、完整 score、方差、p 值，记录软件差异。无 R 时完成独立手工公式校验，并明确作者软件交叉核验尚未完成。
5. 不改变配置重跑同一 replication，结果应可复现；少量任务的串行与并行结果应一致。
6. 验证常数索引、零辅助方向、非有限值、零 score 方差的处理及失败记录不会被汇总成成功覆盖。

完成上述检查后，跑 3 个场景 × 200 个数据集，每个数据集共享 A/B/C。先测量冒烟检查耗时，给出预计时长；控制线程避免 joblib 与 HGBR/R 多层并行占满机器，不因加速改变估计定义。预估成本明显超出小规模筛查时，先交付可运行实现与实测耗时，说明受限环节。

保存进度并支持断点续跑：唯一键至少为 `(dgp,p,n,error_family,error_parameter,replication,method,config_hash)`。只补缺失任务；配置改变必须新开 run，不能混入已有结果。不要每次重启重新随机切 folds。

## 9. 输出、Monte Carlo 误差与失败分母

### 9.1 每次重复必须保留

- 场景、原编号、replication、数据/fold/model seeds、fold 索引摘要或哈希、方法、配置哈希、运行时间。
- A：theta、SE、区间和覆盖指示；B/C：`theta_tested`、score 均值、score SD、T、p、接受真值指示。
- `status`、失败类型、失败折、警告；核回归带宽、alpha/beta、更新量范数、方向分母、score 方差。
- 复用旧 U 诊断：样本均值、样本方差、最大绝对值；lognormal 可另报最大值。这些只作诊断，不参与估计与调参。

旧程序中 NaN 区间比较可能产生 `covered=False`，而 `valid_runs` 只统计 theta 非空。新报告需显式区分算法失败和有限但不覆盖；为核对可单独保留 `legacy_covered`，不能无声更改旧结果口径。

### 9.2 Coverage 和误差条

最终汇总以下公式以计划 `R` 次均已实际尝试为前提，其中有效 `R_v` 次，成功覆盖 `S` 次。中断或续跑期间另外区分 `planned/attempted/pending/failed`，尚未运行的任务不能计为算法失败：

- `coverage_valid=S/R_v`，并明确标注是有效运行条件下的比例；若 `R_v=0` 则为空。
- `failure_rate=(R-R_v)/R`，始终报告。
- 给出所有计划运行覆盖率的缺失结果界限 `[S/R,(S+R-R_v)/R]`；不把失败默认为覆盖，也不只删除失败而隐瞒。
- 有效运行上的 `MCSE=sqrt(c*(1-c)/R_v)` 与二项 Wilson 95% 区间。失败存在时，此区间不能解释为整个算法无条件覆盖率的区间。

无失败且 `R=200,c≈0.95` 时，MCSE 约为 0.0154，粗略双侧误差约 ±0.030；最坏情况 MCSE 约 0.0354。不要把 0.93 与 0.95 的差异当作确定结论，也不要用“95% 区间包含 0.95”证明等价。

### 9.3 配对方法比较

同一场景中用同一数据集的覆盖指示做配对差：

$$
d_r=I_{C,r}-I_{B,r},\quad
\widehat\Delta_{C-B}=\bar d,\quad
MCSE(\widehat\Delta)=\operatorname{SD}_{R_{pair}-1}(d)/\sqrt{R_{pair}}.
$$

同时报告 C−A、B−A；注明有效配对数、两方法各自失败数和非配对记录。必须在共同有效的重复上逐对计算，不能相减两列各自不同有效样本上的 `coverage_valid` 冒充配对差。不要把方法当独立样本计算标准误。报告配对差分布的计数，避免少量变化时正态近似被过度解读。双方成功子集可能偏向较容易的样本，有失败时不能把该子集的差值直接解释为全部计划 R 次上的优劣；应另给缺失差值取 −1 到 1 所得的保守界限，或明确暂不作整体判断。

分别展示绝对 coverage 与同方法的 `coverage_tail-coverage_Gaussian`。跨误差场景有各自 seed 设计，不能仅凭相同 replication 编号就称其为同一数据；需要差值不确定性时按实际随机数依赖计算，不能套用方法内的配对解释。三场景是筛查，不宣称完成多重比较下的正式推断。

### 9.4 应交付的新增文件

按仓库风格调整名称即可，职责必须保留：

```text
dukes_pilot/
  README.md
  method_spec.md                 # 公式、来源、软件约定、与作者实现差异
  legacy_adapter.py             # 复用/机械提取的旧纯函数
  dukes_score.py                # 初始预测、核回归、更新与测试
  run_pilot.py                  # 支持 audit-only / smoke / pilot / resume
  summarize_pilot.py
  configs/pilot.json            # 固定原编号、原参数和辅助回归实现选择
  audit/
    legacy_audit.md
    source_manifest.json
    environment.json
    validation.md
  tests/                        # 上述必要的代数/回归/失败处理检查
  results/<run_id>/
    config.json
    raw_results.csv
    fold_diagnostics.csv
    summary.csv
    paired_differences.csv
    failures.csv
    coverage_comparison.png
    decision_note.md
```

实现完成后给出在该机器上**实际验证过**的命令，覆盖只审计、冒烟、筛查和续跑四种模式。上面是要求创建的接口，不表示现在已有这些脚本。`summary.csv` 对 A 可继续汇总 bias、SD、Mean SE、RMSE 和区间长度；对 B/C 不适用项留空。

图表按三个场景展示 A/B/C coverage、0.95 参考线与 Monte Carlo 区间，并显示失败数；明确 A 是 Wald、B/C 是检验反演覆盖。中文结论不要只贴一张图。

## 10. 如何解释筛查结果

`decision_note.md` 先回答“这些结果对当前选题筛查意味着什么”，再说明证据和局限。至少分别回答：

1. A 是否复现旧实现，新的样本与随机设置是否对齐？
2. C 相比 A 改善多少？C 相比 B 改善多少？不能将 A→C 的全部改善归因于修正。
3. Gaussian 与两种重尾下是否仍有明显欠覆盖？改善是否超过 Monte Carlo 噪声，是否伴随失败或退化？
4. 本轮只检查真值处 coverage，尚未检验置信集宽度、信息量、功效或非连通性；高 coverage 也可能来自过度保守，不能宣称方法已全面解决问题。
5. 哪个后续动作对决策最有价值，是否需要用户决定继续投入？

可采用以下表述，不按结论偏好挑结果：

- **三场景明显改善且接近名义水平、失败少：**现有方法已显著缓解当前特定设计的问题，可支持降低继续原选题的优先级；仍只是有限场景初筛，不等于证明全部重尾 DML 问题已解决。
- **只改善 Gaussian 或某一种重尾：**说明改善范围及剩余缺口，保留继续当前方向的可能性。
- **仍严重欠覆盖：**先报告基准/方法验证是否通过、数值诊断如何，再谈剩余研究空间；不自动实现 Huber。
- **区间误差较大或失败多：**结论为暂不明确，不能强行二选一。

需要更精确的 coverage 时优先补同一 `n=1000` 的 R；需要检验样本量趋势时才另加 `n=2000`；需要检查可推广性时再补 DGP1。扩展实验、完整区间反演或替代方法不在本轮自动运行范围。

理论表述必须克制：Dukes 修正主要针对 nuisance estimation 引起的推断偏差，不是 Huber 那类有界影响或重尾专用方法。DR inference 仍需要原文的收敛、矩、经验过程和非退化条件；固定 HGBR 在本 DGP 中是否满足它们不能由单次模拟证明。t(2.1) 有限方差但没有四阶矩；lognormal 的所有正阶矩有限但有限样本可能极不稳定。仅一个 nuisance 一致时还存在论文讨论的非正则性，不能写成统一有效的万能保证。

## 11. 最终验收清单

- [ ] 原材料路径与文件哈希已记录；审计前没有执行旧 Monte Carlo。
- [ ] 旧 notebook、论文、README 和结果文件的哈希未改变。
- [ ] 原 DGP、总体标准化、HGBR 参数、环境差异已核对并记录。
- [ ] DGP2 id=1；Gaussian error id=7；两类极端重尾 error id=0；replication 从 0 起。
- [ ] A/B/C 共享样本与 folds；B/C 共享初始预测；初始 HGBR 无验证折训练泄漏。
- [ ] Dukes 辅助回归与更新使用原论文规定的验证折；核回归软件约定已固定并说明。
- [ ] G/M 响应和索引正确，alpha/beta 更新正确，方差使用完整修正 score。
- [ ] 标准化除以 score SD，B/C 默认 ddof=1；没有照搬纸面疑似排版错误。
- [ ] 原实现一致性、退化为 B、更新矩条件、重复可复现及失败处理检查已通过。
- [ ] 只完成本轮 3 个场景、n=1000、p=5、R=200；没有未经要求扩完整网格。
- [ ] coverage 有 MCSE/Wilson 区间、有效数和失败数；方法差使用配对数据。
- [ ] B/C 未计算的点估计、参数 SE、区间端点和长度没有被伪造。
- [ ] 输出文件、实测运行命令、计算时间和中文选题筛查结论已交付。

**现在按顺序执行：只读审计 → 独立接入 → 正确性检查 → 小规模筛查 → 带不确定性的中文结论。不要改写原研究目的，也不要用另一套 DGP 或随机种子给出看似相近的结果。**

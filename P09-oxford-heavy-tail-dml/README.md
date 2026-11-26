# DML heavy-tail simulation：Student-t、Lognormal 与正态基准（`p = 5`）

## 项目目的

本项目研究重尾结果误差对 Double Machine Learning（DML）估计与推断的影响，对比：

1. `Mean_DML`：第一阶段估计条件均值，第二阶段使用残差 OLS；
2. `Median_DML`：第一阶段估计条件中位数，第二阶段使用 LAD/中位数回归思想。

当前项目分别研究两类重尾误差，并在两个独立 notebook 中加入 $N(0,1)$ 作为同方差 baseline：

- 对称的 Student-t 重尾误差；
- 右偏的 lognormal 重尾误差。

项目不再包含 contamination（混合污染）机制。Student-t 与 lognormal 不混在同一个 notebook 中，避免混淆两类误差分布及其参数含义。

主文件：

- `DML_p5_student_t_heavytail.ipynb`：`p=5`、两种 DGP、Student-t 自由度网格与 $N(0,1)$ baseline 的完整 Monte Carlo 模拟。
- `DML_p10_student_t_heavytail.ipynb`：把协变量扩展到 `p=10`；前 5 维保留原 DGP，后 5 维作为不直接进入真实 nuisance functions 的额外无关变量；其余 Student-t 实验参数与 `p=5` 版本一致。
- `DML_p5_lognormal_heavytail.ipynb`：保持相同实验参数、两种 DGP 与估计方法，将误差改为均值中心化且方差标准化的 lognormal，并加入 $N(0,1)$ baseline。
- `DML_p10_lognormal_heavytail.ipynb`：把 lognormal 实验扩展到 `p=10`；前 5 维保留原 DGP，后 5 维作为额外无关变量，其余参数与 `p=5` lognormal 版本一致。

## 基本模型

使用部分线性回归模型：

$$
Y=D\theta_0+g_0(X)+U,
\qquad
D=m_0(X)+V.
$$

公共固定设定：

- 协变量维度：基础实验为 $p=5$，Student-t 和 lognormal 扩维实验均为 $p=10$；
- 真实处理效应：$\theta_0=1$；
- $E(U\mid X,D)=0$；
- $E(V\mid X)=0$；
- $X\sim N(0,\Sigma)$，其中 $\Sigma_{jk}=0.5^{|j-k|}$；
- $V\sim N(0,1)$。

## 两种 DGP

### DGP 1：线性 nuisance functions

令 $X=(X_1,\ldots,X_5)$，则

$$
m_0(X)=0.8X_1-0.5X_2+0.3X_3+0.2X_4-0.1X_5,
$$

$$
g_0(X)=X_1+0.5X_2-0.5X_3+0.25X_4+0.1X_5.
$$

### DGP 2：非线性 nuisance functions

$$
m_0(X)=0.6\sin(X_1)+0.35(X_2^2-1)
+0.25X_3X_4-0.2\tanh(X_5),
$$

$$
g_0(X)=1.2\sin(X_1)+0.7(X_2^2-1)
+0.5X_3X_4+0.3\tanh(X_5).
$$

$X_2^2-1$ 用于中心化，因为标准正态变量满足 $E(X_2^2)=1$。

### `p=10` 扩维设计

在 `DML_p10_student_t_heavytail.ipynb` 与 `DML_p10_lognormal_heavytail.ipynb` 中，令

$$
X=(X_1,\ldots,X_{10})\sim N(0,\Sigma),
\qquad
\Sigma_{jk}=0.5^{|j-k|}.
$$

真实的 $m_0(X)$ 与 $g_0(X)$ 仍完全使用上面列出的 `p=5` 公式，即只依赖 $X_1,\ldots,X_5$。$X_6,\ldots,X_{10}$ 不直接进入处理方程或结果方程，其结构系数均为 0。它们会作为额外特征输入 nuisance learner，用于研究增加无关协变量后 DML 的有限样本表现。

因为 10 维协变量仍共同服从 AR(1) 相关结构，后 5 维可能与前 5 维相关；“无关”在这里指给定前 5 维后，它们不在真实 nuisance functions 中提供额外结构信号，而不是指它们与前 5 维统计独立。

## Student-t 重尾设定

结果方程误差定义为

$$
U=T_\nu\sqrt{\frac{\nu-2}{\nu}},
\qquad T_\nu\sim t_\nu,
\qquad \nu>2.
$$

缩放后每个 Student-t 场景都满足 $E(U)=0$、$\operatorname{Var}(U)=1$。另设 $U\sim N(0,1)$ baseline，因此所有误差场景均具有相同均值和方差，差异主要来自尾部形态。

默认自由度网格：

```python
T_DF_GRID = [2.1, 2.3, 2.6, 3, 5, 10, 20]
NORMAL_BASELINE = "N(0,1)"
ERROR_GRID = [*T_DF_GRID, NORMAL_BASELINE]
```

自由度越小，尾部越厚。`df=2.1、2.3、2.6` 靠近方差存在的边界 `df=2`，可用于细致观察极重尾区域；这些场景虽然具有有限方差，但有限样本波动可能很大。`df=3` 的四阶矩仍不存在，`df=5、10、20` 则用于观察尾部逐渐变轻的过程。

## Lognormal 右偏重尾设定

令

$$
W=\exp(\sigma Z),
\qquad Z\sim N(0,1),
$$

并将结果方程误差定义为

$$
U=
\frac{W-E(W)}{\sqrt{\operatorname{Var}(W)}}
=
\frac{\exp(\sigma Z)-\exp(\sigma^2/2)}
{\sqrt{(\exp(\sigma^2)-1)\exp(\sigma^2)}}.
$$

这里使用总体矩进行标准化，而不是用每次抽样得到的样本均值和样本方差。每个 lognormal 场景理论上均满足

$$
E(U)=0,
\qquad
\operatorname{Var}(U)=1,
$$

因此仍满足基本模型中的 $E(U\mid X,D)=0$。但 lognormal 是非对称分布：均值中心化后 $U$ 的中位数不为 0，且分布具有明显右偏。解释 Mean DML 与 Median DML 的差异时，必须同时考虑尾部、偏度及目标函数差异，不能将其视为 Student-t 的直接对称替代。

默认形状参数网格：

```python
LOGNORMAL_SIGMA_GRID = [2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25]
NORMAL_BASELINE = "N(0,1)"
ERROR_GRID = [*LOGNORMAL_SIGMA_GRID, NORMAL_BASELINE]
```

`sigma` 越大，右偏和尾部越强。该网格包含 7 个 lognormal 场景，与 Student-t notebook 的 7 个重尾场景数量一致，并按“重尾到轻尾”排序；由于 lognormal 没有自由度参数，`sigma` 与 Student-t 的 `df` 不存在逐项数值对应关系。`sigma=2` 的有限样本波动可能非常大。

## Notebook 结构

1. 集中调参区；
2. 依赖导入与参数检查；
3. 协方差矩阵、两种 nuisance functions 与对应的重尾数据生成；
4. mean/median nuisance learner；
5. cross-fitting；
6. Mean DML 与 Median DML 第二阶段；
7. Monte Carlo 主循环；
8. Bias、SD、RMSE、平均 SE 与 coverage 汇总；
9. 指标图；
10. CSV 导出。

## 调参入口

正式实验参数集中在 notebook 的第一个代码单元：

```python
P = 5
N = 1000
THETA_0 = 1.0
T_DF_GRID = [2.1, 2.3, 2.6, 3, 5, 10, 20]
NORMAL_BASELINE = "N(0,1)"
ERROR_GRID = [*T_DF_GRID, NORMAL_BASELINE]
N_FOLDS = 5
N_REPLICATIONS = 1000
BASE_SEED = 20260826
N_JOBS = -1
```

lognormal notebook 保持上述公共参数不变，只把分布专属网格替换为：

```python
LOGNORMAL_SIGMA_GRID = [2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25]
NORMAL_BASELINE = "N(0,1)"
ERROR_GRID = [*LOGNORMAL_SIGMA_GRID, NORMAL_BASELINE]
```

`p=10` Student-t notebook 使用：

```python
P = 10
N_SIGNAL = 5
N = 1000
THETA_0 = 1.0
T_DF_GRID = [2.1, 2.3, 2.6, 3, 5, 10, 20]
NORMAL_BASELINE = "N(0,1)"
ERROR_GRID = [*T_DF_GRID, NORMAL_BASELINE]
N_FOLDS = 5
N_REPLICATIONS = 1000
BASE_SEED = 20260826
N_JOBS = -1
```

其中 `N_SIGNAL=5` 表示只有前 5 维直接进入真实 DGP。除 `P` 与新增的 `N_SIGNAL` 外，其余正式参数和 Student-t `p=5` notebook 一致。

`p=10` lognormal notebook 使用相同的公共参数，只将分布网格设为：

```python
P = 10
N_SIGNAL = 5
LOGNORMAL_SIGMA_GRID = [2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25]
NORMAL_BASELINE = "N(0,1)"
ERROR_GRID = [*LOGNORMAL_SIGMA_GRID, NORMAL_BASELINE]
N = 1000
N_FOLDS = 5
N_REPLICATIONS = 1000
BASE_SEED = 20260826
N_JOBS = -1
```

快速检查模式：

```python
QUICK_TEST = True
QUICK_N = 300
QUICK_REPLICATIONS = 2
```

- 调试时保持 `QUICK_TEST = True`；
- 正式实验前改为 `QUICK_TEST = False`；
- 快速模式只用于检查代码，不用于解释统计结果；
- 当前正式实验使用 1000 次重复；
- `N_JOBS = -1` 使用全部可用 CPU，若并行环境有问题可改为 `1`。

## Nuisance learner

两类 DML 都使用 `HistGradientBoostingRegressor`：

- `Mean_DML`：`loss="squared_error"`；
- `Median_DML`：`loss="quantile", quantile=0.5`。

公共超参数保存在 `BOOSTING_PARAMS`。若后续替换为其他学习器，应保持相同数据、随机种子、cross-fitting folds，并保留 mean 与 median 目标函数的区别。

## 估计与标准误

### Mean DML

在 cross-fitted 残差上计算

$$
\widehat\theta
=\frac{\mathbb E_n[\widetilde D_i\widetilde Y_i]}
{\mathbb E_n[\widetilde D_i^2]}.
$$

标准误来自正交得分的经验影响函数。

### Median DML

第二阶段求解

$$
\widehat\theta
=\arg\min_\theta\sum_i
|\widetilde Y_i-\theta\widetilde D_i|.
$$

代码使用 $\widetilde Y_i/\widetilde D_i$、权重 $|\widetilde D_i|$ 的加权中位数计算 LAD 斜率。标准误使用零点核密度估计的 sandwich 公式，并用 MAD 选择稳健带宽。

该 Median DML 标准误仍是最需要根据目标论文进一步核对理论依据的部分。若以后改用 bootstrap、交叉拟合 quantile score 或其他推断方法，应同步更新 notebook 与本 README。

## 输出指标

每个 `(DGP, error_distribution, estimator)` 场景汇总。Student-t notebook 的 `error_distribution` 包括所有自由度和 `N(0,1)` baseline；lognormal notebook 包括所有 `sigma` 场景和同一 baseline：

- `mean_estimate`：估计值均值；
- `bias`：$E(\widehat\theta)-\theta_0$；
- `sd`：Monte Carlo 估计值的经验标准差；
- `mean_se`：每次模拟报告标准误的均值；
- `rmse`：均方根误差；
- `coverage`：95% 置信区间覆盖率；
- `mean_sample_u_variance`：样本误差方差的平均值；
- `mean_sample_max_abs_u`：样本最大绝对误差的平均值；
- `valid_runs`：有效重复次数。

lognormal 汇总 CSV 还包含：

- `lognormal_sigma`：lognormal 形状参数，正态基准处为空；
- `mean_sample_u_mean`：样本误差均值的 Monte Carlo 平均值；
- `mean_sample_max_u`：样本最大正向误差的平均值；
- `mean_sample_skewness_u`：样本误差偏度的平均值。

判断标准误是否合理时，应同时比较 `sd` 与 `mean_se`，不能只看 coverage。

## 依赖环境

需要：

```text
numpy
pandas
scipy
scikit-learn
joblib
matplotlib
seaborn
```

notebook 不会自动执行 `pip install`，也不依赖 `statsmodels` 或 `Jinja2`。

## 保存结果

Student-t notebook 默认：

```python
SAVE_RESULTS = False
```

因此运行 notebook 不会额外产生 CSV。需要保存时改为 `True`，将生成：

- `dml_p5_student_t_raw.csv`；
- `dml_p5_student_t_summary.csv`。

`p=10` Student-t notebook 同样默认 `SAVE_RESULTS=False`。需要保存时改为 `True`，将生成：

- `dml_p10_student_t_raw.csv`；
- `dml_p10_student_t_summary.csv`。

lognormal notebook 默认：

```python
QUICK_TEST = False
SAVE_RESULTS = True
OUTPUT_CSV = Path.home() / "Desktop" / "dml_p5_lognormal_summary.csv"
```

正式运行全部单元后，会在桌面生成一个汇总文件：

- `dml_p5_lognormal_summary.csv`。

`p=10` lognormal notebook 同样默认 `QUICK_TEST=False`、`SAVE_RESULTS=True`，正式运行后在桌面生成：

- `dml_p10_lognormal_summary.csv`。

当 `QUICK_TEST=True` 时，lognormal notebook 会完成快速检查，但不会写出 CSV，防止把 2 次重复的测试结果误当作正式结果。当前 notebook 已在快速模式下完成过端到端运行检查；交付版本已恢复为 `QUICK_TEST=False`、`N=1000` 和 `N_REPLICATIONS=1000`。

## 后续修改规则

1. 不把 contamination 机制加入当前重尾实验，也不把 Student-t 与 lognormal 机制混入同一个 notebook；
2. 不改变 $\theta_0$、DGP 公式或协变量相关结构，除非实验设计明确要求；
3. 不删除 cross-fitting，也不用训练内拟合值替代 out-of-fold 预测；
4. 保留 Bias、SD、平均 SE、RMSE 与 coverage；
5. 修改估计量或标准误公式时同步更新 README；
6. 新增参数放到 notebook 开头的调参区；
7. 先在 `QUICK_TEST=True` 下完整跑通，再做正式模拟；
8. 不在桌面残留临时脚本、测试 CSV、缓存或重复 notebook；
9. 保留中文注释，并明确说明 Student-t 与 lognormal 的中心化、方差标准化方式；
10. 修改 lognormal 的中心化方式时，必须说明该修改对 $E(U\mid X,D)=0$、误差中位数及 Mean/Median DML 解释的影响。
11. 两个 `p=10` 实验均固定 `N_SIGNAL=5`；除非研究设计明确改变，否则 $X_6,\ldots,X_{10}$ 不得加入真实的 $m_0(X)$ 或 $g_0(X)$。

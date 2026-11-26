# Rank-DML 第一阶段 G0

本项目依据本次提供的 `sources/README.md` 和导师 `sources/rank_dml_guide.tex`，与上层目录的旧四阶段方案独立。当前范围仅 G0：标量 smoke → 多元 smoke → 全网格 pilot → 实现核对及冻结。不得把 G0 当作正式主实验。

## 运行环境与入口

当前 Python：`C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe`。数值依赖为 NumPy、SciPy、pandas；各批 config 记录精确版本。当前及以后默认使用 12 个单线程计算进程，上限 12。历史 smoke 使用 8 个进程；pilot 重启前使用 16，恢复后使用 12，原记录保留。

```powershell
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' .\checks.py
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' .\check_supplement.py
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' .\reports\additional_checks.py
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' .\run.py --stage g0_smoke_scalar --workers 12
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' .\run.py --stage g0_smoke_vector --workers 12
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' .\run.py --stage g0_pilot --workers 12
& 'C:\Users\Lenovo\Documents\ATLAS\runtime\Scripts\python.exe' .\reports\finalize_g0.py
```

`--stage g0` 按以上三批顺序执行；`--replications --n --law --d --rho --learner` 可限定配置。默认完整规模为 smoke 每条件 20 次、pilot 每条件 100 次。部分网格仅用于额外验证，不构成 G0 完成。恢复运行时，源码哈希、网格、重复数必须一致。运行器不提供 G1–G6。

## 冻结的数值约定

- 十个变量各自使用三次、端点夹持 B-spline。五个等距 knots 包含 min/max，端点总重数 4，每变量 7 个基函数删除最后一列；加前五变量的 10 个交互项，共 70 个非截距特征。样本外线性延拓使用边界函数值和一阶导数。
- 标准差按训练总体口径（ddof=0）；恒定列尺度设 1。每个 target 独立选 GCV penalty，截距不罚，完全相同 criterion 取较大 penalty。
- W 配对仅剔除精确零分母，不丢弃任何有限正权重。取累计权重第一次到达半数的 slope。
- rank 数值并列容差为 `64 * machine_epsilon * max(abs(fold_residual - median(fold_residual)))`。相邻差不超过此容差的残差共享所占 grid scores 的均值。这用于修复直接 W 最优点理论并列的浮点舍入，保证数值平移/尺度等变；不是根据 coverage 调参。
- Huber IRLS 从 LS 初始化，最多 1000 步，系数变化阈值严格使用导师公式；另要求最大 score residual 不超过 `1e-6 * (1 + mean(abs(z)))`。保留所有候选是否收敛、criterion 和失败原因。
- rank 主步长 c=1；不对非正 slope 做替换。协方差不加 ridge。无效区间按不覆盖计入总 coverage，另报 valid-only coverage。
- root-n covariance 按导师公式计算，获得估计量方差时除 n 一次。

## 随机流与保存

master key 为 2026111401，但 stage 参与 SHA-256 seed 派生，smoke、pilot、额外 validation 相互独立。数据和 outer-fold key 不含 learner；inner folds、data、outer、fresh diagnostics 的 stream 标签独立。replication 从 0 起算，worker 调度不影响结果。

每批 `runs/<stage>/config.json` 在计算前保存；`records/*.json` 是每数据集 10 条方法/流程记录及完整训练选择、诊断；`cache/*.npz` 保存 outer residuals、oracle residuals、fold、scale、multiplier、penalties、truth。JSON 使用嵌套坐标向量/协方差矩阵，坐标顺序明确。非有限值存 null，不隐藏失败。

`summary/coordinates.csv` 含逐坐标 Bias/SD/RMSE/MAE/MeanSE/coverage 等；`vectors.csv` 含 vector RMSE、联合椭球覆盖/大小；`contrasts.csv` 含两个 contrast；`paired.csv` 为 W/N 对三基线的配对比较及 2000 次联合 bootstrap；`runtime.csv` 报运行成本；`audit.json` 核对数量；`failures.json` 保留失败类型。paired 的 RMSE 与区间大小分别报告各自 common-success denominator。

G0 所有 cache 和训练选择永久保留至用户另行决定，不自动清理。`archive/pre_numeric_tie_fix` 是数值并列修复前的源码和作废 smoke，不能与当前结果合并。修复后同一数据种子完整重跑两批 smoke；不把旧版本重复计入有效 R。

## 核对材料

`reports/implementation_checks.json`：T1–T9 证据；`checks.py`、`check_supplement.py`：可重跑检查。T1 每分布/维度/相关参数共 100 万独立 draws，以 40 块估计抽样波动，t3 无限四阶矩下 covariance block SE 仅作描述。T8 标量及 d=3、rho=.5 的 Oracle Gaussian 各 n=500/1000/2000、各 200 次，使用独立测试随机流。有限次数的 root-n 方差不应要求单调趋向理论基准。

`reports/additional_checks.py` 补充全 5 个 outer folds 和 15 个 inner folds 的数据扰动测试，以及数据集失败路径的记录留存测试。故意无效的 `failure_check` 不属于 G0 数据网格。`reports/finalize_g0.py` 对全部正式 G0 输出逐条核验，并对每批每个 cell 的 replication 0 进行缓存复算。

`reports/G0_report.md`（完成时生成）为最终验收报告。G0 结束不自动进入 G1。


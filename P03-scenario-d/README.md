# Rank-DML Scenario D

项目包含论文 Proposition `prop:estimand` 的 D1、D2、D3。论文规定正式每条件 2,000 次。本项目预设样本量 500、1000、2000，共九个条件和 18,000 次正式重复。D1 的阶跃尺度、D3 的处理组尺度与样本量均记录为实现约定。[实验配置](experiment_manifest.json)

## 安装和启动

在 PowerShell 进入本目录。已验证环境为 Python 3.11，优先使用本机 Anaconda 的 `environment2`。如需单独建环境，可运行：

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

五个 BAT 优先使用项目 `.venv`，其次使用本机已核验的 `environment2`，最后使用 PATH 中的 Python。每个脚本结束后停留在窗口内，便于查看结果与错误信息。

| 文件 | 用途 |
|---|---|
| [run_smoke.bat](run_smoke.bat) | 每组四次，同时生成三张 smoke PDF |
| [run_pilot.bat](run_pilot.bat) | 每组 100 次单独的 pilot |
| [run_main.bat](run_main.bat) | **手动双击启动正式实验**，显示进度条 |
| [resume_main.bat](resume_main.bat) | 跳过已提交的正式重复，继续计算 |
| [generate_report.bat](generate_report.bat) | 正式结果齐全后生成表格、三张矢量 PDF 和 Markdown |

用 VS Code 打开 [Scenario_D.ipynb](Scenario_D.ipynb)，运行第一个代码单元即可启动或续跑正式实验。第二格每 3 秒读取一次检查点，显示已保存数量和最近检查时间，第三格生成正式报告。如果实验进程已退出，第一格会移开确认属于已退出进程的遗留锁，再从检查点续跑。

默认 12 个 worker，BLAS、OpenMP、NumExpr 各限 1 线程。也可以使用以下终端命令。

~~~powershell
python run.py --mode smoke --workers 12 --resume
python report.py --mode smoke
python run.py --mode pilot --workers 12
python run.py --mode main --workers 12
python run.py --mode main --workers 12 --resume
python report.py --mode main
~~~

可用 `--scenario D2 --n 1000` 只运行指定条件。每条重复一完成就提交到该阶段的 SQLite 检查点，主键为 `scenario,n,rep`。按 Ctrl+C 后用 `--resume` 继续。异常关机若遗留 `checkpoints/running.lock`，确认其中的 PID 已退出后再移开锁文件。阶段和科学配置会在打开检查点时核对。

## 单条复现和存档核验

复现 ID 从 0 起算。单条命令把当次记录打印为 JSON，适合核对数据哈希、seed、估计值、SE 和选阶。

~~~powershell
python run.py --mode main --scenario D2 --n 1000 --rep 137
python -m unittest discover -s tests -v
~~~

统一 master seed 为 `20261120`。每个场景、样本量、重复、阶段、用途各有独立的 SeedSequence 键。数据、外折、Huber 内折及各 max-t 模拟分别使用命名子流。正式汇总前核对预期的全部 18,000 个键、方法、区间及子种子。相同 Python 和数值库版本下，worker 调度与随机数无关。

## 结果和推断口径

- **D1** 使用原 Spline Ridge 与五折。按边际方差 1 标准化阶跃尺度，保持 v 独立。Rank-W、Rank-N、Rank-Ad 输出 naive 和 sandwich。W/N 在 W 初值估计斜率，在最终点构造 sandwich 分子。Rank-Ad 的点估计使用未修正斜率平方和，选择和 studentization 使用原文交叉信息修正。Sandwich 以同一组权重构造候选经验协方差，再对方差乘以未修正信息与修正信息的比值。Sandwich max-t 使用候选贡献的经验相关矩阵。
- **D2** 数据使用论文的两组镜像指数误差，沿用五折秩。独立总体积分分别计算固定 F₀ 的分数根、按候选残差重排的分数根，以及 W 初值代入一步更新的总体目标。论文的 Wilcoxon 数值使用 (2F-1)，程序的 W 采用 (F-1/2)。Rank-N、Rank-Ad 和固定混合分数的报告目标对应实际的一步估计。D2 的 Bickel 无独立基准，真值偏差仍报告。
- **D3** odd-only 使用第 1、3、5 阶走相同的信息修正和容忍规则。固定混合 (J_1+J_2) 使用 W 初值的一步估计。总体结果文件同时给出混合方程根和一步目标。保存原 Rank-Ad 所选阶数、估计权重、偶数阶权重平方占比及偶数项的一步增量。独立误差公式在 D2/D3 提供名义区间作为适用边界诊断。A 的独立误差效率界没有移用。
- 主要覆盖率用所有重复作分母，区间失败计为未覆盖。另有仅有效区间覆盖率、Median SE / empirical SD 校准、逐次失败原因及计数。有限的极端值计入误差汇总。
- 每阶段的逐次 CSV、汇总 CSV 和图表分别位于 `results/<mode>/replications`、`summary` 和 `figures`。正式输出恰好三张矢量 PDF，每个场景一张。报告位于 `reports`。

[Smoke 报告](reports/Scenario_D_Smoke.md) · [测试与校验](reports/Scenario_D_Validation.md) · [依赖版本](requirements.txt)


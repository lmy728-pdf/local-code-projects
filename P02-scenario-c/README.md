# Rank-DML 论文 Scenario C

C1 复现论文 Proposition `prop:lsfail` 中的确定性罕见区域误差。C2 在有 \(t_3\) 重尾协变量的尾部设置阶跃，使用 Scenario A 的 Spline Ridge 真实拟合 D 与 Y。五个样本量为 500、1000、2000、4000、8000，正式阶段每设计每档 2,000 次。两种设计均比较 OLS、Rank-W、Huber、LAD。

**正式阶段已完成。** C1 与 C2 各 10,000 次。

[正式结果报告](结果_MD/Scenario_C_Results.md) · [完成核对](结果_MD/Scenario_C_Completion_Report.md) · [四张 PDF](结果_MD/figure_index.md) · [性能报告](结果_MD/performance_benchmark.md)

[运行配置](experiment_manifest.json) · [运行 Notebook](代码/Scenario_C.ipynb) · [源论文副本](<sources/Rank DML.tex>)

在 VS Code 打开本目录，选 Python 3.11 内核。已运行环境为 Windows 的 `environment2`。依赖版本保存在 [依赖清单](代码/requirements.txt)。Linux 也使用 spawn 多进程。正式设为最多 12 个 worker，每个数值库单线程。每次只跑一个条件。

在终端进入 `代码`，依序执行以下命令。已有提交记录会由 `--resume` 跳过。

~~~sh
python check.py math
python run.py --stage smoke --workers 12 --resume
python check.py smoke
python run.py --stage pilot --workers 12 --resume
python check.py pilot
python run.py --stage main --workers 12 --resume
python check.py main
python report.py
~~~

跑指定单组时给 `--case C2 --n 8000`。可用 `--reps` 先跑该组的前若干次。完整跑法和单组跑法使用同一 stage 的相同 ID 与种子。中断后带 `--resume` 继续。强制关闭进程后，如果看到残留的 `logs/running.lock`，先确认其中 PID 已退出。把锁文件移开后可继续。

## 文件

- `汇总表_CSV/` 存正式逐次及汇总 CSV、pilot CSV、优化对照和性能测量。
- `结果_MD/` 存科研分析、完成报告、性能报告及图表索引。
- `figures/` 存四张 300 dpi PDF。
- `checkpoints/{stage}/{case}_n{n}.sqlite` 存逐次元数据，以及验证折、尾部标签和有符号的个体 nuisance 误差。
- `logs/` 存进度、耗时与完整性检查。

逐次表以 `case,n,rep,method` 标识方法结果。种子由版本号、阶段、设计、样本量、重复 ID 和用途确定。C2 的逐折条件 RMSE 来自独立于模拟样本的分层 Sobol 诊断点。两区各 1,024 点，按总体区域概率合成全局范数。全部模拟样本个体误差可用 `from report import load_arrays` 后调用 `load_arrays("main", "C2", 8000, 0)` 读取。

区间为普通 95% Wald 区间。OLS 和 Huber 使用折截距与 sandwich，Huber 的每折 MAD 由外层训练数据内三折残差产生。LAD 使用 Scenario A 的线性规划及 KDE sandwich。Rank-W 用直接成对凸目标，五点差分半宽为 2。正性规则触发时保留点估计与区间失败的区别。

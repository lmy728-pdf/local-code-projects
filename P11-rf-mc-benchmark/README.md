# 独立 RF Monte Carlo CPU Benchmark

完全独立于 Rank-DML，无 GPU、无外部数据。默认 12 个单线程 worker，Windows/Linux 统一使用 spawn 多进程。每次 replication 为 5-fold × 2 个目标 × 500 棵树；残差 OLS 含截距，报告斜率。θ 仅检查工作负载，不评价统计方法。

## Notebook 使用

将整个文件夹上传到矩池云，打开同目录 `run_benchmark.ipynb`，选择目标 Python kernel，顺序执行：

1. 环境单元格检测缺失依赖，并通过当前 kernel 的 pip 安装；测试数值库/RF，保存环境版本。
2. 默认 n=1000/reps=40/trees=500/workers=12。改 `MODE='suite'` 才会顺序跑全部五组，不同时跑多组。
3. 正式任务通过独立 Python 子进程启动，支持 Windows notebook，不依赖 notebook 内函数的 pickle。

首次安装包需要网络（只下载依赖，不下载数据）。已安装依赖时不访问网络。安装失败/包损坏时明确报错，不静默升级既有环境。建议独立环境、Python 3.10+，优先 3.11/3.12。环境检查不计入正式计时。

## 矩池云 Linux

在上传目录的终端执行：

```bash
cd ~/rf_mc_benchmark  # 改为实际上传路径
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python setup_env.py
# 可选：给 VS Code/Jupyter 注册环境
python -m pip install ipykernel
python -m ipykernel install --user --name rf-mc-benchmark --display-name "RF MC Benchmark"
bash run_benchmark.sh --n 1000 --reps 40 --workers 12 --trees 500 --run-id cpu_compare
```

若系统没有 venv，可用 `conda create -n rfbench python=3.12 -y` 后 `conda activate rfbench`。脚本使用当前 PATH 中的 python3，也可设置 `PYTHON=/完整路径/python bash run_benchmark.sh ...`。

五组顺序比较（同一 output 和 run-id）：

```bash
for w in 12 16 24 32; do
  python benchmark.py --n 1000 --reps 40 --workers "$w" --trees 500 --seed 20261117 --run-id cpu_compare
done
python benchmark.py --n 2000 --reps 40 --workers 24 --trees 500 --seed 20261117 --run-id cpu_compare
```

n=2000 不会借用 n=1000 的 baseline；需要其加速比时另跑 n=2000/reps=40/trees=500/workers=12。

## Windows

在 Anaconda Prompt 中：

```bat
conda create -n rfbench python=3.12 -y
conda activate rfbench
cd /d C:\Users\Lenovo\Desktop\rf_mc_benchmark
python setup_env.py
run_benchmark.bat --n 1000 --reps 40 --workers 12 --trees 500 --run-id cpu_compare
```

可用现有环境。本机默认 python 若为商店占位入口，请选 VS Code 的 Conda kernel，或在 CMD 中 `set "PYTHON=D:\Anoconda\envs\environment2\python.exe"` 再执行 bat。PowerShell 调用完整解释器路径时使用 `& '路径' benchmark.py ...`。

## 参数、断点与公平比较

默认参数：`--n 1000 --reps 40 --workers 12 --trees 500 --seed 20261117 --run-id default`，可指定 `--output`、`--force`。默认 output 是脚本同目录 results；显式相对路径以当前工作目录为基准。

- SeedSequence 只取决于 master seed 和 replication 编号；数据、fold、10 个 RF 分别使用子流，与 workers、任务顺序、run-id 无关。跨机器比较需匹配 Python/NumPy/scikit-learn 版本。
- 输出为 `output/run-id/n..._r..._t..._w..._s.../`；不同 workers 不共用 checkpoint，完整执行同一批随机任务。
- 重复相同命令仅补未记录的任务。已记录成功/失败均不自动重跑；`--force` 将旧测量移入 `_archive`，完整重测。新 run-id 也是完整重测。
- 每完成一次 replication 原子保存 checkpoint 和 CSV。断电可能丢失正在执行/尚未写入的任务，下次重做。禁止同时启动同一配置。异常退出遗留 `.lock` 时，确认旧进程停止后手动删除该锁。
- 单任务异常记录失败并继续。进程被系统杀死/进程池损坏时，未完成的 future 记录失败。失败退出码为 1，已经完成的结果保留。
- speedup 仅针对同机器、环境、代码、run-id、n/reps/trees/seed 相同的完整无失败、单次连续测量。续跑汇总会话时长，但不参与 speedup。断电会话只知道最后 checkpoint 时长。baseline 后运行也会刷新同批次报告。
- workers 是进程上限。报告实际任务执行区间峰值重叠数、逻辑 CPU、affinity、可读时的 cgroup v2 配额；均不等于独占物理核心。40 reps 在 24/32 workers 下存在尾批效应，这是本批次真实吞吐量的一部分。
- 所有耗时用 perf_counter。总时间含进程启动、调度、计算、checkpoint、退出；不含安装、环境查询和最终报告/绘图。单任务耗时不含队列等待，分位数只用成功任务。
- 内存每 0.2 秒采样父子进程 RSS 合计，是近似峰值，可能重复统计共享页、漏掉瞬时峰值；不可测时留空。
- 同时不要运行其他重计算任务。两机同配置的 replications/min 之比是跨机器吞吐量倍数；报告 speedup 专指本机 workers=12 baseline。不要跨机器共用同一个结果目录。

## 输出与环境保存

每组输出 `replications.csv`、`benchmark_summary.csv`、`benchmark_report.md`、`checkpoint.json`、`replication_durations.pdf`（矢量图，报告含索引）。CSV 为 UTF-8 BOM。保留 θ、耗时、种子标识、状态、错误、进程 ID 和执行区间，不保存大型中间数组。

环境检查保存到 `environment_snapshot/`：

- `benchmark-versions.txt`：核心包锁定版本。新环境可先 `python -m pip install -r environment_snapshot/benchmark-versions.txt`。
- `pip-freeze.txt`：完整当前环境快照；可能含本地路径/平台特有包，不保证跨系统可直接复用。
- `environment.json`：Python 和平台信息。

安装 ipykernel 后再跑一次 setup_env.py 可更新快照。矩池云的保存环境/镜像操作请在环境检查通过、kernel 能运行后使用平台自身功能；本程序只保存依赖快照。

## 极小 smoke test（非正式性能结果）

```bash
python benchmark.py --n 200 --reps 2 --workers 2 --trees 10 --run-id smoke
```

预期 2 个成功任务、进度 2/2、输出齐全；没有匹配 workers=12 baseline 时 speedup 留空。改 workers=1 可验证两行 θ 完全一致。小任务启动开销大，不能据此判断正式并行性能。

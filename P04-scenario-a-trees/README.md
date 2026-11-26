# Scenario A · Random Forest 与 Gradient Boosting

## 矩池云 Notebook

1. 把桌面压缩包上传到实例的 `/mnt`，在该目录解压。
2. 在 VS Code Remote-SSH 打开 `/mnt/Rank_DML_Scenario_A_Trees`，打开 `Scenario_A_Trees.ipynb`，选用 Linux 服务器上的 Python 3.11 或 3.12 内核。
3. 第一格会列出当前内核及依赖版本。已有内核可以直接使用。需要补齐依赖时，在第一格把 `INSTALL_REQUIREMENTS` 设为 `True`。如果 VS Code 没有发现内核，请在现有 Conda 环境运行以下命令，再重新选择内核。

~~~bash
cd /mnt/Rank_DML_Scenario_A_Trees
python -m pip install -r requirements.txt
python -m ipykernel install --user --name rank-dml-trees --display-name "Rank-DML Trees"
~~~

第二格默认停在 `START_RUN = False`。准备启动时选择 `LEARNER` 和 `MODE`，再设为 `True` 并运行。每组显示已保存次数和后台任务数。任务中断后重跑相同配置会续跑。第三格生成汇总。

## 模式和阶段

| 参数 | 作用 |
| --- | --- |
| `LEARNER = "rf"` 或 `"gbdt"` | 每次选择一个学习器 |
| `MODE = "smoke"` | 默认 Gaussian、n=500、2 次，使用正式的学习器配置 |
| `MODE = "pilot"` | 每个选定条件 100 次，种子独立 |
| `MODE = "main"` | 每个学习器 3 个样本量 × 9 个分布 × 1,000 次 |
| `STAGE = "1"` | 训练树模型，写入第一阶段缓存和诊断 |
| `STAGE = "2"` | 读取第一阶段缓存，计算原九种估计量和区间 |
| `STAGE = "both"` | 先完整处理选定范围的 Stage 1，再运行 Stage 2 |

`N = None` 和 `LAW = None` 表示使用该模式的默认范围。正式样本量为 500、1000、2000。分布键见 `experiment_manifest.json`。默认 12 个 worker，实际进程数受 CPU 配额和任务数量限制。每个 worker 的数值库使用一线程，RF 内部 `n_jobs=1`。

项目应留在 `/mnt` 下。缓存和结果写在项目目录内。迁移时在任务停止后复制整个目录，保留代码、配置、`nuisance_cache/` 和 `checkpoints/`，并在目标环境安装同版本依赖。两个学习器的正式缓存和结果需要数 GiB，开始前确认网盘空间。数据库中的每次重复均有独立主键，已保存的失败记录同样保留。

## 科学设置和复现

- RF 使用 500 棵树、最小叶节点 5、squared-error 回归、bootstrap=True、max_features=1.0。GBDT 为 scikit-learn 的 `GradientBoostingRegressor`，上限 200、深度 3、学习率 0.05。early stopping 在传入的训练样本内留出 10%，patience=10、tol=0.0001。两种响应分别训练。
- 各外层验证折只用于预测与验证诊断。Huber 阈值沿用原来外层训练集内三折校准的规则，内部 nuisance 使用当前学习器。每次重复有 10 次外层模型拟合和 30 次 Huber 内层模型拟合。所有二阶段可行方法读取同一组外层预测；Efficient Oracle 使用真实 nuisance。
- 保留原 `methods.py`、`legacy_core.py` 和已验证的总体积分信息。原第二阶段包括 Rank-Ad 的 5 点斜率、修正信息、选阶、Wald 和 max-t。两种无限方差分布仍用平方损失做第一阶段压力测试。
- 正式种子清单包含 Spline Ridge 中前 1,000 个 ID 的 27,000 个条件内重复。其中 10,000 个复用较早的 Scenario A 生成器，按历史种子选择原路径。包括额外历史 `sln100_tau050` 的原生成规则。RF 与 GBDT 共享数据、外折和诊断参考网格，树模型随机流按学习器、训练折和响应派生。
- RF 与 GBDT 逐次比较以完整 `X,D,Y,folds` 的样本和折哈希为核验依据。Spline Ridge 历史工件保存的是残差缓存及种子，缺少完整原始输入的联合哈希。三学习器汇总采取聚合比较，Spline Ridge 的 n=1000 基准沿用已发布的 R=2000。
- 每次重复生成 10,000 个独立于训练样本的 X，用同一参考网格评估五折。记录真实 m、ℓ 的预测 RMSE，令 `b=mhat-m0`、`a=ellhat-ell0-b-c`，其中 `c` 是参考网格上 `ellhat-ell0-b` 的折内均值。记录 alignment RMSE、`sqrt(n)*RMSE(b)*RMSE(a)` 和中心化后的有符号协方差。训练与外层验证的观测预测误差另列，GBDT 训练误差基于传入估计器的外部训练集，涵盖内部 early-stopping 留出部分。

## 文件和其他运行入口

- `nuisance_cache/<mode>/<learner>.sqlite` 包含两列预测、原响应、外折、Oracle 残差、Huber 尺度、实际迭代数、种子、诊断与校验值。
- `checkpoints/<mode>/<learner>.sqlite` 保存九种方法的逐次结果和失败状态。
- `results/<mode>/` 保存汇总表、逐次 CSV、选阶频数、折诊断、配对验证和报告。主表包含绝对与相对误差、覆盖率、MCSE、区间长度和 SE 校准。
- `figures/<mode>/` 保存三张核心矢量 PDF。图以 n=1000 为主，其他样本量看 CSV。Smoke 和 pilot 的输出在各自目录。
- Windows 的 BAT 从已激活的 Python 环境运行。可为 `run_smoke.bat`、`run_pilot.bat`、`run_main.bat` 及 `resume_main.bat` 传入 learner、stage 两个参数。其默认值为 rf、both。报告用 `generate_report.bat`，可传入模式。
- Linux 命令行入口示例，适合单独读缓存生成第二阶段结果。

~~~bash
python run.py --mode main --learner gbdt --stage 1 --workers 12 --resume
python run.py --mode main --learner gbdt --stage 2 --workers 12 --resume
python report.py --mode main
~~~

## 交付前校验

已检查 27,000 条正式种子、科学代码哈希、学习器参数、训练折传入范围、缓存格式和续跑。单元测试使用模拟拟合，通过了训练折隔离、缓存复用与续跑校验。另用测试数据生成并检查了三张矢量 PDF 和 CSV。Smoke、pilot 和 main 等待手动启动。实际机器上的时间和多 worker 数值一致性需要在用户启动 smoke 后确认。

Linux 的运行锁在实例本机的 `/tmp`，SQLite 缓存和结果仍在项目目录。运行同一个项目时请使用同一台实例。

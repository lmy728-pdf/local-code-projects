# 本机代码项目

11 项代码按原清单编号归档。各目录保留原项目的子目录结构。

| 编号 | 项目 | 内容 |
| --- | --- | --- |
| P01 | [Scenario A · Spline Ridge](P01-scenario-a-spline-ridge/) | 27 组模拟，覆盖不同样本量和误差分布 |
| P02 | [Scenario C · 罕见区域误差](P02-scenario-c/) | C1/C2 比较 OLS、Rank-W、Huber 和 LAD |
| P03 | [Scenario D · 估计目标](P03-scenario-d/) | D1–D3 异质性模拟，含 D3 Sandwich 补算 |
| P04 | [Scenario A · RF / GBDT](P04-scenario-a-trees/) | 树模型实验、GBDT 测速、RF 二阶段续跑 |
| P05 | [RF 二阶段抽样 200](P05-rf-stage2-sample/) | 二阶段抽样运行脚本和 Notebook |
| P06 | [Rank-DML · G0–G6](P06-rank-dml-g0-g6/) | 八组实验代码，含 G6A、G6B 和复现辅助脚本 |
| P07 | [Rank-DML · 原工作目录](P07-rank-dml-working/) | 包含 G0–G3 冻结源码、工作版和核验脚本 |
| P08 | [DML 旧版实验合集](P08-dml-legacy/) | 22 个历史分支，含 Dukes、Huber、Spline Ridge 等 |
| P09 | [牛津项目 · 重尾 DML](P09-oxford-heavy-tail-dml/) | Student-t、Lognormal、正态基准和 Dukes 预试 |
| P10 | [Prithvi / CNN 灾害预测](P10-prithvi-cnn/) | Prithvi 训练与预测，另有 CNN 灾害分类实验 |
| P11 | [RF Monte Carlo Benchmark](P11-rf-mc-benchmark/) | 随机森林 Monte Carlo CPU 并行性能基准 |

仓库包含源码、清除运行输出的 Notebook、依赖清单和相关运行配置。实验产物、缓存、模型权重、数据库和参考论文保存在原目录。P04 附带运行入口引用的基线表。原 README 中指向本机素材或实验输出的链接可能需要本地文件。

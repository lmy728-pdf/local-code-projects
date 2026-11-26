# 毕设正式论文 numerical simulation

**Scenario A — Spline Ridge：27组全部完成，共36,000次正式模拟。**
RF与GBDT尚未纳入本次结果；结果均为scalar。

## 看结果

- [综合报告](结果_MD/A_Spline_综合报告.md)
- [正式主表CSV](汇总表_CSV/main/main_results.csv)
- [综合图索引](汇总表_CSV/综合图索引.csv)
- [全部PDF图表索引](结果_MD/图表总索引.md)
- [分组结果索引](汇总表_CSV/分组结果索引.csv)
- [完成情况](汇总表_CSV/main/experiment_completion_matrix.csv)
- [正式结果核对](结果_MD/final_validation_report.md)

## 文件位置

| 内容 | 统一目录 |
|---|---|
| 综合与逐次CSV | 汇总表_CSV/main/ |
| 各组CSV | 汇总表_CSV/分组结果/ |
| 全方法全分布综合图 | 图/综合图/ |
| 跨n趋势与诊断图 | 图/main/ |
| 各组PDF | 图/分组结果/ |
| 综合报告与图索引 | 结果_MD/ |
| 各组报告 | 结果_MD/分组结果/ |

分组实验/ 中保留各组notebook、原始SQLite记录和原有正式输出；根目录上述位置是统一阅览入口。
PDF是正式矢量图，PNG只是预览。测试输出已清理。

## 代码与复现

- [27个独立Notebook入口](开始运行_实验索引.md)
- [仅汇总已有结果的Notebook](代码/汇总已完成结果.ipynb)
- [数值实现约定](config/implementation_conventions.md)
- [实验配置](config/experiment_manifest.json)
- [当前环境版本](config/environment/versions.json)
- [锁定依赖](代码/requirements.txt)

默认12个单线程worker，每次只运行一组；后续模拟由用户手动启动。
完整复现需保留本文件夹中的公共代码、sources、config及cache/reused_inputs.sqlite。
原始结果位于分组实验/各组/代码目录/_work/replications/records.sqlite。

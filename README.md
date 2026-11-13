# 跨模态知识迁移：当前工作区

**2026-11-13 ebcloud 迁移入口：** [当前主线的代码、资产版本与运行方式](EBCLOUD_README.md)。目标为 `/root/data/xmodal`，使用规则 v7、实体修复版 1814 组及 8 个最终 LoRA；下方实验说明保留其原有日期语境。

**2026-11-04 实验梳理入口：** [研究主线、现有证据与待解决问题](reports/实验梳理_20261104/README_ZH.md)，另有[全部实验目录登记表](reports/实验梳理_20261104/experiment_registry.csv)和[评测版本表](reports/实验梳理_20261104/evaluation_versions.csv)。已覆盖全部22个顶层运行目录，区分训练、评测、解析和追加推理。**下文保留为2026-10-27历史快照**，其中“最新”“当前”均指该快照；不代表11月的完整状态。运行中进度需查看对应任务状态文件。

整理日期：2026-10-27。当前实验的统一入口是[实验数据汇总](runs/experiment_summary_20261027/README_ZH.md)，包含指标、训练检查点、未完成项及来源核验。

## 最新结果

最新规则全参数 SFT（TEXT/VISION 各10 epoch）已完成四方向评测，每方向2,000题，无无效输出：

| T2T | T2V | V2T | V2V |
|---:|---:|---:|---:|
| 26.05% | 26.80% | 25.85% | 26.65% |

采用 [batch4 完整结果](runs/rules_full_sft_ep10_20261026/evaluation_strict_b4/summary.md)；两个最终 checkpoint 均保留。该轮尚无相同配置的新基座对照，不能据历史基座直接计算本轮训练增益。

规则/实体 × SFT/PT × TEXT/VISION 的8个 LoRA 均已完成5 epoch训练。全量 CoT 已保存33,990/42,180条，完成15/20个方向，当前暂停；同协议基座尚未完成。严格FINAL、纯解析和追加读取成绩在总汇中分列。

64规则、三个seed的 TEXT 过程学习实验已完成：原题文字推理均100%，图片推理均值53.13%。其范围与完整1,000规则实验分别解读。

## 当前数据

| 类别 | 当前数据根目录 | 学习材料 | 每方向评测 | 原图 |
|---|---|---:|---:|---:|
| 规则类 | [data/rules_repaired_single_image_v7](data/rules_repaired_single_image_v7/README.md) | 1,000 条 | 2,000 题 | 3,000 张 PNG |
| 实体类 | [data/entity1020/merged_1109](data/entity1020/merged_1109/README.md) | 1,109 条 | 2,218 题 | 2,218 张 |

四方向为 T2T、T2V、V2T、V2V，TEXT检查点提供T2T/T2V，VISION检查点提供V2T/V2V。图片路径以各自数据根目录解析；顶层同名目录仅保留指引。实体版包含21组修复候选，语义及时间新颖性认证未完成；规则数据也保留独立语义认证和104道复合记录题的解释边界。详见数据README和[研究协议](CROSS_MODAL_RESEARCH_PROTOCOL.md)。

## 运行结果

详细入口见[运行结果索引](runs/README.md)及[产物清单](ARTIFACT_INVENTORY_ZH.md)。规则、实体、全量实验及小规模诊断分别统计，无效答案计错，未完成方向留空。

- [中文总报告](runs/experiment_summary_20261027/README_ZH.md)
- [指标 CSV](runs/experiment_summary_20261027/metrics.csv)、[训练 CSV](runs/experiment_summary_20261027/training.csv)、[结构化 JSON](runs/experiment_summary_20261027/summary.json)
- [按规则/实体分别报告](runs/xmodal_dataset_reports/README_ZH.md)
- [核验结果](runs/experiment_summary_20261027/validation.json)与[来源 SHA256](runs/experiment_summary_20261027/sources.json)

只整理现有产物的复用命令：`python scripts/summarize_current_experiments.py --output runs/experiment_summary_20261027`。该命令不启动训练或评测。

## 代码与环境

环境与基础模型见[RUNTIME_SETUP_ZH.md](RUNTIME_SETUP_ZH.md)。当前8个正式LoRA和新全参数检查点均在本地；迁移前的历史正式训练权重仍未恢复。历史文档提到的`废弃/`归档及PPT未包含在当前快照中。

`instruct.md`及旧manifest保存历史命令和原服务器路径；当前入口以本页及运行结果索引为准。本次未移动实验目录、改写原始预测或恢复暂停任务。

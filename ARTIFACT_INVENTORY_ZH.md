# xmodal 现有产物清单

**2026-11-04 补充：** 最新的实验身份与证据梳理见[实验总入口](reports/实验梳理_20261104/README_ZH.md)和[目录登记表](reports/实验梳理_20261104/experiment_registry.csv)，覆盖新版实体、定义扩展和MMKU等后续工作。下文保留为2026-10-27的产物历史快照，数值与“当前”状态均按当时理解，不与新实验混合使用。

首次盘点：2026-10-24；本次更新：2026-10-27。当前指标和进度以[实验数据总汇](runs/experiment_summary_20261027/README_ZH.md)为统一入口，另有[指标CSV](runs/experiment_summary_20261027/metrics.csv)、[检查点CSV](runs/experiment_summary_20261027/training.csv)及[核验结果](runs/experiment_summary_20261027/validation.json)。历史实验目录及原始预测未移动或覆盖。

## 一眼看全

| 产物组 | 当前内容 | 状态 |
|---|---|---|
| 规则类数据 | `data/rules_repaired_single_image_v7/`：1,000条规则、四方向各2,000道题、3,000张原图 | 本地数据齐备；数据验证通过，104道题的复合记录任务及图像语义认证边界见下文 |
| 实体类数据 | `data/entity1020/merged_1109/`：1,109组知识、四方向各2,218道题、2,218张图 | 数据、标签和来源审计齐备；完整语义与时间新颖性认证尚未通过 |
| 完整规则实验 | `runs/v7_full_experiment/`、`runs/v7_fullparam_sft_ep10_20261019/` | 评测及报告完成，正式权重不在当前工作区 |
| 实体基座实验 | `runs/entity1020_merged1109_qwen3_base_20261020/` | 历史短答案基座评测完成；新实体LoRA已有训练后CoT结果，见下方新全量评测 |
| 规则过程学习 | `runs/v7_process_confirmation_20261020/`、`runs/v7_process_scale64_20261021/` | 8规则确认与64规则三seed实验均完成，属于TEXT学习小规模结果 |
| 诊断实验 | `runs/v7_cot100_20261021/`、`runs/v7_process_image_diagnostic_20261021/` | 辅助诊断完成；不能替代完整数据的正式四方向结果 |
| 代码与测试 | `scripts/`、`tests/` | 含正式流程、审计、探索性实验代码及新增环境验证入口；11个顶层Python测试文件 |
| 新环境与模型 | `.venv/`、`model_cache/Qwen3-VL-4B-Instruct/` | ms-swift 4.5.3环境与完整Qwen3基础模型已准备；见[环境说明](RUNTIME_SETUP_ZH.md) |
| 新执行检查 | `runs/environment_smoke_20261024/` | 6个小训练任务、8个推理任务全部通过，新增6个两步LoRA检查点；不作为正式迁移结果 |
| 新全量 LoRA 学习 | `runs/qwen3_lora_learning_ep5_20261024/` | 2026-10-24 20:41北京时间全部完成：规则/实体 × SFT/PT × TEXT/VISION，共8任务，各5 epoch；8个最终adapter及40个epoch检查点均保留，见[状态](runs/qwen3_lora_learning_ep5_20261024/status.json) |
| 新全量 CoT 评测 | `runs/qwen3_lora_cot_eval_20261025/` | 已暂停；33,990/42,180条保存，15/20方向完成；根summary仍为14方向，应读取各方向report。见[状态](runs/qwen3_lora_cot_eval_20261025/status.json)和[统一汇总](runs/experiment_summary_20261027/README_ZH.md) |
| 新全参数 SFT | `runs/rules_full_sft_ep10_20261026/` | TEXT/VISION各10epoch完成，两个最终权重本地齐备；四方向26.05%/26.80%/25.85%/26.65%，见[batch4评测](runs/rules_full_sft_ep10_20261026/evaluation_strict_b4/summary.md) |
| 输出修复及回忆v2 | `runs/answer_output_repair_20261026/`、`runs/qwen3_lora_recall_apply_v2_20261026/` | 均暂停；追加读取剩276条，回忆完成3/10个配对方向；纯解析与追加推理分列。回忆v1因规则名错误已作废 |

## 两套数据

### 规则类：受控的规则学习与应用

数据入口：[规则类数据 README](data/rules_repaired_single_image_v7/README.md)（`data/rules_repaired_single_image_v7/`）；四方向数据在`T2T/`、`T2V/`、`V2V/`、`V2T/`，共用学习材料在`shared/`，规则、标签、题目和图像审计在`audit/`。可先看[验证报告](data/rules_repaired_single_image_v7/validation_report.json)和[修订说明](data/rules_repaired_single_image_v7/repair_report_zh.md)。

每种学习方式1,000条材料，每种测试方式2,000题。四方向矩阵中的学习/测试文件按设计两两复用，改变的是学习模态或测试模态，不是另造四套知识。

当前版本保留3,000张PNG原图。数据构建验证通过，独立重算标签的2,000题均有唯一答案，图像哈希未变。但v7将104道二值直接题改成“逐条件检查并选择记录代码”的复合任务；代码约定由学习文字/配文提供。验证文件明确记录没有独立人工批准，原图哈希不能认证图中材质判断。正式汇总时应保留这104题及其偏差审查项，解释准确率时也应说明其任务范围。

### 实体类：实体绑定的事实知识

数据入口：[实体类合并版 README](data/entity1020/merged_1109/README.md)（`data/entity1020/merged_1109/`）；包含四方向问答、`shared/`学习输入、`source/`事实来源和`audit/`标签及单位审计。数据状态详见[验证报告](data/entity1020/merged_1109/verification.json)和[数据摘要](data/entity1020/merged_1109/summary.json)。

当前合并版由1,088组基础条目和21组修题候选组成，共1,109组知识、每方向2,218题和2,218张图。选项按seed 42平衡；来源、图像和标签结构验证通过。但数据清单明确标记语义和时间新颖性认证未完成；21组候选待复核，1,088组也不是已认证无问题的数据。

实体类4个LoRA均已完成5epoch学习；SFT的四方向CoT已完成，PT已完成T2T/T2V/V2T，V2V仅保存512/2218条。严格FINAL存在大量格式失败，纯解析与追加读取后的成绩见[实体报告](runs/xmodal_dataset_reports/entities/report_zh.md)。历史基座文字28.64%、图像27.55%来自四轮选项轮换短答案评测，协议不同，不作为新CoT的同条件基座对照。

## 运行结果与报告入口

| 运行目录 | 产物范围 | 完成情况与阅读入口 |
|---|---|---|
| `runs/v7_full_experiment/` | 规则类全量1,000条；3个模型×SFT/PT×TEXT/VISION；48个strict评测单元、96,000条回答 | 训练、评测均完成；CoT未运行。先看[报告](runs/v7_full_experiment/summary/report_zh.md)、[对比表](runs/v7_full_experiment/summary/comparison.csv)、[完成状态](runs/v7_full_experiment/final_status.json)。保存逐题`raw.jsonl`、`predictions.jsonl`、`results.jsonl`和checkpoint清单。完整训练权重不在当前工作区。 |
| `runs/v7_fullparam_sft_ep10_20261019/` | Qwen3全参数SFT；TEXT、VISION各10 epoch；各方向训练前后对照 | 完成审计，训练后四方向约25%。看[报告](runs/v7_fullparam_sft_ep10_20261019/report_zh.md)、[对比表](runs/v7_fullparam_sft_ep10_20261019/comparison.csv)、[摘要](runs/v7_fullparam_sft_ep10_20261019/summary.json)、[完成审计](runs/v7_fullparam_sft_ep10_20261019/completion_audit.json)。有预测、逐题结果、loss图和审计文件；当前未见完整模型权重。 |
| `runs/entity1020_merged1109_qwen3_base_20261020/` | 实体类Qwen3未训练基座；每种测试模态四轮选项轮换 | 17,744条生成完成，标签与选项平衡审计通过。看[报告](runs/entity1020_merged1109_qwen3_base_20261020/reports/报告.md)、[统计](runs/entity1020_merged1109_qwen3_base_20261020/reports/statistics.json)、[验证](runs/entity1020_merged1109_qwen3_base_20261020/verification.json)。 |
| `runs/v7_process_confirmation_20261020/` | 8规则、1 seed、TEXT学习；定义/定义复习/具体过程对照 | 2,656条生成和独立审计完成，来源应用门槛通过。看[解释边界](runs/v7_process_confirmation_20261020/INTERPRETATION_ZH.md)、[分析](runs/v7_process_confirmation_20261020/analysis_zh.md)、[配对比较](runs/v7_process_confirmation_20261020/paired_comparisons.json)及[人工错误审查](runs/v7_process_confirmation_20261020/manual_error_review.json)。只覆盖小规模规则集合。 |
| `runs/v7_process_scale64_20261021/` | 64规则、3 seeds、TEXT学习；定义、定义复习和过程训练 | 已于2026-10-22完成。5,312道不同探针由10个评测臂生成，共53,120条回答；独立重评分及数据审计通过。看[报告](runs/v7_process_scale64_20261021/report_zh.md)、[汇总表](runs/v7_process_scale64_20261021/summary.csv)、[逐规则指标](runs/v7_process_scale64_20261021/per_rule_metrics.json)、[配对比较](runs/v7_process_scale64_20261021/paired_comparisons.json)、[最终完成审计](runs/v7_process_scale64_20261021/final_completion_audit.json)。这些不是全量1,000规则，也没有VISION学习臂。 |
| `runs/v7_cot100_20261021/vllm/` | 50规则、每模态100题；基座/训练后，直接CoT与自我回忆后CoT | 900次生成及输入输出审计完成。唯一正式入口是[vLLM报告](runs/v7_cot100_20261021/vllm/report_zh.md)和[最终审计](runs/v7_cot100_20261021/vllm/final_completion_audit.json)；父目录旧引擎结果不与vLLM结果合并。回忆条件是辅助诊断。 |
| `runs/v7_process_image_diagnostic_20261021/` | 上一轮8规则模型的图片、回忆、正确规则/观察控制 | 128次诊断完成。看[分析](runs/v7_process_image_diagnostic_20261021/analysis_zh.md)、[报告](runs/v7_process_image_diagnostic_20261021/report_zh.md)、[完成审计](runs/v7_process_image_diagnostic_20261021/completion_audit.json)。事后控制用于定位问题，不替代正式闭卷四方向分数。 |

上表保留历史完成实验的入口；2026-10-24之后的新训练与评测见本页首表及[完整运行索引](runs/README.md)。不同数据集、数据规模、训练方案或辅助条件分别统计。

### 对最终研究目标的覆盖

| 数据与方案 | T2T | T2V | V2V | V2T |
|---|---|---|---|---|
| 全量规则：LoRA SFT/PT | 已完成 | 已完成 | 已完成 | 已完成 |
| 全量规则：Qwen3全参数SFT | 已完成 | 已完成 | 已完成 | 已完成 |
| 8/64规则：TEXT过程学习 | 已完成 | 已完成 | 未覆盖 | 未覆盖 |
| 实体：1,109组，SFT LoRA CoT | 已完成 | 已完成 | 已完成 | 已完成 |
| 实体：1,109组，PT LoRA CoT | 已完成 | 已完成 | 512/2218，暂停 | 已完成 |

四方向的正式提取口径是：**TEXT checkpoint提供T2T与T2V，VISION checkpoint提供V2V与V2T**。全量LoRA旧表将每个checkpoint都列成四个方向，其中同模态测试输入重复；48个评测单元包含这类重复，不能解释成48种独立实验条件。相同模型和训练目标下，组合两个学习模态才得到用户定义的一张四方向矩阵。实体基座只有文本和图片两种测试条件，不能据此声称已经测出学习后的跨模态迁移。

全量LoRA中的GLM低分还涉及输出格式解析，补充证据见[GLM格式分析](runs/v7_full_experiment/analysis/glm_output_format_20261014/report_zh.md)；原strict分数与补充解析结果应分别保留。

## 代码、测试和说明文档

- `scripts/`包含数据构建、训练协调、GPU准入、四方向评测、续跑、解析、汇总和审计工具；其中有大量历史探索脚本，并非每个脚本都是当前主流程。
- `tests/`包含11个Python测试文件，覆盖评测契约、格式解析、GPU资源账本、checkpoint存储和数据构建等。
- [CROSS_MODAL_RESEARCH_PROTOCOL.md](CROSS_MODAL_RESEARCH_PROTOCOL.md)定义研究问题和正式评测边界；[second_question_construction_guide.md](second_question_construction_guide.md)定义题目构造原则。
- `instruct.md`保存大量历史命令和实验过程，是历史记录，不是当前唯一可执行入口。
- `environment_pip_freeze.txt`与`environment_nvidia_smi.txt`是环境记录，不是可直接使用的运行环境。

此前进行的只读检查中，13项现有单元测试通过；64规则实验的53,120条输出、输入哈希和汇总曾独立核验一致。此记录不表示本次重新执行了全部测试或模型推理。

## 当前复现与迁移限制

**同日环境准备后的更新：** 已新增项目`.venv`、Qwen3-VL-4B完整基础权重和6个执行检查adapter；64条回答完成输入匹配和独立评分核查，详见[最终验证](runs/environment_smoke_20261024/final_verification.json)与[环境说明](RUNTIME_SETUP_ZH.md)。历史正式训练权重仍未恢复，其他两个基础模型尚未下载。以下缺失记录描述的是迁移包初始状态。

**2026-10-27检查点更新：** 新8个正式LoRA及两份新全参数最终模型均在本地，路径和存在性见[training.csv](runs/experiment_summary_20261027/training.csv)。这不表示迁移前的历史正式权重已恢复。

当前工作区的正式数据位于`data/`；一些协调脚本和历史manifest仍含源服务器绝对路径或旧的顶层数据路径。`MIGRATION_README.md`列出的新环境smoke步骤可作为迁移指引，但须先确认数据根路径和模型路径。

顶层`rules_repaired_single_image_v7/`只有说明文件，顶层`entity1020/`也不是完整数据入口。历史README中的“运行中”描述需要结合各实验最终审计判断：64规则实验已完成。旧文档提及的`废弃/`归档、PPT，以及图片诊断报告引用的额外计数诊断目录，当前快照均未包含，不能列为本地已交付产物。

迁移包初始状态没有`.venv/`、`model_cache/`、正式训练adapter或全参数最终权重。当时两个找到的`safetensors`均为`runs/v7_full_experiment/smoke_*/checkpoint-1/`的Qwen3 smoke adapter，不能作为正式训练产物。迁移包保留了相当完整的**数据、原始预测、逐题评分、报告、manifest与审计证据**。同日已补齐Qwen3基础模型与可运行环境，但历史正式训练权重仍需要另行恢复或迁移。

`transfer_manifest.json`是2026-10-22迁移快照的文件清单，不应视作随本清单更新的当前工作区manifest。对该清单所列7,015个文件做过整包SHA256核查，其中7,014个匹配，唯一发现差异为`scripts/build_transfer_bundle_20261022.py`；数据集和实验结果文件均匹配。新清单本身不属于当时的快照哈希范围。

## 建议的查阅顺序

1. 先读`CROSS_MODAL_RESEARCH_PROTOCOL.md`，确定四方向矩阵和结果边界。
2. 按研究对象选择规则类或实体类数据，查看各自README、verification及audit。
3. 看本次实验总汇、完整规则实验和实体CoT报告，严格区分历史基座与本轮同协议基座。
4. 看8规则与64规则过程实验，了解来源模态学习能力的后续确认，以及当前TEXT学习下的图片测试结果。
5. 查看CoT与图片诊断时，保持短答案、推理、自我回忆和事后控制各自分列。


# 实验指令（更新于 2026-10-21）

工作目录：`/mnt/yunpan2/tanjiyang/xmodal`。当前规则实验使用 `rules_repaired_single_image_v7`。下方先给出当前 v7 指令，2026-10-08 的旧命令保留在文末历史部分。

## 当前研究主线：同一模型的跨模态知识迁移

研究约束见 `CROSS_MODAL_RESEARCH_PROTOCOL.md`。TEXT学习后的同一检查点评测T2T/T2V，VISION学习后的同一检查点评测V2T/V2V；测试只给原题、原选项和对应测试图，不提供学习材料或切换到其他模型解题。先验证来源模态应用是否学会，再判断跨模态增益；同一模型自身推理可单列评测，基座须使用相同协议。

下列记忆路由、冻结基座解题及视觉文字桥接命令保留作为历史诊断实验。其高分不代表单模型迁移已修复，也不是当前主方案。历史完整1000条10epoch单模型实验的四方向仍约25%；下方8规则具体过程确认已改善来源文字应用，但尚未验证完整1000条或总体跨模态迁移。

## 2026-10-21：100题/模态 CoT扩大诊断

目录：`runs/v7_cot100_20261021`。使用已完成的64规则定义SFT、seed42、3epoch最终`checkpoint-3840`，另设未训练基座。按固定哈希顺序从已学规则中选50条（OR/区间/AND/排除各13/13/12/12），每条保留原direct和diagnosis各一题，文字100题、对应原图100题。四类集合选择规则不代表完整1000条；也不是上一轮8规则atomic检查点的直接扩样。

每组运行直接CoT 200次、仅规则名的自身回忆50次、同一模型使用原样回忆后CoT 200次，共450次，两组900次，已全部完成。正式结果位于`runs/v7_cot100_20261021/vllm/`：统一vLLM 0.11.0、temperature0、seed42、连续批处理并发上限32、CoT最多3072 tokens、回忆768 tokens。Instruct使用正常assistant模板（`enable_thinking=False`），通过指令要求显式解释，不强制加入`<think>`。

原Swift/Transformers batch16推理过慢，部分输出保留在父目录base/trained，不进入正式报告。vLLM动态LoRA与视觉模块冲突的失败smoke保存在`implementation_revisions/vllm_unmerged_lora_failure/`；正式训练组使用单独BF16合并权重副本`vllm_merged_model/`，未重新训练，原模型、适配器、原题原图均不变。逐tensor合并审计验证252处语言层更新、315个视觉tensor完全不变；合并舍入可能造成与原动态LoRA不同的输出，不混合两种引擎评分。

结果（各100题，包含无效输出）：基座直接CoT文字26%、图片21%；训练后直接CoT文字43%、图片32%；基座自我回忆后CoT文字27%、图片26%；训练后自我回忆后CoT文字88%、图片55%。人工核查训练后50条独立回忆的核心条件及逻辑，48条正确（96%）。自我回忆是辅助诊断，不能将其结果当作原始闭卷训练成绩。

```bash
# 已完成；以下评测命令按条续跑，已有450条时不会重复生成
/mnt/yunpan/anaconda3/envs/qwen3vl-rollout/bin/python scripts/eval_v7_cot100_vllm.py --arm trained --gpu 0
/mnt/yunpan/anaconda3/envs/qwen3vl-rollout/bin/python scripts/eval_v7_cot100_vllm.py --arm base --gpu 1
/mnt/yunpan/anaconda3/envs/qwen3vl-rollout/bin/python scripts/audit_v7_cot100_vllm.py --check-merge
/mnt/yunpan/anaconda3/envs/qwen3vl-rollout/bin/python scripts/report_v7_cot100_vllm.py
```

准确率分母包含格式错误及截断；另报有效输出准确率、题型及结构分项，最终差值按50条规则配对bootstrap。回忆后结果有自身生成文字上下文，单列为辅助诊断。`vllm/matched_historical_short_report.json`是同一100题与同一检查点的已有Swift短答案对照，不能冒充新CoT结果或将引擎差异归因于CoT。`vllm/data_audit.json`独立解析原文字观察、计算100题答案，并检查原题/原图、训练覆盖与无应用选择题训练；`vllm/prediction_audit.json`和`engine_audit.json`核验900条原始回答、token解码、实际prompt与自身回忆来源。汇总见`vllm/report_zh.md`、`summary.csv/json`；人工错因见`text_error_manual_audit.json`和`recall_manual_audit.json`。完成状态以vllm内各组`state.json`和`launcher_state.json`为准。

## 2026-10-21：64条规则、3个seed的具体过程学习扩展

目录：`runs/v7_process_scale64_20261021`。沿用Qwen3-VL-4B的TEXT学习，固定64条规则，类别/材质OR、数值区间、多条件AND、包含＋排除各16条。按SHA256顺序选择，排除上一轮确认用的8条，不按分数筛选。三个seed为42/43/44，使用相同训练与留出文件，仅训练随机种子和打乱不同。

每个seed从干净基座开始做定义SFT，再从最终定义检查点分别继续定义复习和具体过程SFT，共9个训练任务。每任务5120行、3epoch、3840步，batch2×accum2，LoRA32/alpha64，定义LR3e-5，续训LR1e-5，冻结视觉/对齐层。过程组每规则16行事实复习、32行新场景直接答案、32行同场景具体过程。等行数/等步数不代表等监督token。

训练与留出各2048个新场景，按无序四候选集合全局隔离，并排除2149个可解析正式/历史场景。AND全KEEP的观测只有唯一一种，留出direct保留一次，另外三个对应槽位补充不同的全OMIT场景，避免复制进训练；AND训练/留出diagnosis覆盖15种KEEP组合，其他结构覆盖16种。正式原题、选项、答案、3000张PNG均保持不变；训练只用规则知识和独立新场景。

每评测组5312次生成，共基座一次和9个训练检查点、53120次生成。原题T2T/T2V各128题，短答案和显式推理分开；另报告新文字留出、训练场景探针、定义及字段回忆。每seed分别判断来源应用门槛，不以T2T是否超过T2V为目标，不按测试挑检查点。仅四种集合选择结构，无VISION学习，不能称为完整1000条或完整四方向结果。

数据、标签闭式计算、选项均衡、场景隔离、全部15360行编码/助手mask检查、基座和3000原图哈希核对已通过；定义组两步smoke及文字/图片重新加载通过。正式工作流进度以`workflow_state.json`、`queue0_state.json`、`queue1_state.json`及各任务`state.json`为准。早期独立启动已停止并保存在`prelaunch_archive/`，统一工作流从干净基座重启，未依据评测选择权重。

2026-10-21故障修订r1：seed42定义训练/定义复习已完成（2/9任务），基座5312条已完成，过程组原先因5类样例超过smoke预检查的4类上限而中断，尚无训练后对照成绩。已将过程组smoke改为3步、其余两组保留2步；5类覆盖、有限loss、LoRA权重及文字/图片checkpoint重载检查通过，重新执行数据/图像审计通过。正式训练3840步/3epoch、数据、原题原图和评测协议不变。旧脚本、manifest及失败记录保存在`implementation_revisions/r1_smoke_channel_coverage/`，当前manifest明确记录新脚本哈希。续跑复用已完成训练及基座评测。阶段基座成绩见`partial_status_zh.md`（原题短答案T2T 30/128、T2V 31/128）；推理输出大量截断，不能混同短答案或当作纯知识能力分数。

```bash
.venv/bin/python scripts/scale_v7_process_learning.py --stage prepare
.venv/bin/python scripts/scale_v7_process_learning.py --stage audit
# 可复用完成任务、恢复中间checkpoint；不要重复启动同组任务
.venv/bin/python -u scripts/scale_v7_process_learning.py --stage workflow
# 全部完成后独立重评分与权重增量核查（workflow会自动执行）
.venv/bin/python scripts/audit_v7_process_scale64.py
```

冻结方案见`PLAN_ZH.md`、样例见`TRAINING_DATA_ZH.md`、参数/输入哈希见`manifest.json`。输出包括`summary.json/csv`、`report_zh.md`、按规则/结构/题型统计、三个seed离散度、loss及`final_completion_audit.json`。训练和评测使用共享GPU准入，不终止无关进程。

### 上一轮8规则模型的图片诊断

目录：`runs/v7_process_image_diagnostic_20261021`，128次生成已完成并通过哈希审计。同一检查点原图闭卷推理5/16；给正确规则9/16，保留原图并给正确文字观测12/16，同时给两者16/16。同一模型未经修正的图片上下文回忆再放回题目仍5/16。以上全部为事后控制，不替换正式跨模态成绩。

上下文完整定义精确匹配为文字10/16、图片2/16；精确字面分数包含近义改写，不等于语义记忆正确率。原“只读图”提示对数值题使用了“displayed numeric value”，但原图用托盘内物体个数表示value，因此3/16的整体字面匹配不能直接当读图能力结论。补充明确计数与输入约定的控制保存在`runs/v7_process_count_diagnostic_20261021`，16次生成，仍与正式分数分开。

补充控制也已完成：明确计数后8张图有6张全对、30/32个候选计数正确；但全部输出平铺JSON，违反要求的嵌套value字段，冻结严格格式分数仍为0/8、格式错误8条。语义核对独立保存，未改写严格分数。仅解释value对应物体个数而不给规则/数量，原题区间选择仍2/8，与原区间闭卷相同。上下文规则回忆逐条语义复核为文字13/16正确，图片3/16正确、1条类别表述不确定；部分图片回答把区间规则换成类别/材质规则。分析与原始示例见`runs/v7_process_image_diagnostic_20261021/analysis_zh.md`，不能据此声称唯一因果机制。

```bash
# 已完成控制结果
cat runs/v7_process_image_diagnostic_20261021/report_zh.md
# 补充计数诊断，可按ID续跑
.venv/bin/python -u scripts/diagnose_v7_count_representation.py
# 汇总已完成控制、语义计数及逐条规则回忆复核，不重新生成回答
.venv/bin/python scripts/analyze_v7_process_image_controls.py
```

## 2026-10-20：新规则具体过程学习确认

目录：`runs/v7_process_confirmation_20261020`。从干净Qwen3-VL-4B开始，TEXT-only LoRA32/alpha64，先做定义SFT，再从其最终checkpoint分出“定义复习”和“具体应用过程”两组。每个任务640行、3epoch、batch2×accum2、固定480步；共同定义阶段LR3e-5，两组续训LR1e-5，seed42，冻结视觉/对齐层。不是继承旧8条规则的checkpoint。两组训练内容、监督token数不同，不称为等token预算。

按SHA256固定选出4条类别/材质OR规则、4条数值区间规则，排除扫描到的小规模历史实验规则ID。历史全量1000条实验可能涉及它们；本轮从干净基座开始，因此不继承历史权重。类别/材质域不同于旧形状/颜色实验，区间包含开闭边界，属于扩展域确认，不是完全相同条件的复现。选择依据及扫描路径见`selection_audit.json`，不按测试分数选样本。

过程组含128条事实复习、256条新场景直接答案、256条同场景具体过程。256个独立训练场景与256个留出场景按无序四候选集合隔离，并排除可解析原题/历史场景；单候选允许重复。原题、选项、标签和3000张PNG不变。评测为同一模型闭卷作答，无学习材料或其他模型解题。

四组各664次生成：基座、定义SFT、定义复习、过程SFT。主来源应用观察预先冻结的256道新文字留出显式推理与16道原题T2T推理；T2V使用相同检查点，全部短答案结果另列。每组推理均最多768token、温度0、native thinking关闭，允许模型自行解释，最终仍四选一；截断和格式错误分别记录。每方向原题仅16题，8规则/1seed，不能外推完整1000条。门槛与全部配置在`manifest.json`，不以T2T高于T2V选择checkpoint或规则。

```bash
.venv/bin/python scripts/confirm_v7_process_learning.py --stage prepare
.venv/bin/python scripts/audit_v7_process_confirmation.py
# 会复用已完成任务；不要和现有workflow同时启动同组训练
.venv/bin/python -u scripts/confirm_v7_process_learning.py --stage workflow
# 全部四组完成后汇总及独立审计
.venv/bin/python scripts/confirm_v7_process_learning.py --stage score
.venv/bin/python scripts/audit_v7_process_confirmation.py --final
```

训练可恢复保存了优化器的checkpoint，评测按ID续跑；固定最终epoch3检查点。共享显存准入ledger为`runs/v7_full_experiment/`，不终止无关进程。训练样例在`TRAINING_DATA_ZH.md`，主进度在`workflow_state.json`及各组`state.json`/`evaluation/state.json`，日志为`workflow.log`与各组`train.log`。结果生成`report_zh.md`、`summary.json/csv`、`paired_comparisons.json`、`per_rule_metrics.json`、`trace_diagnostics.jsonl`与`training_loss.png`。

<!-- v7_process_confirmation_final_results -->
本轮已完成：三次训练、四组各664次生成，共2656条，数据/最终独立审计通过，3000张原PNG哈希不变。固定使用各任务epoch3/checkpoint-480。

| 评测 | 基座 | 定义SFT | 等步数定义复习 | 具体过程SFT |
|---|---:|---:|---:|---:|
| 新文字留出，推理 | 24/256（9.4%） | 87/256（34.0%） | 88/256（34.4%） | 246/256（96.1%） |
| 新文字留出，短答案 | 71/256（27.7%） | 72/256（28.1%） | 72/256（28.1%） | 161/256（62.9%） |
| 原题T2T，推理 | 2/16 | 1/16 | 1/16 | 16/16 |
| 原题T2V，推理 | 1/16 | 2/16 | 2/16 | 5/16 |
| 原题T2T，短答案 | 2/16 | 2/16 | 2/16 | 7/16 |
| 原题T2V，短答案 | 1/16 | 2/16 | 2/16 | 7/16 |

预先固定的来源应用可行性门槛通过。定义复习/过程组新留出推理的格式错误分别112/256与1/256，低基线含大量截断，不能将全部增益归因于规则推理能力。过程组事后比较模板诊断：249/256题四候选规则参数全匹配、248/256题Result正确，短答案提升也提供支持。两组监督token为25212与54204，不能声称纯表达形式或等token因果对照。

过程组原题32条推理全部人工复核：文字最终字母16/16，完整理由一致13/16；图片最终字母5/16，完整理由一致1/16。图片存在错误调用规则、观测读取、条件执行及选项映射等混合问题，不能只凭T2T−T2V差值断言纯迁移失败。短答案两方向均7/16，不能以推理结果替换。仅8规则/1seed，无VISION学习组，不能报告V2T/V2V或外推1000条。

完整报告：`runs/v7_process_confirmation_20261020/analysis_zh.md`；训练样例：`TRAINING_DATA_ZH.md`；原题理由复核：`manual_error_review.json`；汇总：`summary.json/csv`；loss：`training_loss.png`；完成凭据：`final_completion_audit.json`。工作流先前因评测并发冲突退出，已有评测随后完成；历史退出状态另存，最终收尾使用已完成输出，没有重训或改动生成协议。

```bash
# 已完成结果的事后诊断和解释生成，不重新训练/推理
.venv/bin/python scripts/diagnose_v7_process_traces.py
.venv/bin/python scripts/report_v7_process_confirmation.py
```

## 2026-10-20：SFT / DPO / GRPO 学习方法小试

目录：`runs/v7_preference_learning_20261020`。使用干净Qwen3-VL-4B与8条固定规则，TEXT-only LoRA32/alpha64；共同规则定义SFT起点为256行、2epoch/128步。继续SFT、DPO与GRPO从同一个warmup最终权重开始，使用相同128个新四候选训练场景。SFT/DPO为2epoch/64步，GRPO为1epoch/128步、每提示4次采样；预算不同，属于方法可行性小试。

训练输出是规则条件与四个KEEP布尔值，不含正式选择题、选项或正确选项字母。DPO负例为程序构造的逻辑/执行错误；GRPO奖励为0.5精确规则+0.5候选决策正确比例，非法格式为0。测试闭卷，由同一个训练后模型直接回答文字题和原图题，不提供规则或外部解题。232个探针/组，五组共1160个回答；包含128个新文字留出、固定32个留出推理题、原始T2T/T2V各16题×短答案/推理，以及8个定义回忆。

两项预检查与早期运行修复及作废日志保存在manifest修订记录和同目录快照中：奖励将4与4.0视为等价数值，GRPO从共同起点重跑；新留出选项位置独立于KEEP组合编号随机化并保持平衡。全部训练文件与正式原题/原图未因选项随机化而改变。仅8条规则、两类结构、一个seed；不是全量1000条结果，也不以T2T是否高于T2V来选择规则或检查点。

```bash
.venv/bin/python scripts/pilot_v7_preference_learning.py --stage prepare
# 完整工作流：共同warmup、SFT/DPO/GRPO及五组评测；完成的任务自动复用
.venv/bin/python -u scripts/pilot_v7_preference_learning.py --stage workflow
# 单独续跑某组评测示例
.venv/bin/python -u scripts/pilot_v7_preference_learning.py --stage evaluate --arm grpo --gpu 1
# 五组均完成后汇总与独立审计
.venv/bin/python scripts/pilot_v7_preference_learning.py --stage score
.venv/bin/python scripts/analyze_v7_preference_learning.py
```

结果文件：`report_zh.md`、`summary.json/csv`、`paired_comparisons.json`、`per_rule_metrics.json`；训练示例为`TRAINING_DATA_ZH.md`，曲线为`training_curves.png`，GRPO逐样本奖励复算为`grpo_reward_audit.json`，完整原始回答在各组`evaluation/predictions.jsonl`。`completion_audit.json`与`independent_audit.json`均通过才表示本轮完整完成。

<!-- v7_preference_final_results -->
本轮已全部完成：主评测1160条、事后JSON诊断320条，共1480条。新文字留出短答案依次为基座38/128、共同定义SFT起点41/128、继续SFT41/128、DPO39/128、GRPO41/128；原题T2T五组均6/16，T2V依次5/16、7/16、6/16、7/16、7/16。学习后定义回忆均8/8，来源模态应用未见可靠提升，不扩展为全量1000条结论。

事后同训练格式JSON诊断：新场景规则全对依次2/32、0/32、15/32、1/32、2/32；四候选决策全对依次2/32、6/32、7/32、6/32、5/32。DPO负例未覆盖错误规则参数；GRPO有69/128组任务优势全零，训练奖励未见上升。独立参数审计确认三个分支全部504个LoRA张量更新；主图片/输入/标签审计通过。完整解释、限制及后续建议见`runs/v7_preference_learning_20261020/analysis_zh.md`。这些是本轮配置的负结果，不代表DPO/GRPO一般无效。

追加诊断与主分数分开保存；以下命令仅汇总已完成诊断，不重新训练或推理：

```bash
.venv/bin/python scripts/diagnose_v7_preference_json.py --stage score
```

追加诊断冻结配置在`posthoc_json_diagnostic/manifest.json`，逐条回答在其各组`predictions.jsonl`，汇总在`posthoc_json_diagnostic/summary.json`。总完成状态见`final_completion_audit.json`。

## 2026-10-20：具体比较过程与抽象YES/NO的匹配续训

目录：`/home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/run`。两组从上一轮multicandidate/checkpoint-480出发，各640行、3epoch/480步、LR1e-5、LoRA32/alpha64；128完整定义、96原子事实、32逻辑说明、192新场景直接答案、192同场景过程。全部输入与非过程答案、场景和顺序相同，仅过程回答是否写清观测值/规则参数/比较运算不同。TEXT-only，不改变原题、选项、标签和3000张PNG。
192个训练场景与192个新留出场景按无序四候选集合隔离，排除可解析原题和15份历史数据场景；单候选可以重复。两组监督token32969与41104，不能称为同token预算。固定最终checkpoint，不按测试选优；仅8条历史开发规则、两类规则、一个seed。

| 评测 | 起点 | 抽象 | 具体 |
|---|---:|---:|---:|
| 短答案T2T | 6/16（37.5%） | 5/16（31.2%） | 8/16（50.0%） |
| 短答案T2V | 3/16（18.8%） | 4/16（25.0%） | 3/16（18.8%） |
| 推理T2T | 11/16（68.8%） | 14/16（87.5%） | 15/16（93.8%） |
| 推理T2V | 8/16（50.0%） | 9/16（56.2%） | 8/16（50.0%） |
| 新留出短答案 | 98/192（51.0%） | 132/192（68.8%） | 130/192（67.7%） |
| 旧难题短答案 | 47/192（24.5%） | 47/192（24.5%） | 49/192（25.5%） |
| 旧难题推理（追加） | 126/192（65.6%） | 131/192（68.2%） | 183/192（95.3%） |

两种协议在训练前分别冻结：短答案32token，显式推理512token，temperature0，enable_thinking=false；同模型自己回忆/计算，无材料、检索、外部模型。每组主评测856+128=984条，三组2952条；旧难题追加诊断192×3=576条，总3528条，分开保存。
训练数据与例子：`abstract_training.jsonl` / `grounded_training.jsonl` / `TRAINING_DATA_ZH.md`。完整结果：`summary.json/csv`、`report_zh.md`、`analysis_zh.md`、`training_loss.png`、`original_comparisons.jsonl`。完整输出在各组`evaluation/predictions.jsonl`；追加诊断在`hard_reasoning/`。`completion_audit.json`通过。起点重测出现少量同输入输出漂移，当前对照使用本轮重测，不挑历史分数。

```bash
.venv/bin/python /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/pilot.py --stage prepare
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/run.py --arm abstract --gpu 0
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/run.py --arm grounded --gpu 1
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/pilot.py --stage evaluate --arm initial --gpu 1
.venv/bin/python /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/hard_reasoning.py --stage prepare
# 对initial/abstract/grounded分别运行追加旧难题诊断
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/hard_reasoning.py --stage queue --arm initial abstract --gpu 0
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/hard_reasoning.py --stage queue --arm grounded --gpu 1
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_grounded_comparisons_20261020/scripts/finish.py
```

本轮逐条复核：具体组旧难题区间94/96（起点38/96、抽象42/96），OR89/96（起点88/96、抽象89/96）；9错中4题调用错误规则、5题候选算对但报告操作/选项错。原题T2T15/16仍含一道答案正确但Saldal上限误记的题，不能把16/16真值轨迹视为规则全部正确。图题既有计数读错，也有规则调用错；短答案仍明显弱于显式过程。详见`manual_error_review.json`和`analysis_zh.md`。

## 2026-10-20：多候选应用与报告核查学习

当前开发实验目录：`/home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/run/`。训练仅文字，仍使用8条历史开发规则，不更改原题、选项、标签或3000张PNG。训练从固定的事实绑定适配器`/home/duyuntao/xmodal_runs/v7_fact_binding_20261019/run/suffix_control/checkpoint/checkpoint-480`出发；与两候选续训使用同一起点，不是继续累加上一轮训练。

640行包含128条完整定义、96条原子事实、32条逻辑/边界说明、192条新场景直接答案及192条同场景逐候选检查过程。192个训练场景与192个新留出场景按无序四候选集合全局隔离，并排除可解析原始测试和历史留出场景；单个候选可以重复。直接答案与过程监督使用同一场景，不将384行应用监督视为384个独立场景。覆盖OR四种条件组合、区间端点、全选/全不选和报告接受/纠正；不强制场景区分同族全部规则。每规则/题型正确字母均衡，正确位置及报告接受位置分别独立随机化。

LoRA32/alpha64/dropout0.05，LR1e-5，batch2×accum2，3epoch/480步，seed42，max8192，冻结视觉与对齐层，样本等权CE含结束符；匹配上轮行数与更新步数，但不宣称token预算相同。单batch检查使用两步覆盖全部五个训练通道，并重新加载checkpoint检查文字和原图输入。固定最终step480，不按测试选检查点或筛规则。

```bash
.venv/bin/python /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/pilot.py --stage prepare
# run：smoke训练/重载、正式训练、最终checkpoint评测
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/run.py --gpu 0
# 在另一张GPU依次运行，使用同一个显存准入ledger
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/pilot.py --stage evaluate --arm base --gpu 1
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/pilot.py --stage evaluate --arm initial --gpu 1
.venv/bin/python /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/score.py
```

每组664个探针：原始T2T/T2V各16题、旧版未见文字选择题192题、本轮新文字留出192题、本轮训练场景直接作答192题、定义回忆8题、字段回忆48题。正式测试均为闭卷直接字母协议，temperature0、最多32token，基座/起点/训练后相同；没有附材料或外部解题。训练过程监督不改变正式答题协议。各组原始回答在`evaluation/predictions.jsonl`，训练数据在`training.jsonl`、`TRAINING_DATA_ZH.md`；`report_zh.md`、`summary.json/csv`、`original_comparisons.jsonl`和`training_loss.png`由独立分析脚本生成，`completion_audit.json`通过后才表示整轮完成。

前置复核发现第一版生成器的答案位置与命中组合可能相关，已停止本轮自有试跑、独立随机化后重新冻结数据；未完成试跑只留存在同级`preflight_superseded_run/`，不计入结果。

<!-- v7_multicandidate_final_results -->
本轮已完整完成（8条开发规则、一个seed，非全量1000条）：主短答案原始T2T6/16、T2V3/16，基座均4/16、训练起点均5/16；旧文字留出47/192（起点同47/192），新文字留出90/192（基座49/192、起点51/192），训练场景117/192。主来源应用筛查未通过，未扩大到1000条。

另外单列闭卷显式推理诊断：新留出固定32题，本轮26/32，完整四候选轨迹23/32。原题同一最终模型T2T12/16、T2V8/16；起点同协议分别2/16（4题截断）、6/16（无格式错误）。两组均有效的12道T2T上，起点2/12、本轮8/12，增益不仅来自格式。基座同协议0/16、1/16，但分别有15、13题截断，不可据此声称基座能力为0。三组推理预算均512token，temperature0，原问题/选项/图片不变，只调整输出指令允许同模型显式解释；没有提供正确规则或外部解题。该追加协议不混入短答案主结果，也不是独立确认；普通输出显式推理，enable_thinking仍false。

最终解释及逐题错误在`analysis_zh.md`；推理结果在`original_reasoning/summary.json`，逐题评分使用`scored_predictions.jsonl`中的`reasoning_correct`，不要把原始runner的null correct当作答错。图题有自由表述导致轨迹不可解析，条件表解析率不等于语义正确率。`manual_error_review.json`记录4条文字原题的明确条件执行错误。预期T2T高于T2V目前仅在这批开发规则上观察到，不能当作完整数据集结论。

```bash
# 已冻结的小规模诊断；不覆盖主短答案结果
.venv/bin/python /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/trace_diagnostic.py --stage prepare
.venv/bin/python /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/original_reasoning.py --stage prepare
# 对base/initial/multicandidate分别使用同一协议；完整回答按ID续跑
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/original_reasoning.py --stage evaluate --arm multicandidate --gpu 0
.venv/bin/python /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/analyze.py
.venv/bin/python /home/duyuntao/xmodal_runs/v7_multicandidate_learning_20261020/scripts/finalize_notes.py
```

上一轮两候选续训目录：`/home/duyuntao/xmodal_runs/v7_counterfactual_apply_20261020/run/`。显式两候选完整轨迹留出51/64、候选决策109/128，但原始T2T5/16、T2V4/16（起点均5/16），未见文字选择题50/192（基座48/192）。这证明部分显式检查可以学会，尚不能证明原题应用改善。该轮训练/留出按无序候选对隔离，并非单个候选完全不重复；历史说明中较强的表述以此为准。

## 2026-10-19：从学习端修复TEXT应用，保持原答题协议

### 最新小规模对照：集中训练名称与事实绑定

独立输出目录：`/home/duyuntao/xmodal_runs/v7_fact_binding_20261019/run/`；独立脚本：`/home/duyuntao/xmodal_runs/v7_fact_binding_20261019/scripts/pilot.py`。不覆盖上一轮`v7_learning_repaired_20261019`结果。

沿用8条历史开发规则，640行、3epoch、480步、LoRA32/alpha64、LR3e-5、microbatch2×accum2、seed42、冻结视觉和对齐层。与上一轮`suffix_control`保持相同步数、初始化及样本等权CE（含结束符）；token预算不同。每条规则32条完整定义、24条原子事实、8条边界/逻辑说明、16条单候选判断；移除全部选择题字母答案和附正确材料的解题回放。所有训练输入只使用文字。

```bash
.venv/bin/python /home/duyuntao/xmodal_runs/v7_fact_binding_20261019/scripts/pilot.py --stage prepare
# run：单batch训练和重载检查、正式训练、同一最终checkpoint闭卷评测
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_fact_binding_20261019/scripts/pilot.py --stage run --gpu 0
.venv/bin/python -u /home/duyuntao/xmodal_runs/v7_fact_binding_20261019/scripts/pilot.py --stage evaluate --arm base --gpu 1
.venv/bin/python /home/duyuntao/xmodal_runs/v7_fact_binding_20261019/scripts/score.py
.venv/bin/python /home/duyuntao/xmodal_runs/v7_fact_binding_20261019/scripts/analyze.py
```

每组728个探针：保留上一轮680个输入，并加48条新事实问法。上一轮已见的256道MCQ本轮**全部未训练**，标为`legacy_mcq_untrained`，不得再当作本轮训练拟合率。原始T2T/T2V各16题、新文字应用192题；其余为局部判断、事实回忆和单列给材料控制。正式评测不改原题、原选项和PNG，不提供材料、不回忆路由、不借助其他模型。固定epoch3最终第480步，T2T高于T2V是待检验假设，不能作为选模型或筛规则条件。来源学习未通过开发筛查时不扩大到1000条。

结果看`report_zh.md`、`summary.json`、`paired_changes.json`；学习数据看`training.jsonl`和`TRAINING_DATA_ZH.md`；原始回答保存在各组`evaluation/predictions.jsonl`。`completion_audit.json`生成后才表示完整训练、两组评测和最终核验完成。

本轮已完成：完整规则逐字回忆8/8，新事实字段47/48；具体名称参数15/16。未见文字选择题47/192（基座48/192），原始T2T/T2V均5/16（基座均4/16）；来源学习筛查未通过，不扩大到1000条。完整回忆可成功但换问法提取或应用仍失败，不能把复述成功解释为已学会应用。汇总使用独立`score.py`，修正了初稿将预测句末标点单侧移除造成的回忆误判；训练、推理及原始回答不变，记录见`scoring_correction.json`。

目录：`runs/v7_learning_first_20261019/`。先检验TEXT学习后T2T是否真正提升，再测同一检查点的原图T2V；同时比较相对基座的增益。T2T高于T2V是待检验假设，不以压低图片分数选择方案。

沿用已审核的8条历史开发规则（形状颜色OR、数值区间各4条），另外8条不同规则用于回放。三组从干净Qwen3-VL-4B基座重新开始，TEXT-only、各1024行/1epoch/256步、LoRA rank32/alpha64/dropout0.05、LR3e-5、microbatch2×accum2、seed42、max8192、冻结vision/aligner，统一使用样本等权CE。行数和步数匹配，不宣称token预算相同。

- `explain`：768行目标规则解释＋256行独立规则事实复述。
- `semantic`：128行目标解释＋320行新文字选择题＋320行同场景具体条件检查及结果监督＋256行独立规则事实复述。
- `semantic_replay`：保留`semantic`的全部768行目标知识训练，另256行改为给独立规则文字材料的解题回放。

目标应用输入不附正确规则；具体条件与结果作为辅助训练输出，正式测试仍只输出字母。训练文件在读取正式测试标签前生成；全部新场景与正式测试场景核对不重合，训练/生成holdout不重合，两个独立求值方式核对结果。原题、原选项、原PNG保持不变。每模型392个探针：正式T2T/T2V各16题、新文字holdout192题、已见文字练习64题、独立解题控制64题、单列给材料控制32题、单列回忆8题。回忆及给材料控制不进入任何主评测输入。

```bash
.venv/bin/python scripts/pilot_v7_learning_first.py --stage prepare
# 每组run含单batch训练、checkpoint重载文字/原图检查、正式训练、直接答题评测
.venv/bin/python -u scripts/pilot_v7_learning_first.py --stage run --arm explain --gpu 0
.venv/bin/python -u scripts/pilot_v7_learning_first.py --stage run --arm semantic --gpu 1
.venv/bin/python -u scripts/pilot_v7_learning_first.py --stage run --arm semantic_replay --gpu 1
.venv/bin/python -u scripts/pilot_v7_learning_first.py --stage evaluate --arm base --gpu 0
.venv/bin/python scripts/pilot_v7_learning_first.py --stage summarize
.venv/bin/python scripts/pilot_v7_learning_first.py --stage audit
```

同一GPU的命令顺序执行，通过共享显存ledger准入。固定最终第256步，不按评测挑检查点。已保存原答支持按ID恢复。正式每模态仅16题，生成holdout不能替代原题成绩；这批规则已参与历史调试，只是训练方案开发。来源应用通过筛查后才进行不同规则确认，不直接扩大到1000条或宣称四方向修复。

## 2026-10-19：闭卷记忆适配器与冻结基座解题开发实验

目录：`runs/v7_memory_router_20261019/`。沿用上一轮固定18条开发规则和原题、原图，从干净Qwen3-VL-4B基座训练3个独立LoRA：`text_memory`（名称→具体规则）、`vision_memory`（原学习图+名称→具体规则）、`vision_bridge`（288行带原图+288行无图文字事实回放）。每组576行、每规则32次呈现，在扩充后的训练文件上1epoch/144步；rank32、alpha64、LR1e-4、batch4、视觉与对齐层冻结。不是原1000规则的完整实验，也不是单次呈现材料。

视觉两组的目标使用原文字材料中的具体事实，区别于旧版泛化配文；属于有文字事实目标的多模态SFT。`vision_bridge`另含名称到事实的无图监督，必须单列桥接，不算纯视觉学习。没有应用题或选项进入训练。

```bash
.venv/bin/python scripts/explore_v7_memory_router.py --stage prepare
# 三组分别执行smoke、train、memory；示例：
.venv/bin/python scripts/explore_v7_memory_router.py --stage smoke --arm text_memory --gpu 0
.venv/bin/python scripts/explore_v7_memory_router.py --stage train --arm text_memory --gpu 0
.venv/bin/python scripts/explore_v7_memory_router.py --stage memory --arm text_memory --gpu 0
# 已有模型也生成相同名称回忆和原题短答案：--arm base/text_ep10/vision_ep10
.venv/bin/python scripts/explore_v7_memory_router.py --stage solve --arms text_memory --gpu 0
# 所有记忆组就绪后，可统一求解；oracle是单列给正确材料控制
.venv/bin/python scripts/explore_v7_memory_router.py --stage solve --gpu 0
.venv/bin/python scripts/explore_v7_memory_router.py --stage summarize
.venv/bin/python scripts/explore_v7_memory_router.py --stage audit
```

`memory`保存18次仅名称回忆、18次带原学习图读取、72次闭卷原题短答案。`solve`只取**仅名称回忆**的原样输出，交给冻结原基座：隐藏选项执行，再从未经修正工作记录选择原选项。带学习图回忆仅用于诊断，绝不进入闭卷路由；失败回忆不纠正、不筛除。每组路由144次生成，文字/图片最终各36题。基座自身回忆使用完全相同路由作为方法匹配对照；正确材料oracle单列。所有文件支持按ID恢复。`summary.json`、`report_zh.md`、`completion_audit.json`及各组原始`predictions.jsonl`保留全部结果；任何局部提升都需在新规则上确认后才能扩大。

另加预先固定的图片观察转录对照：不提供规则、选项或正确文字观察，仅由冻结基座读取36张原测试图，生成未经修正的事实转录；再把同一份转录分别与基座、三个新记忆组的名称回忆及单列oracle组合。所有5种来源都运行，不按成功案例筛选。最终题目仍选原选项，转录错误也保留。它增加推理步骤，不能混入原单步主分数。

```bash
.venv/bin/python scripts/explore_v7_observation_router.py --stage prepare
# 全部记忆组就绪后执行
.venv/bin/python scripts/explore_v7_observation_router.py --stage run --gpu 0
.venv/bin/python scripts/summarize_v7_memory_router.py
```

最终`all_methods.csv/json`汇总全部流程，`paired_changes.json`保存同题转对/转错与描述性的规则级bootstrap区间；`training_loss.png`只画本轮正式训练曲线。`target_reconstruction.json`是归一化后的逐字重构统计，不是所有自由回忆的人工语义评分。基座与历史10epoch模型权重身份、所有输入传递与原PNG哈希在最终汇总时复核。

追加`text_pt_memory`材料式PT对照：同样18规则、576行、32次呈现、干净基座和rank32/144步，使用同一份具体事实材料，去掉问答提示、关闭聊天模板，按全部材料token训练。独立manifest保存在`text_pt_memory/`；这是新增训练目标对照，最终四条正式训练曲线一并保留。

```bash
.venv/bin/python scripts/explore_v7_material_memory.py --stage prepare
.venv/bin/python scripts/explore_v7_material_memory.py --stage run --gpu 1
.venv/bin/python scripts/explore_v7_material_memory.py --stage audit
```

新规则确认目录为`runs/v7_memory_router_confirm_20261019/`。从相同18个结构子类按固定SHA256另取18条，排除已索引小实验knowledge文件中的规则；配置在`confirmation_manifest.json`，所有36道文字标签在生成前独立重算通过。全新训练TEXT及视觉文字桥接适配器，沿用相同配方和全部原图；保留纯视觉组在开发实验的负结果，确认不把桥接冒充纯视觉。确认模型从原基座重新开始，不继承完整1000条的历史训练模型。

```bash
.venv/bin/python scripts/confirm_v7_memory_router.py --stage prepare
.venv/bin/python scripts/confirm_v7_memory_router.py --stage run --gpu 0
.venv/bin/python scripts/confirm_v7_memory_router.py --stage run --gpu 1
.venv/bin/python scripts/confirm_v7_memory_router.py --stage finish
# 开发与确认均核验完成后，生成中文解释及预设门槛判定
.venv/bin/python scripts/report_v7_memory_solution.py
```

两个`run`可独立按原命令恢复；GPU0训练TEXT、评测自身与基座，GPU1训练桥接、评测自身与oracle。`finish`核验所有原答及图像哈希，生成`confirmation_results.json`、`target_reconstruction.json`、`final_inventory.json`。这是新规则ID确认，仍为相同结构模板与单seed，不能等同于完整1000条或所有模型的结论。

## 2026-10-19：规则应用流程修复开发实验

本轮先验证给正确材料时的规则应用，不继续追加原解释SFT训练。固定另18条规则，覆盖9类及18个结构子类，排除已索引小实验knowledge文件和上一轮诊断中的70条规则；历史完整1000条模型仍见过本轮材料，因此不是独立新基准。保留原3000张PNG、题面、标签及历史结果。

第一轮对照原短答案max32、只增长度max768、明确记录结果与完整选项的JSON流程max768。第二轮是开发中追加的分步流程：隐藏选项生成工作记录，再由同一模型用自身未经纠正的记录回答原选项字母。两轮均使用Qwen3-VL-4B基座与已完成的10epoch TEXT/VISION模型；每个测试模态36题，direct/diagnosis各18题。

```bash
cd /mnt/yunpan2/tanjiyang/xmodal
.venv/bin/python scripts/repair_v7_application_protocol.py --stage prepare
.venv/bin/python scripts/audit_v7_application_protocol_labels.py
# 独立任务，可使用相同命令断点恢复；GPU编号按空闲情况设置
.venv/bin/python scripts/repair_v7_application_protocol.py --stage run --arm base --gpu 0
.venv/bin/python scripts/repair_v7_application_protocol.py --stage run --arm text --gpu 1
.venv/bin/python scripts/repair_v7_application_protocol.py --stage run --arm vision --gpu 0
.venv/bin/python scripts/repair_v7_application_protocol.py --stage summarize

.venv/bin/python scripts/repair_v7_application_two_stage.py --stage prepare
.venv/bin/python scripts/repair_v7_application_two_stage.py --stage run --arm base --gpu 0
.venv/bin/python scripts/repair_v7_application_two_stage.py --stage run --arm text --gpu 1
.venv/bin/python scripts/repair_v7_application_two_stage.py --stage run --arm vision --gpu 0
.venv/bin/python scripts/repair_v7_application_two_stage.py --stage summarize
```

结果目录为`runs/v7_application_protocol_repair_20261019/`及其`two_stage/`子目录。全部条件额外给正确文字学习材料，只能作为应用能力控制，不能写成闭卷T2T/T2V/V2T/V2V迁移提升。36道文字题已经独立求解器重算；图像仅有逐图预览审核，不能称全部语义质量已独立确认。按预先保存的应用门槛通过后才考虑训练端的小规模验证；未通过不扩到1000条，不把试验结果写回原题评分。

分步输出中仍出现正确结果对应错误选项的情况，因此追加`isolated_readout/`对照：复用同一份未经纠正的工作记录，第二步仅给该记录和四个原选项，不重新给参考材料、原问题或图片。运行`repair_v7_isolated_option_readout.py --prepare`，再分别使用`--arm base/text/vision --gpu 0/1`；此步骤只新增最终选择请求，不重跑第一步。它仍依赖第一步给正确材料，不是闭卷评测。

另外仅对0771规则准备了学习配文澄清候选：明确左右各一盘、每盘两行都属于同一完整参考集合，左盘定义下边界、右盘定义上边界，不补入数字阈值。原始图像和所有数据未改；这不是全量新版本。`scripts/build_v7_visual_boundary_clarification.py`生成`data_candidate/`中的原材料/SFT候选，`scripts/test_v7_boundary_clarification.py --prepare`及`--gpu 1`运行同图原配文/澄清配文的24请求控制；只有3个语义问题，选项轮换不能当独立题。

最终核验及同题对照表：

```bash
.venv/bin/python scripts/validate_v7_application_protocol.py --complete
cat runs/v7_application_protocol_repair_20261019/comparison_zh.md
```

## 2026-10-19：完整1000条SFT，全参数，10 epoch

本次使用原始规则解释SFT：TEXT/VISION各1000条，从原始Qwen3-VL-4B-Instruct基座开始。TEXT更新全部语言参数，VISION更新整个模型（含视觉与对齐层）。batch4、梯度累积4、LR1e-5、seed42、max8192，标准助手token损失，各10epoch/630步。使用最终第10epoch检查点，随后自动运行短答案四方向评测，各2000题，并重新生成基座对照。

两组FP32优化器与完整恢复状态会占用较多主机内存，训练通过共享锁错开；基座评测、已完成模型的评测可与另一组训练重叠，分别使用GPU0/1。

实验目录：`runs/v7_fullparam_sft_ep10_20261019/`。数据选择和完整配置保存在其中的`manifest.json`，训练文件为`data/sft_text.jsonl`和`data/sft_vision.jsonl`，与原始完整实验SFT文件逐字相同。

```bash
cd /mnt/yunpan2/tanjiyang/xmodal
.venv/bin/python -u scripts/run_v7_fullparam_ep10.py --stage prepare
.venv/bin/python -u scripts/run_v7_fullparam_ep10.py --stage encode
.venv/bin/python -u scripts/run_v7_fullparam_ep10.py --stage run
```

`run`包含全参数两步训练/恢复及重载小测试、正式训练和评测；已验证的小测试及已保存的预测会复用。重复启动同一目录会取得运行锁，不会并发覆盖同一实验。训练进程中断后使用相同命令继续。

完整DeepSpeed优化器恢复检查点暂存于`manifest.json`记录的`/dev/shm/`专属目录，主机重启后不可用；磁盘每epoch验证保存最近两份完整模型权重和训练状态。共享内存状态丢失时，程序拒绝静默重置优化器继续训练。最终模型核验后清理本次共享内存恢复状态，保留磁盘权重及全部日志。

查看进度和结果：

```bash
tail -n 20 runs/v7_fullparam_sft_ep10_20261019/jobs/text/train.log
tail -n 20 runs/v7_fullparam_sft_ep10_20261019/jobs/vision/train.log
cat runs/v7_fullparam_sft_ep10_20261019/report_zh.md
```

最终对照为`comparison.csv`、`summary.json`、`report_zh.md`和`loss_and_accuracy.png`；每个方向另有原始回答、严格评分与错误明细。下文的5epoch三模型SFT/PT记录属于历史LoRA实验。

## 1. 当前实验与训练数据

- 已完成的完整实验：3 个模型 × SFT/PT × TEXT/VISION，共 12 个 LoRA 适配器，实际均为 **5 epoch**，最终 `checkpoint-315`。
- 后续修复小实验主要使用 **1 epoch**。下面的单任务新训练示例默认 1 epoch；需要复现完整实验预算时，将 `V7_EPOCHS` 改为 5。不要修改历史运行目录的 epoch 或把两套结果混在一起。
- FT 在本项目中指原始材料 **PT**，启动命令为 `swift pt`；SFT 和 PT 都是 LoRA，不是全参数更新。
- 当前完整评测只运行短答案 `strict`，COT 暂缓。48 个评测单元、每单元 2000 题，共 96000 次回答，已全部完成。
- 保留原有 3000 张 PNG；训练和评测使用同一组 1000 条规则，但学习图和测试图分别保存。

实际使用的训练文件均位于 `runs/v7_full_experiment/data/`，每份 1000 行：

| 文件 | 输入与目标 | 聊天模板 | loss |
|---|---|---|---|
| `sft_text.jsonl` | user 请求解释规则，assistant 为完整文字材料 | 开启 | assistant 回复 |
| `sft_vision.jsonl` | user 带原学习图并请求解释，assistant 为视觉学习配文 | 开启 | assistant 回复 |
| `pt_text.jsonl` | 原始文字学习材料 | 关闭 | 材料文字 |
| `pt_vision.jsonl` | 原学习图＋原始视觉配文 | 关闭 | 材料文字，图像作为条件 |

SFT 训练提示词保存在数据的 `messages` 中，例如：

```text
Explain the rule or codebook named Nornat, including its conditions, mappings, and how to apply it.
```

VISION 加上 `<image>Using this learning reference image, ...`。完整实验的 SFT 是每条规则一组解释问答；条件回忆、代码映射、应用选择题等增强练习仅用于后续小实验，尚未用于全量 1000 规则重训。

转换脚本为 `scripts/build_v7_experiment_data.py`：SFT 从 `shared/train_text.jsonl` / `shared/train_multimodal.jsonl` 加固定提问，保留原材料作为答案；PT 逐字复制 v7 的 `T2T/train.jsonl` / `V2V/train.jsonl`。正式评测答案不用于生成训练目标。来源与文件 SHA256 在 `runs/v7_full_experiment/data/manifest.json`。

## 2. 单任务新训练：先设置环境和公共参数

以下 Bash 设置只定义环境和参数；之后从第 3 节选择需要的训练命令。新输出目录带时间戳，与历史完整实验分开保存。

```bash
cd /mnt/yunpan2/tanjiyang/xmodal

export PATH="$PWD/.venv/bin:$PATH"
export ROOT_IMAGE_DIR="$PWD/rules_repaired_single_image_v7"
export MAX_PIXELS=1003520
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MODELSCOPE_CACHE="$PWD/model_cache/modelscope"
export HF_HOME="$PWD/model_cache/huggingface"
export HF_DATASETS_CACHE="$PWD/model_cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$PWD/model_cache/huggingface/hub"
export HF_HUB_OFFLINE=1

V7_MODEL_KEY=qwen3   # 可选：qwen3、qwen25、glm46
V7_EPOCHS=1          # 新训练示例默认 1；历史完整实验为 5
V7_DATA_DIR="$PWD/runs/v7_full_experiment/data"
V7_OUT_ROOT="$PWD/runs/v7_manual_$(date +%Y%m%d_%H%M%S)"
V7_MODEL_ARGS=()

case "$V7_MODEL_KEY" in
  qwen3)
    V7_MODEL_PATH="$PWD/model_cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
    ;;
  qwen25)
    V7_MODEL_PATH=/mnt/yunpan2/huggingface_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
    ;;
  glm46)
    V7_MODEL_PATH="$PWD/model_cache/modelscope/models/ZhipuAI--GLM-4.6V-Flash/snapshots/master"
    V7_MODEL_ARGS=(--template glm4_5v)
    ;;
  *)
    printf '未知模型：%s\n' "$V7_MODEL_KEY" >&2
    exit 1
    ;;
esac

V7_COMMON_ARGS=(
  --model "$V7_MODEL_PATH"
  --tuner_type lora --target_modules all-linear
  --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05
  --torch_dtype bfloat16 --freeze_vit true --freeze_aligner true
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16
  --learning_rate 1e-4 --num_train_epochs "$V7_EPOCHS"
  --gradient_checkpointing true --max_length 8192 --warmup_ratio 0.05
  --save_strategy epoch --save_total_limit 2 --split_dataset_ratio 0
  --logging_steps 5 --seed 42 --data_seed 42 --add_version false
  --strict true --enable_thinking false
)
```

三个模型的本地权重路径与完整实验 manifest 一致。以上公共参数对应完整实验配置，只有示例的 `V7_EPOCHS` 默认改为 1；这些命令仍读取全量 1000 条材料，不是 8/12 规则子集。后续某些修复小实验的 batch、梯度累积和学习率另有设置，以各实验 manifest 为准。

## 3. 显式 SFT / PT 启动命令

在已执行第 2 节设置的同一 Bash 终端中，按需选择一条。四条命令各启动一次训练；不加 `&` 时顺序执行。不同任务目录独立，不重复使用已经写入 checkpoint 的输出目录。

### SFT TEXT：文字规则解释问答

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/swift sft \
  "${V7_COMMON_ARGS[@]}" "${V7_MODEL_ARGS[@]}" \
  --dataset "$V7_DATA_DIR/sft_text.jsonl" \
  --use_chat_template true --loss_scale default \
  --output_dir "$V7_OUT_ROOT/${V7_MODEL_KEY}_sft_text/checkpoint"
```

### SFT VISION：原学习图＋规则解释问答

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/swift sft \
  "${V7_COMMON_ARGS[@]}" "${V7_MODEL_ARGS[@]}" \
  --dataset "$V7_DATA_DIR/sft_vision.jsonl" \
  --use_chat_template true --loss_scale default \
  --output_dir "$V7_OUT_ROOT/${V7_MODEL_KEY}_sft_vision/checkpoint"
```

### PT TEXT（FT）：原始文字材料学习

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/swift pt \
  "${V7_COMMON_ARGS[@]}" "${V7_MODEL_ARGS[@]}" \
  --dataset "$V7_DATA_DIR/pt_text.jsonl" \
  --use_chat_template false --loss_scale all \
  --output_dir "$V7_OUT_ROOT/${V7_MODEL_KEY}_pt_text/checkpoint"
```

### PT VISION（FT）：原学习图＋原始视觉配文学习

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/swift pt \
  "${V7_COMMON_ARGS[@]}" "${V7_MODEL_ARGS[@]}" \
  --dataset "$V7_DATA_DIR/pt_vision.jsonl" \
  --use_chat_template false --loss_scale all \
  --output_dir "$V7_OUT_ROOT/${V7_MODEL_KEY}_pt_vision/checkpoint"
```

这四条是实际 Python 命令构造逻辑的 Bash 展开版，已包含公共参数；没有使用 `...` 占位。它们只执行单次训练，不自动创建完整实验协调器的任务状态、checkpoint manifest 或启动评测。

## 4. 在 Python 脚本中核对命令

`scripts/run_v7_full_experiment.py` 的 `train_command()` 动态拼接参数，不会包含一整行 `swift sft ...` 的 Bash 文本：

```python
# obj 决定执行 swift sft 还是 swift pt
cmd = [str(ROOT/'.venv/bin/swift'), obj, ...]

# 根据目标切换模板和 loss，位于 train_command() 内
'--use_chat_template', 'true' if obj == 'sft' else 'false',
'--loss_scale', 'default' if obj == 'sft' else 'all',

# run_job() 选择相应训练文件
cmd = train_command(
    model, obj, work/'data'/f'{obj}_{mod}.jsonl',
    out, epochs, resume=resume
)
```

上面的 Python 是定位用摘录，不是独立可执行程序。完整函数目前在第 46 行附近，模板/loss 在第 54 行附近，调用处在第 176 行附近。

运行后实际保存的参数以 checkpoint 中的 `args.json` 为准，例如：

- SFT：`runs/v7_full_experiment/jobs/qwen3_sft_text/checkpoint/checkpoint-315/args.json`。
- PT：`runs/v7_full_experiment/jobs/qwen3_pt_text/checkpoint/checkpoint-315/args.json`。

## 5. 已有完整实验的恢复、评测和汇总

以下命令针对历史 **5 epoch** 的 `runs/v7_full_experiment`。当前训练和 strict 评测均已完成，查看结果不需要再次启动。若有任务中断，可使用同一路径和原配置恢复。

恢复训练协调器（复用已有最终 checkpoint，不设置为 1 epoch）：

```bash
cd /mnt/yunpan2/tanjiyang/xmodal
.venv/bin/python scripts/run_v7_full_experiment.py \
  --root rules_repaired_single_image_v7 \
  --work runs/v7_full_experiment \
  --epochs 5 --jobs-per-gpu 3
```

恢复 strict 评测（每个适配器四方向，已保存的回答会复用）：

```bash
.venv/bin/python scripts/run_v7_evaluations.py \
  --root rules_repaired_single_image_v7 \
  --work runs/v7_full_experiment \
  --protocols strict
```

仅重新汇总已有评测，不加载模型训练或推理：

```bash
.venv/bin/python scripts/summarize_v7_experiment.py \
  --work runs/v7_full_experiment
```

这些协调器没有通用的 `--resume` 参数；内部根据任务状态、checkpoint 和落盘回答续跑。`run_v7_adapter_eval.py` 当前校验固定的第 5 epoch checkpoint，并读取协调器生成的任务 manifest，因此**不能直接用上述协调器评测第 3 节的手工 1 epoch 输出**。后续 1 epoch 修复试验使用各自的专用脚本、manifest 和评测；不要伪造历史任务状态来绕过校验。

COT 尚未补测；只有需要单独补充时才将协议设为 `--protocols strict cot`。短答案历史结果与 COT 单独报告，不能把预检查结果当完整 COT 成绩。

## 6. 查看当前结果、日志和曲线

```bash
# 完成情况和完整结果
cat runs/v7_full_experiment/final_status.json
cat runs/v7_full_experiment/summary/report_zh.md

# 以 Qwen3 SFT TEXT 为例
tail -n 30 runs/v7_full_experiment/jobs/qwen3_sft_text/train.log
tail -n 5 runs/v7_full_experiment/jobs/qwen3_sft_text/checkpoint/logging.jsonl
```

| 文件 | 位置 |
|---|---|
| 训练 loss 曲线 | `runs/v7_full_experiment/jobs/<任务>/checkpoint/images/train_loss.png` |
| 梯度范数曲线 | `runs/v7_full_experiment/jobs/<任务>/checkpoint/images/train_grad_norm.png` |
| 实际训练参数 | `runs/v7_full_experiment/jobs/<任务>/checkpoint/checkpoint-315/args.json` |
| checkpoint 选择及 SHA256 | `runs/v7_full_experiment/jobs/<任务>/checkpoint_manifest.json` |
| 原始回答 | `runs/v7_full_experiment/evaluations/<任务>/strict/<方向>/raw.jsonl` |
| 逐题评分 | `runs/v7_full_experiment/evaluations/<任务>/strict/<方向>/results.jsonl` |
| 方向报告 | `runs/v7_full_experiment/evaluations/<任务>/strict/<方向>/report.json` |
| 全部48单元汇总 | `runs/v7_full_experiment/summary/comparison.csv` / `comparison.json` |

任务名使用 `<模型>_<目标>_<模态>`，例如 `qwen3_sft_text`、`qwen25_pt_vision`、`glm46_sft_vision`；方向为 `T2T`、`T2V`、`V2T`、`V2V`。

正式总准确率把缺失、空输出、格式错误计错，有效输出准确率另报。GLM 的原正式报告存在空 `<think></think>` 标签导致的格式失败；已完成的一组补充解析见 `runs/v7_full_experiment/analysis/glm_output_format_20261014/`，不能拿补充解析覆盖原协议分数。

每个适配器都评测四个目录，其中 `T2T`/`V2T` 题面相同，`T2V`/`V2V` 题面相同；96000 次回答不是96000道独立问题。训练 loss 和 token accuracy 不能直接当作选择题准确率。

## 7. 后续小实验和整理资料

- 12规则练习增强：`runs/v7_natural_1epoch_20261014/`。
- 12规则混合语义监督：`runs/v7_semantic_1epoch_20261014/`。
- 单规则3组容量/迁移诊断：`runs/v7_rule_mastery_1epoch_20261014/`。
- 集合筛选、比例分档8规则修正重训：`runs/v7_grounded_families_1epoch_20261014/corrected_candidate/`；三组1epoch，完整报告为 `report_zh.md`。
- 1000条VISION明确答案候选：`runs/v7_grounded_sft_candidate_20261014/sft_vision_explicit.jsonl`，只是候选，尚未用于全量重训。
- 汇总说明：`runs/v7_experiment_review_20261014/results_zh.md`、`training_data_zh.md`；这些是按标注时间保存的整理快照，后续试验的最新状态以各运行目录原始报告为准。

## 历史命令：2026-10-08

以下保留旧规则包、实体包和旧 SFT 对照的命令，**不属于当前 v7 实验配置**。其中 `rules`、`rules_corrected_final`、`rules_sft` 是旧版数据/运行目录；旧文档中的 `entity/` 路径在当前工作区不存在，现有实体包为 `entity_v19_1478`，使用历史实体流程前需按实际数据与脚本另行核对，不能只替换目录名便假定结果可比较。

<details>
<summary>展开 2026-10-08 原指令</summary>

当时工作目录：`/mnt/yunpan2/tanjiyang/xmodal`。当时 Qwen 使用 `Qwen/Qwen3-VL-4B-Instruct`，下面旧流程训练为 **1 个 epoch**。


## 2. 规则类：Qwen 基座评测

不训练，评测全部四方向：

```bash
MODEL=Qwen/Qwen3-VL-4B-Instruct \
OUTPUT_DIR=/mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_base_full_run2 \
bash /mnt/yunpan2/tanjiyang/xmodal/rules/scripts/evaluate_rules.sh base
```

## 3. 规则类：重新训练 1 个 epoch

终端 A（先激活环境），训练文字模型：

```bash
EPOCHS=1 MODEL=Qwen/Qwen3-VL-4B-Instruct \
bash /mnt/yunpan2/tanjiyang/xmodal/rules/scripts/train_rules.sh TEXT 0 lora8_ep1_run2
```

终端 B（先激活环境），训练多模态模型：

```bash
EPOCHS=1 MODEL=Qwen/Qwen3-VL-4B-Instruct \
bash /mnt/yunpan2/tanjiyang/xmodal/rules/scripts/train_rules.sh VISION 1 lora8_ep1_run2
```

输出分别为 `rules/runs/experiments/lora8_ep1_run2-TEXT/` 和 `rules/runs/experiments/lora8_ep1_run2-VISION/`。两次训练完成后：

```bash
MODEL=Qwen/Qwen3-VL-4B-Instruct METHOD=lora8_ep1_run2 \
OUTPUT_DIR=/mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_lora_ep1_full_run2 \
bash /mnt/yunpan2/tanjiyang/xmodal/rules/scripts/evaluate_rules.sh lora
```

`METHOD` 必须与训练实验名一致，评测脚本自动选择最新 checkpoint。规则训练脚本拒绝已有输出目录；重新训练请换实验名。

以前的 `train_rules_qwen.py --epochs 1` 固定写入 `lora8_ep1-TEXT/VISION`，这两个目录已有结果。重新实验使用上面的可命名脚本。`--reuse-queued` 仅用于尚未开始训练的排队目录，不是训练断点恢复。

## 4. 规则类：直接评测已经训练好的模型

已有 1 epoch 模型为两个 `checkpoint-63`。无需再训练，以下命令重新推理并保存到新目录：

```bash
python /mnt/yunpan2/tanjiyang/xmodal/run_evaluation.py \
  --data-root /mnt/yunpan2/tanjiyang/xmodal/rules \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --text-adapter /mnt/yunpan2/tanjiyang/xmodal/rules/runs/experiments/lora8_ep1-TEXT/checkpoint-63 \
  --visual-adapter /mnt/yunpan2/tanjiyang/xmodal/rules/runs/experiments/lora8_ep1-VISION/checkpoint-63 \
  --directions T2T T2V V2T V2V \
  --gpu 0 --visual-gpu 1 --parallel \
  --output-dir /mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_lora_ep1_repeat_run2
```

这一节与第 3 节最后的命令评测不同的训练产物，按需要选择。

## 5. 实体类：Qwen 基座评测

```bash
python /mnt/yunpan2/tanjiyang/xmodal/run_evaluation.py \
  --data-root /mnt/yunpan2/tanjiyang/xmodal/entity \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --base-model-only \
  --directions T2T T2V V2T V2V \
  --gpu 0 --visual-gpu 1 --parallel \
  --output-dir /mnt/yunpan2/tanjiyang/xmodal/entity/runs/evaluation/qwen3vl4b_base_full_run2
```

## 6. 实体类：重新训练 1 个 epoch

终端 A：

```bash
EPOCHS=1 MODEL=Qwen/Qwen3-VL-4B-Instruct \
OUTPUT_ROOT=/mnt/yunpan2/tanjiyang/xmodal/entity/runs/repeat_ep1_run2 \
bash /mnt/yunpan2/tanjiyang/xmodal/train_corrected_one.sh TEXT 0
```

终端 B：

```bash
EPOCHS=1 MODEL=Qwen/Qwen3-VL-4B-Instruct \
OUTPUT_ROOT=/mnt/yunpan2/tanjiyang/xmodal/entity/runs/repeat_ep1_run2 \
bash /mnt/yunpan2/tanjiyang/xmodal/train_corrected_one.sh VISION 1
```

两个模型完成后：

```bash
python /mnt/yunpan2/tanjiyang/xmodal/run_evaluation.py \
  --data-root /mnt/yunpan2/tanjiyang/xmodal/entity \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --text-adapter /mnt/yunpan2/tanjiyang/xmodal/entity/runs/repeat_ep1_run2/xmodal-qwen3vl4b-corrected-v3-TEXT \
  --visual-adapter /mnt/yunpan2/tanjiyang/xmodal/entity/runs/repeat_ep1_run2/xmodal-qwen3vl4b-corrected-v3-VISION \
  --directions T2T T2V V2T V2V \
  --gpu 0 --visual-gpu 1 --parallel \
  --output-dir /mnt/yunpan2/tanjiyang/xmodal/entity/runs/evaluation/qwen3vl4b_lora_ep1_full_run2
```

评测支持传训练目录，会自动寻找最新 checkpoint。实体训练默认长度上限 4096，规则训练为 8192。实体训练目录已有 checkpoint 时拒绝重新训练，请更换 `OUTPUT_ROOT`。

## 7. 实体类：直接评测已经训练好的模型

已有模型为两个 `checkpoint-96`：

```bash
python /mnt/yunpan2/tanjiyang/xmodal/run_evaluation.py \
  --data-root /mnt/yunpan2/tanjiyang/xmodal/entity \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --text-adapter /mnt/yunpan2/tanjiyang/xmodal/entity/runs/xmodal-qwen3vl4b-corrected-v3-TEXT/checkpoint-96 \
  --visual-adapter /mnt/yunpan2/tanjiyang/xmodal/entity/runs/xmodal-qwen3vl4b-corrected-v3-VISION/checkpoint-96 \
  --directions T2T T2V V2T V2V \
  --gpu 0 --visual-gpu 1 --parallel \
  --output-dir /mnt/yunpan2/tanjiyang/xmodal/entity/runs/evaluation/qwen3vl4b_lora_ep1_repeat_run2
```

## 8. GLM-4.6V-Flash：基座评测

实体类重新评测：

```bash
python /mnt/yunpan2/tanjiyang/xmodal/run_evaluation.py \
  --data-root /mnt/yunpan2/tanjiyang/xmodal/entity \
  --model ZhipuAI/GLM-4.6V-Flash \
  --base-model-only --answer-format glm-box \
  --directions T2T T2V V2T V2V \
  --gpu 0 --visual-gpu 1 --parallel \
  --output-dir /mnt/yunpan2/tanjiyang/xmodal/entity/runs/evaluation/glm46v_flash_base_full_run2
```

规则类评测：

```bash
MODEL=ZhipuAI/GLM-4.6V-Flash ANSWER_FORMAT=glm-box \
OUTPUT_DIR=/mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/glm46v_flash_base_full_run2 \
bash /mnt/yunpan2/tanjiyang/xmodal/rules/scripts/evaluate_rules.sh base
```

GLM 使用 `glm-box` 答案解析；Qwen 默认使用 `strict`。已有完整 GLM 实体结果在 `entity/runs/evaluation/glm46v_flash_base_full_v2/`，续跑已有实验应保留原路径和原参数。

## 9. 评测预检查与断点续跑

所有 Python 评测命令均可在末尾添加：

- `--dry-run`：检查数据、图片、答案 ID 和 checkpoint，保存执行计划，不加载模型推理。
- `--resume`：沿用同一输出目录中已保存的预测，继续未完成样本。

`--dry-run` 会创建 manifest。检查通过后，移除 `--dry-run`，添加 `--resume`，其余参数和输出目录保持一致，才能开始推理。规则包装脚本同样支持在最后添加这两个参数。

例如续跑第 2 节的规则基座评测：

```bash
MODEL=Qwen/Qwen3-VL-4B-Instruct \
OUTPUT_DIR=/mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_base_full_run2 \
bash /mnt/yunpan2/tanjiyang/xmodal/rules/scripts/evaluate_rules.sh base --resume
```

续跑必须保持数据、模型、checkpoint、方向列表、生成参数和输出目录一致。不要为续跑只保留未完成的方向，原命令的四方向列表也要保留。

**已保存的错误答案和无效答案也会复用。** 已完成的实验添加 `--resume` 只复用结果并重新汇总，不会重新生成答案。想重新推理必须使用新输出目录。这里的 `--resume` 是评测续跑，不是训练恢复。

## 10. 查看日志和结果

规则新训练日志（实体训练日志同样位于模型目录的 `logs/train.log`）：

```bash
tail -f /mnt/yunpan2/tanjiyang/xmodal/rules/runs/experiments/lora8_ep1_run2-TEXT/logs/train.log
```

查看规则新实验准确率：

```bash
cat /mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_base_full_run2/summary.md
cat /mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_lora_ep1_full_run2/summary.md
```

每个评测目录包含：

| 文件 | 用途 |
|---|---|
| `summary.md` / `summary.csv` / `summary.json` | 四方向状态和准确率 |
| `T2T/report.json`（其他方向同理） | 单方向详细统计 |
| `T2T/results.jsonl` | 标准答案、模型答案和逐题评分 |
| `T2T/predictions.jsonl` | 保存的预测 |
| `T2T/errors.jsonl` | 错题明细 |

先确认四方向均为 `complete`，样本数齐全：规则每方向 2000，实体每方向 3058。`Accuracy (all)` 将空回答、缺失回答和无效格式计为错误；实验对比保持相同评分口径。

### 已有结果目录（截至 2026-10-08）

这些目录已经有结果，可直接查看，无需重新推理：

| 实验 | 相对于工作目录的结果路径 |
|---|---|
| 规则 Qwen 基座 | `rules/runs/evaluation/qwen3vl4b_base_full/` |
| 规则 Qwen 1 epoch LoRA | `rules/runs/evaluation/qwen3vl4b_lora_ep1_full/` |
| 实体 Qwen 基座 | `entity/runs/evaluation/qwen3vl4b_base_full/` |
| 实体 Qwen LoRA | `entity/runs/evaluation/qwen3vl4b_full/` |
| 实体 GLM 基座 | `entity/runs/evaluation/glm46v_flash_base_full_v2/` |

```bash
cat /mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_base_full/summary.md
cat /mnt/yunpan2/tanjiyang/xmodal/rules/runs/evaluation/qwen3vl4b_lora_ep1_full/summary.md
```

## 规则类：SFT 对照（1 epoch）

本次使用独立的 user/assistant 问答数据训练，TEXT、VISION 各 1000 条。仅助手回复计算 loss；LoRA 配置保持 rank 8、alpha 32、dropout 0.05。任务是解释规则，回答为原学习正文，不包含正式评测题或答案。数据、配置和结果说明见 `rules_sft/README.md`。

重新跑一组（两张卡空闲时执行，输出目录必须不存在）：

```bash
cd /mnt/yunpan2/tanjiyang/xmodal
.venv/bin/python -u scripts/run_rules_sft.py \
  --run-dir rules_sft/runs/lora8_ep1_repeat
```

脚本先并行训练 TEXT（GPU 0）和 VISION（GPU 1），各 1 epoch，再在 `rules_corrected_final` 上依次评测 SFT、已有 PT、基座的四方向。三组推理统一 batch size 4。模型和数据缓存均指定为工作目录内的 `model_cache/`。

本次实验的评测续跑（仅复用已经完整训练成功的 checkpoint）：

```bash
cd /mnt/yunpan2/tanjiyang/xmodal
.venv/bin/python -u scripts/run_rules_sft.py \
  --run-dir rules_sft/runs/lora8_ep1_20261008 --resume
```

查看训练或评测进度：

```bash
tail -n 30 /mnt/yunpan2/tanjiyang/xmodal/rules_sft/logs/launcher_20261008.log
tail -n 20 /mnt/yunpan2/tanjiyang/xmodal/rules_sft/runs/lora8_ep1_20261008/TEXT/logs/train.log
tail -n 20 /mnt/yunpan2/tanjiyang/xmodal/rules_sft/runs/lora8_ep1_20261008/VISION/logs/train.log
```

默认会完整执行训练与评测；仅训练可加 `--train-only`。脚本拒绝覆盖训练目录，`--resume` 不用于中途失败训练的断点恢复。各组结果保存在 `rules_sft/runs/lora8_ep1_20261008/evaluation/`。

本组已完成结果：[SFT/PT/基座对照报告](rules_sft/runs/lora8_ep1_20261008/comparison.md)。SFT 四方向平均 30.81%，PT 29.93%，基座 30.35%，均在修正版评测集上测得。

</details>

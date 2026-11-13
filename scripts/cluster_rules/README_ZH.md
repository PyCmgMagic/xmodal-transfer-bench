# 1000 条规则：KIT/Horeka 集群迁移包

此包准备原始组与 10 条/规则增强组，沿用 Qwen3-VL-4B-Instruct、ms-swift 和现有全参数 SFT 配置。只包含数据、图片和运行工具；不含模型权重、虚拟环境、登录凭据。未提交集群任务。

## 数据与对照

| 组别 | 每条规则的样本 | TEXT 条数 | VISION 条数 | 默认 10 epoch 的每规则呈现次数 |
|---|---|---:|---:|---:|
| original | 1 条原始问答 | 1,000 | 1,000 | 10 |
| repeat10 | 原始问答重复 10 次 | 10,000 | 10,000 | 100 |
| multiform10 | 10 种完整定义问法，原答案不变 | 10,000 | 10,000 | 100 |

三个组、两种学习模态，共六份训练文件，总计 42,000 行。增强组保留原始问法作为其中一种问法。repeat10 与 multiform10 的规则、行数、目标答案、图片和记录顺序配对一致；只有用户问法不同。此处 10 条表示 10 条训练记录，并不表示新增 10 个事实。原始组的消息与此前 1000 条 SFT 输入逐条相同；仅添加 ID 并迁移图片路径。

**按用户选择保持相同 epoch，不控制相同计算预算。** 扩展组样本呈现量为原始组的 10 倍，训练耗时通常接近 10 倍，受序列长度、图片编码和检查点保存影响。原始组与扩展组的差异同时包含更多学习次数；repeat10 与 multiform10 的比较帮助检查问法变化的作用。无需默认运行所有组：每组每模态独立提交。

本次是完整定义问法增强，**没有复刻旧 64 规则实验的 48 条完整定义＋24 条原子事实＋8 条逻辑边界课程**，不能据此宣称复现旧实验。所有监督答案都来自对应模态的原始学习材料，不从测试问题或测试答案生成训练内容。VISION 保留原始视觉配文，不将 TEXT 的完整形式规则补进视觉答案；学习图像与视觉配文共同组成视觉输入，因此不是纯图像学习。1000 条规则覆盖 9 个规则家族，详见 manifest。

## 包内布局

```
train/{original,repeat10,multiform10}/{text,vision}.jsonl
eval/{T2T,T2V,V2T,V2V}.jsonl
eval/gold.jsonl                         # 单独保存，不能传入训练或推理输入
images/<qid>/{learning_visual,test_visual_1,test_visual_2}.png
audit/                                 # 学习来源、规则名称/家族、原始数据验证说明
tools/                                 # CPU 校验、迁移、全参数训练与 Slurm 模板
manifest.json, SHA256SUMS, model_identity.json, environment_reference.json
```

四个评测方向各有 2,000 道题，共享同一套 2,000 条 gold；T2T/V2T 题目配对，T2V/V2V 题目配对。第一位表示模型学习模态，第二位表示测试输入模态。TEXT 模型测 T2T/T2V，VISION 模型测 V2T/V2V。每条规则 1 张学习图与 2 张测试图，共 3,000 张 PNG，原始字节不变。评测针对已学习规则的应用，训练和评测共享规则 ID；学习图与测试图分开。推理输入中没有 assistant 答案。

图片哈希只证明字节一致，不代表全部图像语义已经人工认证。v7 的 104 道二值直接题测独立条件检查与记录代码；部分记录约定由配文提供。原数据限制见 audit/source_README.md、audit/source_validation_report.json。

## 在集群解包、校验和准备

登录流程以用户提供的 Horeka 文档为参考：VPN、SSH/OTP；文档出现 accelerated 与 accelerated-h100 两种分区名，提交前用实际账户查询。不要直接套用文档中他人的账户、项目路径或旧 VERL 容器。以下路径均由实际集群位置填写：

```bash
sha256sum -c rules1000_10x_20261027.tar.gz.sha256
tar -xzf rules1000_10x_20261027.tar.gz
export PACKAGE_ROOT="$PWD/rules1000_10x_20261027"
export WORK_ROOT="$PWD/rules1000_work"          # 新目录，放在包目录外
export MODEL_PATH=/your/project/models/Qwen3-VL-4B-Instruct
export OUTPUT_ROOT=/your/project/outputs/rules1000

PYTHONDONTWRITEBYTECODE=1 python "$PACKAGE_ROOT/tools/validate.py"
python "$PACKAGE_ROOT/tools/verify_model.py" --model "$MODEL_PATH"
python "$PACKAGE_ROOT/tools/prepare.py" --work "$WORK_ROOT"
```

大文件应放项目的合适存储区并先检查配额。模型需要单独迁移：本包 model_identity.json 记录已有模型文件哈希；verify_model.py 检查完整权重。不能用训练后的检查点代替初始化模型。prepare.py 把十份训练/评测文件写入新工作目录，并将图片路径转换为当前解包位置的绝对路径；不复制图片，也不修改包内原件。移动解包目录后，重新 prepare 到新工作目录。工具需要 Python 3.10+；纯数据校验不需要 PyTorch/GPU。SHA256SUMS 覆盖包内所有正式文件，允许 Python 生成的 __pycache__ 缓存。

## 训练环境与默认配置

environment_reference.json 记录已有环境：ms-swift 4.5.3、PyTorch 2.6.0+cu124、transformers 4.57.1、bitsandbytes 0.48.1、accelerate 1.8.1、peft 0.18.1、datasets 3.3.2。它是版本参考，不是完整环境锁文件。需在集群配置匹配的 GPU/Python 环境后先做小步检查；旧文档中的容器不保证支持 Qwen3-VL。数据包不自动安装环境或下载模型。

默认采用最新 rules-only 全参数 SFT：10 epoch，BF16，SDPA，学习率 1e-5，cosine，warmup 0.05，batch 1、梯度累积 16，seed 42，max_length 8192、max_pixels 1003520，梯度检查点。AdamW 使用 8-bit 优化器状态，模型权重不量化。TEXT 冻结视觉塔和 aligner，VISION 全部训练。保留真实可训练参数与权重更新审计插件。所有组均从同一原始模型独立初始化，不串接上一组模型。

单 GPU 原始组预计 630 个优化步，扩展组各 6,250 步（具体以 ms-swift 的实际 Trainer 状态为准）。这里按每 epoch 分别向上取整，尾 batch 使步数比值略小于 10。保存 2 个包含优化器状态的检查点，需要预留模型与优化器磁盘空间。Slurm 示例按 1 张 GPU 编写，沿用调度器的 CUDA_VISIBLE_DEVICES，不引用本机资源锁或硬编码 GPU 编号。是否能装入显存及实际时长仍需在目标 GPU 上检查。

## 预览与 Slurm 提交示例

训练工具默认仅打印命令，不执行；正式执行必须在 Slurm 分配内。先激活上述环境，确认 Python 与 swift 属于同一环境：

```bash
python "$PACKAGE_ROOT/tools/train.py" --work "$WORK_ROOT" --model "$MODEL_PATH" \
  --output "$OUTPUT_ROOT/preview_original_text" --arm original --modality text

sinfo -o "%P %a %l %D %G"
squeue -u "$USER"
```

下面 `YOUR_PARTITION` 和可选 `YOUR_ACCOUNT` 必须替换为实际值；按集群规则通过 sbatch 命令行覆盖 GPU、内存和时限参数，保留包内模板原件。以下是供用户执行的示例，本地准备工作不会提交它：

```bash
export ARM=original MODALITY=text EPOCHS=10 SEED=42
export SMOKE=1
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT "$PACKAGE_ROOT/tools/train.slurm"
# 检查 smoke 日志与 parameter_audit.jsonl 后，再提交完整训练：
export SMOKE=0
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT "$PACKAGE_ROOT/tools/train.slurm"
```

可独立设置 ARM=original/repeat10/multiform10、MODALITY=text/vision。SMOKE=1 只跑 2 个优化步；取最长的四条目标做检查，使用单独输出目录，不作为准确率结果。正式训练输出路径包含 Slurm job ID，避免覆盖。续训用 train.py 的 --resume 指定含优化器/调度器/RNG 状态的完整检查点。Slurm 示例时限为占位值，不能据此估计实际训练耗时。

本包提供评测输入与 gold，未附集群推理任务；比较时应固定同一检查点选择、解码、答案解析和先回忆再回答协议，同时报告准确率与无效回答率。训练对照本身不能证明推理协议改动的收益。

# ms-swift 本地运行环境

准备日期：2026-10-24。当前环境位于项目 `.venv`，不向 Conda base 安装依赖。所有下载、临时文件和新实验产物使用 `/mnt/vdb` 数据盘。

## 激活与检查

```bash
cd /mnt/vdb/tanjiyang/xmodal
source scripts/activate_xmodal.sh
which python
which swift
python -c 'import torch, swift; print(torch.__version__, swift.__version__, torch.cuda.is_available())'
```

运行环境为 Python 3.10、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、ms-swift 4.5.3、Transformers 4.57.1、PEFT 0.18.1。核心版本见 `requirements-runtime.txt`，全部解析后的依赖见 `requirements-runtime.lock`。

下载的基础模型为 Qwen3-VL-4B-Instruct，位于 `model_cache/Qwen3-VL-4B-Instruct/`。下载脚本记录 ModelScope 的文件修订、大小及 SHA256，逐文件校验后才生成 `download_complete.json`。

## 小样本执行检查

当前状态：**6个训练任务与8个推理任务全部通过**。规则类 SFT/PT 各有 TEXT、VISION 检查点，实体类 SFT 有 TEXT、VISION 检查点；每个检查点均实际完成两步更新并保存、重载。共生成64条回答，全部能解析为A/B/C/D，逐题输入匹配和独立评分复核通过。

训练日志记录的GPU峰值占用约8.75–9.88 GiB；此数字对应小样本、`max_length=2048`和`max_pixels=262144`的本次配置，不是完整长序列训练的显存上限。运行结束后GPU已释放。

```bash
source scripts/activate_xmodal.sh
python scripts/run_environment_smoke.py --include-pt
```

该流程固定选择每套数据的前两个知识单元，每个训练任务运行两次参数更新。规则类检查 TEXT/VISION 的 SFT 和 PT；实体类检查 TEXT/VISION 的 SFT。每个保存的 adapter 重新加载后测试文字题和原图片题，TEXT 提供 T2T/T2V，VISION 提供 V2V/V2T；同时测试未训练基座。每方向只取四题，原始数据不变，标签不会进入推理输入。

检查项包括有限 loss、实际 LoRA-B 更新、冻结视觉骨干、adapter/优化器保存、checkpoint 重载、逐题输入对应及原始回答评分。此处只验证运行链路；两次更新和四题准确率不能证明知识已学会或迁移有效。

`workflow.log` 记录阶段进度，`*_train.log` 和 `*_eval.log` 记录完整命令输出。`checkpoint_audit.json`、`summary.json/csv` 由所有步骤通过后生成。新检查点位于同一目录下的 `rules/` 和 `entities/` 子目录，不覆盖历史实验。

结果入口：

- [汇总与完整配置](runs/environment_smoke_20261024/summary.json)
- [检查点更新与loss审计](runs/environment_smoke_20261024/checkpoint_audit.json)
- [最终独立复核](runs/environment_smoke_20261024/final_verification.json)
- [逐方向小样本统计](runs/environment_smoke_20261024/summary.csv)
- [单元测试日志](runs/environment_smoke_20261024/unit_tests.log)：15项通过，包含四方向输入/标签隔离、输出解析、续跑契约、GPU资源和checkpoint存储。

首次启动遇到`persistent_workers option needs num_workers > 0`，已在验证脚本中显式设置`--dataloader_persistent_workers false`。日志保留首次失败及修正后成功记录，最终阶段状态文件均为`complete`。ms-swift 4.x的Python API使用`swift.model`、`swift.template`和`swift.infer_engine`，不应沿用旧示例的`swift.llm`导入。

本次仅下载并验证Qwen3-VL-4B-Instruct；Qwen2.5-VL-7B及GLM仍需单独准备和验证。小样本检查后，已按用户选择启动全量 PT/SFT LoRA 学习，详见下文；没有启动全参数训练。

## 全量 PT/SFT LoRA 学习

2026-10-24 已启动 `scripts/run_lora_learning.py`：规则 1,000 条、实体 1,109 条，各自包含 PT/SFT × TEXT/VISION，共 8 个独立 LoRA，每个 5 epoch。所有训练使用项目 `.venv`，从原始 Qwen3 基座独立开始，不使用评测题构建监督或选择 checkpoint。单卡最多并发两个任务，后台队列自动继续。

本次实体 SFT 协议已固定：TEXT 用实体名称提问，VISION 用学习参考图提问，答复均取对应模态的原始学习材料。完整输入、配置、来源和源码快照保存在 [运行说明](runs/qwen3_lora_learning_ep5_20261024/README_ZH.md) 所列目录中。动态状态见 [总状态](runs/qwen3_lora_learning_ep5_20261024/status.json) 和各任务 `state.json`；启动不代表训练或四方向评测已经完成。

## 全量规则输入

已用现有构建脚本将全量规则输入重新生成到 `runs/environment_smoke_20261024/prepared_rules/`，修正学习图像在新机器上的绝对路径。该目录包含各 1,000 条的 SFT TEXT/VISION 和 PT TEXT/VISION 输入及来源 manifest。

历史实验协调脚本仍可能带有源服务器模型路径、旧数据根目录或双 GPU 假设；不能直接不加参数启动。单卡正式训练应明确指定新模型路径、数据根目录、输出目录及资源预算。

## 重建环境

用可用的Python 3.10解释器创建独立环境后，分别从CUDA wheel源和PyPI安装依赖。下列命令适用于新环境；不要覆盖正在使用的环境：

```bash
python3.10 -m venv .venv
source scripts/activate_xmodal.sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.lock
python -m pip check
python scripts/download_smoke_model.py --output model_cache/Qwen3-VL-4B-Instruct
```

当前环境使用`uv`安装，二进制保留在`.runtime/uv`，也可用`.runtime/uv pip`执行等价安装。激活脚本将缓存和临时目录固定到数据盘，避免占满系统盘。按用户最终偏好保留项目`.venv`；准备过程中创建的空闲Conda环境已移除，base未安装训练依赖。

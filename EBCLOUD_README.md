# ebcloud 当前主线迁移

目标：`/root/data/xmodal`。本次快照采集于 2026-11-13，源为
`L40:/mnt/vdb/tanjiyang/xmodal`。迁移完成状态以目标 `.migration/` 中的验收记录为准。

## 资产与版本

- 代码：本私有 GitHub 仓库，保留当前脚本、测试和研究协议。仓库已有的 10 月发布记录仍保留。
- 规则：`data/rules_repaired_single_image_v7`，1000 条规则、3000 张图片，每方向 train 1000 / eval 2000。
- 实体：`data/entity1025_repaired_20261103`，1814 组、3628 张图片，每方向 train 1814 / eval 3628。
- 两套数据：`PyCmg/xmodal-transfer-data`，固定提交 `cd520242393548fd595d72f017b54bcc5f38c6c2`，逐文件对照 L40 清单中的 SHA256。
- 基座：Qwen3-VL-4B-Instruct、GLM-4.6V-Flash，从魔搭按原文件修订下载，逐文件核对源哈希。
- 8 个最终 LoRA 和当前研究记录：从 L40 保留原相对 `runs/` 结构传入，SHA256 验收后使用。

实体采用当前 1814 组修复版；根目录 `MIGRATION_README.md` 和部分旧发布文档描述的 1109 组是历史版本。
模型、数据、虚拟环境、凭据和原始实验产物不提交到 GitHub。

## 运行入口

```bash
cd /root/data/xmodal
source cluster/ebcloud/environment.sh qwen
# 或：source cluster/ebcloud/environment.sh glm
```

两个环境各自安装在 `envs/qwen`、`envs/glm`。Python 为 3.10；torch 2.6.0+cu124、
ms-swift 4.5.3、peft 0.18.1、accelerate 1.8.1。Qwen 使用 transformers 4.57.1，
GLM 使用 transformers 5.16.1。源 GLM 的 fsspec/datasets 冲突在目标使用 fsspec 2024.12.0 解决。

最终验收包括全部传入文件和公共资产的哈希、8 个 adapter 的结构检查、数据行数及图片引用、
两套运行环境的 pip check，以及 A800 上的文字/图片生成和最终 LoRA 加载。
运行验证只是部署检查，其输出不计入研究结果。

历史 JSON/JSONL 中的 L40 绝对路径保留为来源记录。可运行输入另存到 `runs/runtime_inputs/`，
并在 `.migration/runtime_training_catalog.json` 记录输入的新旧哈希、目标基座和 Python 路径。
训练或评测应显式使用该目录中的新输入及目标基座。原始 adapter 配置中的基座路径也属于历史记录，
加载时显式指定本机基座；运行所需的迁移副本另存，避免覆盖历史参数。

这次只迁移最终 adapter，未复制 optimizer、scheduler、RNG 状态，因而不支持从原优化步精确续训。
Qwen32B、Qwen3.8-27B 和旧中间检查点未纳入当前主线。

本次网络连通性：魔搭可直接下载；HF 固定提交经镜像返回 403 时，通过本机现有代理的临时 SSH
回环隧道解析下载地址，CDN 文件直接由 ebcloud 下载。账号令牌和本机私钥不复制到目标；迁移完成后关闭隧道。

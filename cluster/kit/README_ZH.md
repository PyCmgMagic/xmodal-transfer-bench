# KIT/Horeka Slurm 入口

完整流程见项目根目录 HUB_MIGRATION_ZH.md。

| 文件 | 作用 |
|---|---|
| setup_env.sbatch | 在计算节点创建独立 Conda 环境并安装依赖 |
| setup_env.sh | 安装过程；不向 base 安装依赖 |
| fetch_assets.sbatch | 在计算节点按 hub_sources.json 下载 HF 资产并保存 commit 锁 |
| rules_sft.sbatch | 规则 TEXT、VISION 全量 SFT，各 10 epoch |

在登录节点查询资源：

```bash
sinfo -o "%P %a %l %D %G"
kit_project_usage
/usr/lpp/mmfs/bin/mmlsquota -u "$(whoami)" --block-size G -C hkn.scc.kit.edu hkfs-work
```

把 CPU/GPU 分区占位符改成实际可用分区；需要项目 account 时加上相应 SBATCH 选项。
模板申请 1 GPU、64 GB 内存、8 小时，建议显存至少 40 GB。历史正式训练峰值约 31.6 GiB，不是集群上已验证的上限。

从项目根目录提交，先创建 logs/。环境安装和资产下载需要计算节点网络；没有外网时用允许联网的分区，或预下载资产和 wheelhouse。
默认下载 dataset/base 用于重训。只要已有模型时设置 XMODAL_ASSETS="dataset text vision"，不提交训练作业。

资产路径默认：

- 数据集：artifacts/dataset/
- 基座：model_cache/Qwen3-VL-4B-Instruct/
- 已训练 TEXT/VISION：artifacts/models/text/ 和 artifacts/models/vision/
- 新训练输出：runs/kit_rules_full_sft_ep10/

XMODAL_DATA_DIR、XMODAL_BASE_MODEL、XMODAL_WORK、XMODAL_CONDA_ENV、CONDA_BASE 可覆盖默认路径。
所有路径必须对计算节点可见。

新训练包括完整保存/重载冒烟测试与正式训练，连同输入和环境应预留至少约 180 GiB 空间，并查询项目实际配额。
历史正式训练 TEXT 约 58 分钟、VISION 约 3 小时 12 分钟；是否能在 8 小时内完成尚需集群实测。

训练输出被协调脚本重定向到各任务 train.log，SBATCH stdout 主要用于启动和错误排查：

```bash
squeue -u "$USER"
tail -f logs/rules-sft-JOB_ID.err
tail -f runs/kit_rules_full_sft_ep10/jobs/text/train.log
tail -f runs/kit_rules_full_sft_ep10/jobs/vision/train.log
```

tail 的 Ctrl+C 仅退出查看。只操作你自己的作业，不干预其他用户任务。

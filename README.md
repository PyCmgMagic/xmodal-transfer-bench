# GitHub / Hugging Face 迁移发布

代码与大型资产分开管理。准备目录为 `hub_release/`，不直接上传当前工作区。

| 本地发布目录 | 目标 | 用途 |
|---|---|---|
| `hub_release/github/` | GitHub 代码仓库 | 可迁移训练入口、Slurm 脚本、版本依赖、下载与核验脚本 |
| `hub_release/huggingface/dataset/` | Hugging Face Dataset 仓库 | 规则类、实体类原始数据及相对路径学习输入 |
| `hub_release/huggingface/model-text/` | Hugging Face Model 仓库 | 本轮规则 TEXT 全量 SFT 最终权重及加载配置 |
| `hub_release/huggingface/model-vision/` | Hugging Face Model 仓库 | 本轮规则 VISION 全量 SFT 最终权重及加载配置 |

不发布运行环境、凭据、旧版 checkpoint、优化器状态和原始服务器的日志。基础模型从官方 `Qwen/Qwen3-VL-4B-Instruct` 下载，不重复上传。原工作区和原始数据均保留。

准备工具将写入文件数量、数据行数、图片路径检查及模型分片检查结果。数据来源和实体数据中未完成语义认证的候选仍保留在数据卡中。HF 上传命令默认创建私有仓库。

## 1. 源服务器：准备和核验

在原项目根目录执行；不会启动训练、推理或任何远程上传：

```bash
.venv/bin/python scripts/manage_hub_release.py prepare
.venv/bin/python scripts/manage_hub_release.py verify
```

已存在的发布目录不会被覆盖。想重新生成时，增加 --output hub_release_v2。
图片与模型大文件采用硬链接，避免再复制约 21 GiB 资产；不要编辑发布目录中的图片或权重。
原始数据、历史日志和 checkpoint 不作迁移或删除。普通文本会计算 SHA256；图片和权重本次仅检查存在、大小和分片结构，未重新读取全部数据载荷。需要完整重算时：

```bash
.venv/bin/python scripts/manage_hub_release.py verify --full-hash
```

此操作会读取约 21 GiB 文件，适合磁盘不繁忙时执行。

## 2. 上传 GitHub：只上传代码目录

先在 GitHub 创建空的私有代码仓库；不要创建默认 README。下面的用户名/组织名需替换成实际目标：

```bash
cd hub_release/github
git init
git symbolic-ref HEAD refs/heads/main
git add .
git commit -m "Prepare portable cross-modal experiments"
git remote add origin git@github.com:PyCmgMagic/xmodal-transfer-bench.git
git push -u origin main
```

若准备目录已有本地 Git 仓库，跳过 git init；缺少提交身份时使用你自己的 Git 姓名和邮箱。
不要把原工作区整体 git add 后上传。环境、数据、模型和实验原始产物由 .gitignore 排除。
代码仓库保留一份当前实验摘要，未把历史分数当作新协议结果。

## 3. 上传 Hugging Face：一个数据集仓库、两个模型仓库

在原项目根目录、激活项目环境后登录 HF。登录信息存放在个人凭据缓存，不写进代码：

```bash
source .venv/bin/activate
hf auth login
python scripts/manage_hub_release.py upload --namespace PyCmg
```

会创建/上传以下私有仓库：

- PyCmg/xmodal-transfer-data（Dataset）
- PyCmg/xmodal-rules-text-sft-ep10（Model）
- PyCmg/xmodal-rules-vision-sft-ep10（Model）

上传工具使用 HF 的大目录上传 API。成功后在发布目录记录 upload_receipt.json，内含实际仓库名与 commit SHA；传输中断可用相同命令重试。
每个仓库上传可能产生多个提交，集群应使用上传完成后的 SHA，而不是上传中途的 main。

两个模型来自规则类全量 SFT 的最终 checkpoint-630，不包含 optimizer.pt、scheduler.pt、rng_state.pth 或 training_args.bin。
它们能用于推理/评测或从权重开始新训练，不能精确续接原优化器状态。
历史 LoRA/PT 权重仍保留在原机器，不包含在本次三个 HF 仓库中。

基础模型从 Qwen/Qwen3-VL-4B-Instruct 获取，不重复发布。
training_summary.json 保留原基座文件 SHA256。首次下载官方仓库的最新提交不等于已证明与历史基座逐文件一致；严格复现应固定并核对对应版本。

## 4. KIT：从 GitHub 拉代码，填写 HF 仓库和版本

以下在头节点只运行 shell/Git/Slurm 命令。环境安装、资产下载和训练均通过提交作业执行：

```bash
cd /YOUR_SHARED_WORK_DIRECTORY
git clone git@github.com:PyCmgMagic/xmodal-transfer-bench.git xmodal
cd xmodal
mkdir -p logs
cp hub_sources.example.json hub_sources.json
```

编辑 hub_sources.json：

- 将 PyCmg 换成实际 HF 用户名或组织。
- dataset/text/vision 的 revision 填入 upload_receipt.json 中相应 commit SHA。
- base 的 revision 可先留 null；首次下载解析到具体 commit 后写入 hub_assets.lock.json，重跑默认复用该 commit。严格复现应显式核验历史基座版本。

代码不包含 token。私有 HF 仓库需要你自己的读权限凭据，可在提交前用 shell 隐藏输入，给作业传环境变量：

```bash
read -rsp 'HF read token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

不要把 token 填进 hub_sources.json、SBATCH 文件或 Git，不要使用 set -x 输出环境。

## 5. KIT：提交安装、下载、训练三个作业

先用 sinfo 查分区，替换两个 CPU 作业和训练模板中的分区占位符。需要 account 时添加实际项目的 SBATCH --account。
CONDA_BASE 默认是个人目录下 miniconda3，可在提交前导出实际位置：

```bash
sinfo -o "%P %a %l %D %G"
export CONDA_BASE="$HOME/miniconda3"
SETUP_JOB=$(sbatch --parsable cluster/kit/setup_env.sbatch)
FETCH_JOB=$(sbatch --parsable --dependency=afterok:"$SETUP_JOB" cluster/kit/fetch_assets.sbatch)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:"$FETCH_JOB" cluster/kit/rules_sft.sbatch)
printf 'setup=%s fetch=%s train=%s\n' "$SETUP_JOB" "$FETCH_JOB" "$TRAIN_JOB"
squeue -u "$USER"
```

安装作业创建项目下独立 Conda 环境，不向 base 安装依赖。
下载默认获取 dataset 和 base。训练先做两步保存/重载验证，再顺序训练规则类 TEXT、VISION 各 10 轮。
新集群训练输出使用 runs/kit_rules_full_sft_ep10，不使用本机完成状态文件。

如只迁移已有权重做后续评测，改为下载三项资产，不提交训练作业：

```bash
export XMODAL_ASSETS="dataset text vision"
sbatch --dependency=afterok:"$SETUP_JOB" cluster/kit/fetch_assets.sbatch
```

模板尚未在 KIT 实际运行。计算节点如果不能访问 PyPI/HF，需使用管理员允许的联网分区，或先在允许联网的机器下载文件和 wheelhouse，再同步到共享目录。不能默认计算节点有外网。

查看日志：

```bash
tail -f logs/setup-JOB_ID.out
tail -f logs/fetch-JOB_ID.out
tail -f runs/kit_rules_full_sft_ep10/jobs/vision/train.log
```

## 验证范围与实验口径

规则类和实体类分开保存。TEXT 学习对应 T2T/T2V，VISION 学习对应 V2T/V2V；同一模态的学习材料不能按两个方向重复计算为独立样本。
原始题干、选项、标签、学习文字和图片不改写；只把准备好的学习 JSONL 图片路径转换为相对路径。
这里的视觉学习包含原始配文，不是纯像素学习。

8 份历史 PT/SFT 学习输入和原四方向数据随数据集保存；当前正式可迁移训练入口仅为规则类 full SFT 10 epoch。
保留的 run_v7_evaluation.py 是严格短答案评测器，answer_readout.py 是答案解析库；二者不自动构成此前的 CoT/回忆实验，不自动恢复暂停任务。
历史摘要在 reports/20261027，摘要中的原 runs/ 路径仅作来源索引，原始日志不在代码仓库中。

不新增项目或数据开源许可证，仍保留原始出处和原有条款。本次上传工具采用私有仓库。

## 账号与当前连接情况

代码仓库建议名为 xmodal-transfer-bench。提供 GitHub 用户名或组织名、HF 用户名或个人主页即可确定目标；无需提供密码或在聊天中粘贴 token。
登录与授权在自己的终端里进行。上传前尚需指定目标用户名/组织，并确保当前终端拥有 GitHub/HF 写权限。

本次源机器访问 HF 官方域名时，Python 的标准库和 requests 均报告证书域名不匹配；尚未完成在线上传验证。
不关闭 TLS 校验，不向未知镜像发送 token。应先检查这台机器的代理、DNS 和证书配置，或在网络正常的可信机器上上传整理后的目录。

登录遇到 Network is unreachable 时，参见 HF_CONNECTIVITY_ZH.md：可用自己的电脑经 SSH 转发，为单次 HF 登录/上传提供代理。

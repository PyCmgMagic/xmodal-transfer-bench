# xmodal 迁移包使用说明

这个包用于把当前跨模态实验搬到另一台服务器。它包含最新规则类和实体类数据、全部当前脚本、测试、运行说明，以及已完成实验的结果摘要和不超过 200 MB 的预测文件。

包中不包含 `.venv`、模型缓存、基础模型权重、优化器 checkpoint 和旧数据。这些文件与服务器 CUDA、驱动和绝对路径绑定，直接搬过去反而容易造成加载错误。原服务器上的完整 checkpoint 仍在 `runs/`，需要时可以单独复制；本包中的 `runs/` 只保留可复核的 manifest、日志、报告和预测。

## 解压与检查

```bash
tar --use-compress-program=unzstd -xf xmodal_core_transfer_20261022.tar.zst
cd 迁移包_xmodal_20261022
python - <<'PY'
import hashlib, json
from pathlib import Path
m=json.loads(Path('transfer_manifest.json').read_text())
for rel, expected in m['sha256'].items():
    h=hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    assert h==expected, (rel,h,expected)
print('transfer manifest verified:', len(m['sha256']), 'files')
PY
```

## 新服务器准备

1. 确认 NVIDIA 驱动、CUDA、PyTorch、Transformers 和 ms-swift 版本兼容。源服务器版本可看 `environment_pip_freeze.txt` 和 `environment_nvidia_smi.txt`。
2. 创建新的 Python 环境并安装依赖；不要直接复制源服务器 `.venv`。
3. 下载或挂载 Qwen3-VL-4B-Instruct、Qwen2.5-VL-7B-Instruct、GLM-4.6V-Flash 三个基础模型。
4. 修改 `scripts/run_v7_full_experiment.py` 中 `MODELS['qwen25']` 的源服务器绝对路径，使其指向新服务器的 Qwen2.5-VL 权重。Qwen3 和 GLM 路径也要确认。
5. 把 `data/rules_repaired_single_image_v7` 和 `data/entity1020/merged_1109` 作为数据根目录传给脚本；不要混入旧版数据。

## 迁移后先做 smoke test

```bash
python scripts/run_v7_full_experiment.py --root data/rules_repaired_single_image_v7 \
  --work runs/new_smoke --prepare-only
python scripts/run_v7_smokes.py
```

正式运行前要确认：模型模板可加载、文字输入和图像输入都可处理、LoRA checkpoint 可保存并重新加载、loss 不是 NaN。完整训练按 `runs/README.md` 和各脚本的参数执行；不要根据评测集挑 checkpoint。

## 当前实验边界

- 规则类当前版本是 v7，1,000 条规则、四方向各 2,000 道题、3,000 张原图。
- 实体类当前版本是 `merged_1109`，1,109 组知识、四方向各 2,218 道题。
- `v7_process_scale64_20261021` 是 64 条规则的小规模 TEXT 实验，不是完整规则实验；迁移包中的记录用于复核和续跑参考。
- 原服务器仍有一个活动的 64 条规则工作流；迁移包是在不停止该工作流的情况下制作的，包内是快照。

## 文件完整性

`transfer_manifest.json` 记录每个迁移文件的 SHA256。数据集本身的原始 manifest、图片哈希和四方向标签也随包保留。打包不会修改原图、标签、题目或历史结果。

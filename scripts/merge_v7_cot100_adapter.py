"""Create a separate BF16 serving copy; preserve original base and LoRA files."""
import json
import shutil
from pathlib import Path

from run_v7_full_experiment import ROOT, MODELS, save, sha

ADAPTER = ROOT / 'runs/v7_process_scale64_20261021/seed42/definitions/checkpoint/checkpoint-3840'
DEST = ROOT / 'runs/v7_cot100_20261021/vllm_merged_model'


def main():
    import torch
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(8)
    DEST.mkdir(exist_ok=True)
    cfg = json.loads((ADAPTER / 'adapter_config.json').read_text())
    assert not cfg['use_rslora'] and not cfg['use_dora'] and cfg['bias'] == 'none'
    assert not cfg['rank_pattern'] and not cfg['alpha_pattern'] and not cfg['modules_to_save']
    lora = load_file(str(ADAPTER / 'adapter_model.safetensors'), device='cpu')
    prefix = 'base_model.model.'
    updates = {}
    for key, a in lora.items():
        if not key.endswith('.lora_A.weight'):
            continue
        b = lora[key.replace('.lora_A.weight', '.lora_B.weight')]
        assert key.startswith(prefix) and '.language_model.' in key
        base_key = key[len(prefix):].replace('.lora_A.weight', '.weight')
        updates[base_key] = (a, b)
    assert len(updates) * 2 == len(lora) == 504
    touched = []
    for src in sorted(MODELS['qwen3'].glob('model-*.safetensors')):
        tensors = load_file(str(src), device='cpu')
        for key in sorted(set(tensors) & set(updates)):
            a, b = updates[key]
            original = tensors[key]
            delta = (b.float() @ a.float()) * (cfg['lora_alpha'] / cfg['r'])
            assert delta.shape == original.shape and torch.isfinite(delta).all()
            tensors[key] = (original.float() + delta).to(original.dtype).contiguous()
            touched.append(key)
        save_file(tensors, str(DEST / src.name), metadata={'format': 'pt'})
        del tensors
        print('saved', src.name, flush=True)
    assert set(touched) == set(updates)
    for src in MODELS['qwen3'].iterdir():
        if src.is_file() and src.suffix in ('.json', '.jinja'):
            shutil.copy2(src, DEST / src.name)
    save(DEST / 'merge_manifest.json', dict(base=str(MODELS['qwen3']), adapter=str(ADAPTER),
        adapter_sha256=sha(ADAPTER / 'adapter_model.safetensors'), merged_layers=len(touched),
        method='BF16 round(base.float + B.float @ A.float * alpha/r); vision unmodified.',
        note='Serving conversion only, no training. BF16 rounding may differ from unmerged runtime LoRA.',
        files={p.name:sha(p) for p in sorted(DEST.glob('*.safetensors'))},
        touched=touched))


if __name__ == '__main__':
    main()

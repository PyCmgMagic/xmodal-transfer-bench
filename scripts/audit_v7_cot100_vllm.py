"""Independently verify serving conversion, prompts, completion tokens and scoring."""
import argparse
import json

import audit_v7_cot100 as audit
import eval_v7_cot100 as ex
import eval_v7_cot100_vllm as run
from run_v7_full_experiment import MODELS, rows, save, sha


def verify_merge():
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    merge = run.read(run.MERGED / 'merge_manifest.json')
    adapter = load_file(str(ex.ADAPTER / 'adapter_model.safetensors'))
    cfg = run.read(ex.ADAPTER / 'adapter_config.json')
    touched = set(merge['touched'])
    matched, unchanged, visual = [], 0, 0
    for src in sorted(MODELS['qwen3'].glob('model-*.safetensors')):
        orig = load_file(str(src))
        new = load_file(str(run.MERGED / src.name))
        assert orig.keys() == new.keys()
        assert sha(run.MERGED / src.name) == merge['files'][src.name]
        for name, weight in orig.items():
            if name in touched:
                key = 'base_model.model.' + name.replace('.weight', '.lora_A.weight')
                a = adapter[key]
                b = adapter[key.replace('.lora_A.weight', '.lora_B.weight')]
                expected = (weight.float() + (b.float() @ a.float()) * (cfg['lora_alpha']/cfg['r'])).to(weight.dtype)
                assert torch.equal(new[name], expected), name
                matched.append(name)
            else:
                assert torch.equal(weight, new[name]), name
                unchanged += 1
                visual += int(name.startswith('model.visual.'))
        del orig, new
    assert set(matched) == touched and visual > 0
    save(run.OUT / 'merge_audit.json', dict(status='pass', merged_language_layers=len(matched),
        unchanged_tensors=unchanged, unchanged_vision_tensors=visual,
        method='Recompute every merged tensor; exact equality for all unmodified tensors.'))


def main(check_merge):
    run.setup()
    audit.main()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(MODELS['qwen3']), local_files_only=True)
    checked = {}
    for arm in ('base', 'trained'):
        path = run.OUT / arm / 'predictions.jsonl'
        if not path.exists():
            continue
        records = rows(path)
        probes = rows(run.OUT / 'questions.jsonl') + rows(run.OUT / 'recall_probes.jsonl')
        if sum(r['protocol'] == 'recall' for r in records) == 50:
            probes += ex.applications(records)
        lookup = {p['id']: p for p in probes}
        for r in records:
            p = lookup[r['id']]
            expected = run.prompt(p, processor)
            assert r['rendered_prompt'] == expected['prompt']
            assert r['rendered_prompt_sha256'] == run.ev.digest(expected['prompt'])
            decoded = processor.tokenizer.decode(r['generated_token_ids'], skip_special_tokens=True)
            assert decoded == r['response'], (r['id'], decoded[-100:], r['response'][-100:])
            assert r['serving_model'] == str(run.MERGED if arm == 'trained' else MODELS['qwen3'])
            assert r['engine'] == 'vllm' and r['arm'] == arm
            if p.get('modality') == 'T':
                assert not p.get('images') and '<|image_pad|>' not in expected['prompt']
            elif p.get('modality') == 'V':
                assert len(expected['multi_modal_data']['image']) == 1
                assert expected['prompt'].count('<|image_pad|>') == 1
        checked[arm] = dict(records=len(records), complete=len(records) == 450,
            tokenizer_decoding_verified=True, rendered_prompts_verified=True)
    if check_merge:
        verify_merge()
    save(run.OUT / 'engine_audit.json', dict(status='pass', checked=checked,
        complete=len(checked) == 2 and all(r['complete'] for r in checked.values())))
    print(json.dumps(checked))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-merge', action='store_true')
    main(parser.parse_args().check_merge)

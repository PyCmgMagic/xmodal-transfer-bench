#!/usr/bin/env python3
"""Measure how well an untrained model answers old and new condition probes."""
import argparse
import copy
import os
from pathlib import Path
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, environment, sha
from v7_gpu_resources import reserve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    root = a.root.resolve()
    out = root / 'probe_quality_control'
    out.mkdir(exist_ok=True)
    selected = []
    for arm in ('structured', 'natural'):
        selected += [{**r, 'source_arm': arm} for r in rows(root / arm / 'probes.jsonl') if r['condition'] == 'atomic_condition']
    assert len(selected) == 48
    ev.save_rows(out / 'probes.jsonl', selected)
    work, data = ROOT / 'runs/v7_full_experiment', ROOT / 'rules_repaired_single_image_v7'
    cp = Path(read(work / 'jobs/qwen3_sft_text/checkpoint_manifest.json')['path'])
    save(out / 'manifest.json', {'model': str(MODELS['qwen3']), 'loaded_but_disabled_adapter': ev.checkpoint(cp),
         'probe_type': 'condition question quality control with untrained base', 'max_batch_size': 8,
         'max_tokens': 32, 'temperature': 0, 'seed': 42, 'gpu': a.gpu,
         'script_sha256': sha(Path(__file__)), 'created_at': ev.now()})
    with reserve(work, a.gpu, str(out), 12000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(environment(a.gpu, data))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16,
            device_map='cuda:0', max_batch_size=8, attn_impl='sdpa')
        engine.template.enable_thinking = False
        results = []
        with engine.model.disable_adapter():
            for start in range(0, len(selected), 8):
                batch = selected[start:start + 8]
                responses = engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch],
                    RequestConfig(max_tokens=32, temperature=0, seed=42), use_tqdm=False)
                assert len(batch) == len(responses)
                for r, response in zip(batch, responses):
                    if isinstance(response, Exception):
                        raise response
                    text = response.choices[0].message.content or ''
                    results.append({'source_arm': r['source_arm'], 'id': r['id'], 'gold': r['gold'],
                        'response': text, 'correct': ev.letter(text) == r['gold'], 'prompt_sha256': ev.digest(r['messages'])})
        ev.save_rows(out / 'predictions.jsonl', results)
        metrics = {arm: {'n': sum(r['source_arm'] == arm for r in results),
                  'correct': sum(r['correct'] for r in results if r['source_arm'] == arm)} for arm in ('structured', 'natural')}
        save(out / 'report.json', {'status': 'complete', 'metrics': metrics, 'finished_at': ev.now()})
        print(metrics, flush=True)


if __name__ == '__main__':
    main()

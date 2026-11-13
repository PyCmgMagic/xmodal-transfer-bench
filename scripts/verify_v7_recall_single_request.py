#!/usr/bin/env python3
"""Check selected recall failures without batching or adapter on/off switching."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    source = a.source.resolve()
    out = source / 'single_request_verification'
    out.mkdir(exist_ok=True)
    selected = ('scale1k_0310_ai_generated', 'scale1k_0947_programmatic',
                'scale1k_0084_programmatic', 'scale1k_0974_programmatic')
    probes = [p for p in rows(source / 'recall_probes.jsonl') if p['qid'] in selected]
    original = {r['qid']: r for r in rows(source / 'predictions.jsonl')
                if r['mode'] == 'adapter' and r['stage'] == 'recall'}
    cp = Path(read(source / 'manifest.json')['config']['adapter']['path'])
    config = {'adapter': ev.checkpoint(cp), 'model': str(MODELS['qwen3']),
              'selection': list(selected), 'selection_reason': 'four illustrative audited failures; not an accuracy sample',
              'max_batch_size': 1, 'max_tokens': 768, 'temperature': 0, 'seed': 42,
              'adapter_switching': False, 'same_prompt': True,
              'source_sha256': sha(source / 'recall_probes.jsonl'), 'script_sha256': sha(Path(__file__))}
    manifest = out / 'manifest.json'
    if manifest.exists():
        assert read(manifest)['config'] == config
    else:
        save(manifest, {'config': config, 'created_at': ev.now()})
    raw = out / 'predictions.jsonl'
    records = rows(raw) if raw.exists() else []
    pending = [p for p in probes if p['id'] not in {r['id'] for r in records}]
    if pending:
        work = ROOT / 'runs/v7_full_experiment'
        with reserve(work, a.gpu, str(out), 22000) as reservation:
            reservation.bind(os.getpid())
            os.environ.update(environment(a.gpu, ROOT / 'rules_repaired_single_image_v7'))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16,
                                        device_map='cuda:0', max_batch_size=1, attn_impl='sdpa')
            engine.template.enable_thinking = False
            save(out / 'adapter_verification.json', {'active_adapters': str(engine.model.active_adapters),
                 'nonzero_lora_B_tensors': sum(bool(torch.count_nonzero(p)) for n, p in engine.model.named_parameters() if 'lora_B' in n)})
            for p in pending:
                response = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']))],
                    RequestConfig(max_tokens=768, temperature=0, seed=42, return_details=True), use_tqdm=False)[0]
                if isinstance(response, Exception):
                    raise response
                c = response.choices[0]
                text = c.message.content or ''
                record = {'id': p['id'], 'qid': p['qid'], 'relation': p['relation'],
                          'response': text, 'finish_reason': c.finish_reason,
                          'prompt_sha256': ev.digest(p['messages']),
                          'identical_prompt_to_batch8': ev.digest(p['messages']) == original[p['qid']]['prompt_sha256'],
                          'identical_response_to_batch8': text == original[p['qid']]['response'],
                          'raw_response': dataclasses.asdict(response), 'saved_at': ev.now()}
                with raw.open('a') as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                records.append(record)
                print(p['id'], record['identical_response_to_batch8'], flush=True)
    assert len(records) == len(probes) == 4
    save(out / 'report.json', {'status': 'complete', 'n': 4,
         'identical_prompts': sum(r['identical_prompt_to_batch8'] for r in records),
         'identical_responses': sum(r['identical_response_to_batch8'] for r in records),
         'truncated': sum(r['finish_reason'] == 'length' for r in records), 'finished_at': ev.now()})


if __name__ == '__main__':
    main()

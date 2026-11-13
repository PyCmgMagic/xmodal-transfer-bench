#!/usr/bin/env python3
"""Independent final re-scoring and adapter delta checks for the frozen 64-rule run."""
import json
import re
from collections import Counter
from pathlib import Path

from audit_v7_process_confirmation import rows, read, sha, digest, signature

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v7_process_scale64_20261021'


def main():
    manifest = read(OUT / 'manifest.json')
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected, path
    for path, expected in manifest['files'].items():
        assert sha(OUT / path) == expected, path
    for path, expected in manifest['base']['sha256'].items():
        assert sha(Path(manifest['base']['path']) / path) == expected
    assert read(OUT / 'independent_data_audit.json')['status'] == 'pass'
    scored = rows(OUT / 'scored_predictions.jsonl')
    indexed = {(r['arm'], r['id']): r for r in scored}
    assert len(indexed) == len(scored) == 53120
    probes = {p['id']: p for p in rows(OUT / 'probes.jsonl')}
    summary = read(OUT / 'summary.json')
    tasks = [('base', OUT / 'base/evaluation')]
    tasks += [(f'{seed}:{arm}', OUT / f'seed{seed}' / arm / 'evaluation')
              for seed in manifest['seeds'] for arm in manifest['arms']]
    for task, folder in tasks:
        counts, correct, invalid, truncated = Counter(), Counter(), Counter(), Counter()
        predictions = rows(folder / 'predictions.jsonl')
        assert len(predictions) == len({r['id'] for r in predictions}) == len(probes)
        for r in predictions:
            p = probes[r['id']]
            assert r['input_sha256'] == digest(dict(messages=p['messages'], images=p.get('images', [])))
            assert all(r[key] == p[key] for key in ('qid', 'gold', 'condition', 'protocol', 'input_mod'))
            response = r['response'].strip()
            if p['protocol'] == 'short':
                answer = response if response in tuple('ABCD') else None
            elif p['protocol'] == 'reasoning':
                matches = re.findall(r'(?im)^\s*Answer:\s*([A-D])\s*[.!]?\s*$', response)
                answer = matches[0] if len(matches) == 1 else None
            else:
                answer = None
            is_fact = p['fmt'] == 'fact'
            good = (' '.join(response.lower().split()) == ' '.join(p['gold'].lower().split())) if is_fact else answer == p['gold']
            bad_format = not is_fact and answer is None
            z = indexed[task, r['id']]
            assert z['response'] == r['response'] and z['scored_correct'] == good
            assert z['scored_answer'] == answer and z['scored_format_error'] == bad_format
            key = ':'.join(p[c] for c in ('condition', 'protocol', 'input_mod'))
            counts[key] += 1; correct[key] += good; invalid[key] += bad_format
            truncated[key] += r['finish_reason'] == 'length'
        for key in counts:
            s = summary[task][key]
            assert (s['n'], s['correct'], s['format_errors'], s['truncated']) == (counts[key], correct[key], invalid[key], truncated[key])
    from safetensors import safe_open
    deltas = []
    for seed in manifest['seeds']:
        start = OUT / f'seed{seed}/definitions/checkpoint/checkpoint-3840'
        for arm in manifest['arms']:
            cp = OUT / f'seed{seed}' / arm / 'checkpoint/checkpoint-3840'
            args, state = read(cp / 'args.json'), read(cp / 'trainer_state.json')
            assert state['global_step'] == 3840 and abs(state['epoch'] - 3) < 1e-6
            assert args['seed'] == args['data_seed'] == seed
            assert args['freeze_vit'] and args['freeze_aligner'] and args['lora_rank'] == 32
            assert not args.get('adapters') if arm == 'definitions' else args['adapters'] == [str(start)]
            cp_manifest = read(cp.parent.parent / 'checkpoint_manifest.json')['model']
            identity = read(cp.parent.parent / 'evaluation/eval_manifest.json')['model']
            assert identity == cp_manifest and identity['path'] == str(cp)
            for filename, expected in identity['sha256'].items():
                assert sha(cp / filename) == expected
            with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as weights:
                keys = list(weights.keys())
                assert keys and not any('.visual.' in k or '.merger.' in k for k in keys)
                assert all(weights.get_tensor(k).isfinite().all().item() for k in keys)
                assert all(weights.get_tensor(k).count_nonzero().item() for k in keys if 'lora_B' in k)
                if arm != 'definitions':
                    with safe_open(str(start / 'adapter_model.safetensors'), framework='pt', device='cpu') as before:
                        assert set(keys) == set(before.keys())
                        changed = [k for k in keys if not weights.get_tensor(k).equal(before.get_tensor(k))]
                    assert len(changed) == len(keys)
                    deltas.append(dict(seed=seed, arm=arm, changed=len(changed), total=len(keys)))
    data = ROOT / 'rules_repaired_single_image_v7'
    images = rows(data / 'image_hashes.jsonl'); assert len(images) == 3000
    for r in images:
        assert sha(data / r['path']) == r['source_sha256'] == r['repaired_sha256']
    result = dict(status='pass', predictions_verified=53120, final_checkpoints=9,
                  original_images_verified=3000, sibling_weight_updates=deltas,
                  independently_rescored=True, frozen_sources_verified=True)
    (OUT / 'final_completion_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

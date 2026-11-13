"""Strictly rescore explicit-reasoning generations without hiding malformed markup."""
import json
import re
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, save, sha

OUT = ROOT / 'runs/v7_atomic_cot_diagnostic_20261021'


def final_letter(text):
    text = text.strip()
    if text in 'ABCD' and len(text) == 1:
        return text
    # Qwen3-VL-Instruct may return an open <think> marker as ordinary content.
    # Accept only one unambiguous terminal FINAL marker; record markup errors separately.
    markers = re.findall(r'\bFINAL\s*[:：]\s*([ABCD])\b', text, re.I)
    final = re.search(r'\bFINAL\s*[:：]\s*([ABCD])\s*\Z', text, re.I)
    if len(markers) == 1 and final:
        return final[1].upper()
    return None


def main():
    result = {}
    all_rows = []
    for arm in ('base', 'trained'):
        folder = OUT / arm
        records = rows(folder / 'predictions.jsonl')
        assert len(records) == len({r['id'] for r in records})
        metrics = {}
        for protocol in ('direct_cot', 'self_recall_cot'):
            for modality in ('T', 'V'):
                rr = [r for r in records if (r['protocol'], r['modality']) == (protocol, modality)]
                pred = [final_letter(r['response']) for r in rr]
                n = len(rr)
                valid = sum(x is not None for x in pred)
                correct = sum(x == r['gold'] for x, r in zip(pred, rr))
                malformed = sum(r['response'].count('<think>') != r['response'].count('</think>') for r in rr)
                metrics[f'{protocol}_{modality}'] = dict(n=n, correct=correct,
                    accuracy=correct/n if n else None, valid=valid,
                    valid_accuracy=correct/valid if valid else None,
                    invalid_final= n-valid, malformed_think_markup=malformed,
                    truncated=sum(r['finish_reason'] == 'length' for r in rr))
                for letter, r in zip(pred, rr):
                    all_rows.append(dict(arm=arm, id=r['id'], qid=r['qid'], protocol=protocol,
                        modality=modality, task=r['task'], gold=r['gold'], predicted=letter,
                        correct=letter == r['gold'], invalid_final=letter is None,
                        malformed_think_markup=r['response'].count('<think>') != r['response'].count('</think>'),
                        finish_reason=r['finish_reason'], response=r['response']))
        result[arm] = metrics
        assert len(records) == 64, (arm, len(records))
    ev.save_rows(OUT / 'scored_predictions.jsonl', all_rows)
    save(OUT / 'cot_accuracy_report.json', dict(status='complete', result=result,
        scoring='A single explicit terminal FINAL: A/B/C/D is scored even if the model emitted an unclosed literal <think> marker. Such malformed markup is reported separately; only missing/ambiguous final answers are invalid.',
        caveats=['8 rules, 16 questions per modality; diagnostic sample, not full v7 accuracy.',
                 'Training model used the same LoRA as prior recall test and has an earlier task warmup.',
                 'Self-recall condition includes the model’s own uncorrected recall; image results are text-assisted.',
                 'Visible explanations are generated rationale, not guaranteed faithful internal computation.'],
        source_sha256={arm: sha(OUT / arm / 'predictions.jsonl') for arm in ('base', 'trained')}))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

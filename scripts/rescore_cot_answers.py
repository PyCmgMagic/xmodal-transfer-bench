#!/usr/bin/env python3
"""Add conservative end-of-response answer recovery without changing formal scores."""
import argparse
import csv
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/qwen3_lora_cot_eval_20261025'

# Only explicit answer declarations ending the response are considered. This avoids
# picking a letter merely mentioned in the chain of thought.
BOXED = re.compile(r'\\boxed\s*\{\s*([A-D])\s*\}', re.I)
TERMINAL = re.compile(
    r'(?:^|[.!?;:\n]\s*)'
    r'(?:the\s+)?(?:final\s+)?(?:answer|choice|option|selection|result|decision)'
    r'(?:\s+is)?\s*(?:option\s+)?[:：=]?\s*[*_`]*([A-D])[*_`]*'
    r'\s*[.!?]*\s*$', re.I)
TERMINAL_SHORT = re.compile(
    r'(?:^|[.!?;:\n]\s*)(?:final|answer)\s*[:：=]\s*[*_`]*([A-D])[*_`]*\s*[.!?]*\s*$', re.I)
TERMINAL_DECLARATION = re.compile(
    r'(?:^|[.!?;:\n]\s*)'
    r'(?:therefore,?\s+)?(?:the\s+)?(?:correct\s+)?(?:answer|choice|option|selection|result|decision|output)'
    r'\s*(?:is|=)\s*(?:option\s+)?[*_`]*([A-D])[*_`]*\s*[.!?]*\s*$', re.I)
TERMINAL_CORRECT = re.compile(
    r'(?:^|[.!?;:\n]\s*)(?:only\s+)?(?:option\s+)?[*_`]*([A-D])[*_`]*\s+is\s+'
    r'(?:the\s+)?correct\s+(?:answer|choice|option)\s*[.!?]*\s*$', re.I)
TERMINAL_VERB = re.compile(
    r'(?:^|[.!?;:\n]\s*)(?:therefore,?\s+)?(?:choose|select|return|output)\s+'
    r'(?:option\s+)?[*_`]*([A-D])[*_`]*\s*[.!?]*\s*$', re.I)


def recovered_answer(response):
    if not isinstance(response, str):
        return None
    text = response.strip()
    if text.count('<think>') != text.count('</think>'):
        return None
    text = text.rsplit('</think>', 1)[-1].strip()
    boxed = list(BOXED.finditer(text))
    if boxed and not text[boxed[-1].end():].strip(' \t\r\n.!?*_`'):
        # Multiple differing boxed answers at the end are ambiguous.
        terminal_boxed = [m.group(1).upper() for m in boxed
                          if not text[m.end():].strip(' \t\r\n.!?*_`')]
        return terminal_boxed[-1] if len(set(terminal_boxed)) == 1 else None
    for pattern in (TERMINAL_SHORT, TERMINAL, TERMINAL_DECLARATION,
                    TERMINAL_CORRECT, TERMINAL_VERB):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def rescore(work):
    manifest = json.loads((work/'manifest.json').read_text())
    by_cell = []
    all_rows = []
    for job in manifest['jobs']:
        labels_path = work/'data'/job['dataset']/'eval_labels.jsonl'
        labels = {r['id']: r for r in (json.loads(x) for x in labels_path.open())}
        for direction in job['directions']:
            folder = work/'jobs'/job['name']/direction
            report_path = folder/'report.json'
            if not report_path.exists():
                continue
            raw_path = folder/'raw.jsonl'
            scored_path = folder/'recovered_results.jsonl'
            existing = [json.loads(x) for x in raw_path.open()]
            answers = []
            strict_valid = strict_correct = recovered_valid = recovered_correct = 0
            for item in existing:
                pred = recovered_answer(item.get('response', ''))
                gold = labels[item['id']]['answer']
                strict = item.get('strict_predicted')
                if strict is not None:
                    strict_valid += 1
                    strict_correct += strict == gold
                recovered_valid += pred is not None
                recovered_correct += pred == gold
                answers.append({'id': item['id'], 'qid': item.get('qid'), 'gold': gold,
                                'strict_predicted': strict, 'recovered_predicted': pred,
                                'recovered_correct': pred == gold,
                                'finish_reason': item.get('finish_reason'),
                                'recovered_from_strict_invalid': strict is None and pred is not None})
            # Re-derive strict predictions from formal results, which are written
            # alongside report.json only after a full cell has completed.
            formal = {r['id']: r for r in
                      (json.loads(x) for x in (folder/'results.jsonl').open())}
            for row in answers:
                row['strict_predicted'] = formal[row['id']]['predicted']
                row['strict_correct'] = formal[row['id']]['correct']
                row['recovered_from_strict_invalid'] = row['strict_predicted'] is None and row['recovered_predicted'] is not None
            strict_valid = sum(r['strict_predicted'] is not None for r in answers)
            strict_correct = sum(r['strict_correct'] for r in answers)
            recovered_valid = sum(r['recovered_predicted'] is not None for r in answers)
            recovered_correct = sum(r['recovered_correct'] for r in answers)
            n = len(answers)
            score = {'dataset': job['dataset'], 'objective': job['objective'],
                     'learning_modality': job.get('learning_modality', job.get('modality')),
                     'direction': direction, 'total': n,
                     'strict_accuracy': strict_correct/n if n else None,
                     'strict_valid': strict_valid, 'strict_invalid': n-strict_valid,
                     'recovered_accuracy': recovered_correct/n if n else None,
                     'recovered_valid': recovered_valid, 'unresolved': n-recovered_valid,
                     'recovered_correct': recovered_correct,
                     'recovered_from_strict_invalid': sum(r['recovered_from_strict_invalid'] for r in answers),
                     'parser': 'terminal_explicit_answer_or_boxed_v1',
                     'scored_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            with scored_path.open('w') as f:
                for row in answers:
                    f.write(json.dumps(row, ensure_ascii=False)+'\n')
            (folder/'recovered_report.json').write_text(json.dumps(score, ensure_ascii=False, indent=2)+'\n')
            by_cell.append(score)
            all_rows.extend(answers)
    (work/'recovered_summary.json').write_text(json.dumps(by_cell, ensure_ascii=False, indent=2)+'\n')
    if by_cell:
        with (work/'recovered_summary.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(by_cell[0]))
            writer.writeheader(); writer.writerows(by_cell)
    return by_cell


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    args=parser.parse_args()
    reports=rescore(args.work.resolve())
    print(json.dumps({'completed_cells_rescored':len(reports),
                      'predictions_rescored':sum(r['total'] for r in reports),
                      'outputs':'recovered_summary.json, recovered_summary.csv, per-cell recovered_report.json/recovered_results.jsonl'},ensure_ascii=False))


if __name__ == '__main__':
    main()

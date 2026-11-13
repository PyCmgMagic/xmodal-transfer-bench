#!/usr/bin/env python3
"""Post-evaluation operand diagnostics; unparsed prose is never graded wrong."""
import json
import operator
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v7_process_confirmation_20261020'
OPS = {'==': operator.eq, '>': operator.gt, '>=': operator.ge,
       '<': operator.lt, '<=': operator.le}
OP_NAMES = {'eq': '==', 'gt': '>', 'ge': '>=', 'lt': '<', 'le': '<='}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def value(field, text):
    return float(text) if field == 'value' else str(text).strip().lower()


def main():
    assert json.loads((OUT/'independent_final_audit.json').read_text())['status'] == 'pass'
    probes = {p['id']: p for p in rows(OUT/'probes.jsonl')}
    knowledge = {k['id']: json.loads(k['fact']['quote']) for k in rows(OUT/'knowledge.jsonl')}
    blocks = {b['id']: b for b in rows(OUT/'blocks.jsonl')}
    original = {(p['qid'], p['task']): p['observations'] for p in probes.values()
                if p['condition'] == 'original' and p['input_mod'] == 'T'}
    diagnostics = []
    for r in rows(OUT/'scored_predictions.jsonl'):
        if r['protocol'] != 'reasoning':
            continue
        p = probes[r['id']]
        rule = knowledge[r['qid']]
        obs = (original[(r['qid'], r['task'])] if r['condition'] == 'original'
               else blocks[r['block']]['observations'])
        expected = {}
        for letter, o in zip('ABCD', obs):
            flags = [OPS[OP_NAMES[c['operator']]](o[c['field']], c['value']) for c in rule['conditions']]
            expected[letter] = 'KEEP' if (any(flags) if rule['mode'] == 'any' else all(flags)) else 'OMIT'
        detail = []
        for letter, o in zip('ABCD', obs):
            lines = re.findall(r'(?im)^' + letter + r':\s*(.+)$', r['response'])
            z = {'candidate': letter, 'parseable': False}
            if len(lines) == 1:
                parts = lines[0].split(';')
                comparisons = [re.fullmatch(r'(category|material|value)\s+(.+?)\s+(==|>=|<=|>|<)\s+(.+?)\s*=\s*(YES|NO)', s.strip())
                               for s in parts[:2]]
                logic = re.fullmatch(r'(OR|AND)\s*=>\s*(KEEP|OMIT)\.?', parts[2].strip()) if len(parts) == 3 else None
                if len(comparisons) == 2 and all(comparisons) and logic:
                    checks = []
                    try:
                        for match, condition in zip(comparisons, rule['conditions']):
                            field, left, op, right, answer = match.groups()
                            lhs, rhs = value(field, left), value(field, right)
                            checks.append(dict(field=field, lhs=lhs, operator=op, rhs=rhs, comparison_answer=answer,
                                rule_parameter_matches=(field == condition['field'] and op == OP_NAMES[condition['operator']]
                                                        and rhs == value(field, condition['value'])),
                                observation_matches=(field in o and lhs == value(field, o[field])),
                                self_comparison_valid=((OPS[op](lhs, rhs)) == (answer == 'YES'))))
                        local = [c['comparison_answer'] == 'YES' for c in checks]
                        logic_name, decision = logic.groups()
                        own_decision = any(local) if logic_name == 'OR' else all(local)
                        z.update(parseable=True, comparisons=checks,
                            rule_parameters_match=all(c['rule_parameter_matches'] for c in checks),
                            observations_match=all(c['observation_matches'] for c in checks),
                            comparisons_self_consistent=all(c['self_comparison_valid'] for c in checks),
                            logic_matches=logic_name == ('OR' if rule['mode'] == 'any' else 'AND'),
                            decision_self_consistent=own_decision == (decision == 'KEEP'),
                            decision_matches_truth=decision == expected[letter])
                    except (ValueError, TypeError):
                        pass
            detail.append(z)
        result_lines = re.findall(r'(?im)^Result:\s*(.+)$', r['response'])
        result = None
        if len(result_lines) == 1:
            pairs = re.findall(r'([A-D])\s*=\s*(KEEP|OMIT)', result_lines[0])
            if [c for c, _ in pairs] == list('ABCD'):
                result = dict(pairs)
        covered = all(z['parseable'] for z in detail)
        diagnostics.append(dict(arm=r['arm'], id=r['id'], qid=r['qid'], task=r['task'],
            condition=r['condition'], input_mod=r['input_mod'], answer_correct=r['scored_correct'],
            format_error=r['scored_format_error'], expected_result=expected,
            result_parseable=result is not None, result_matches_truth=result == expected,
            all_candidate_comparisons_parseable=covered,
            all_rule_parameters_match=covered and all(z['rule_parameters_match'] and z['logic_matches'] for z in detail),
            all_observations_match=covered and all(z['observations_match'] for z in detail),
            all_comparisons_self_consistent=covered and all(z['comparisons_self_consistent'] and z['decision_self_consistent'] for z in detail),
            candidates=detail))
    (OUT/'operand_diagnostics.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in diagnostics))
    summary = {}
    for arm in ('base','definitions','definition_replay','process'):
        summary[arm] = {}
        for condition, mod in (('new_holdout','T'),('original','T'),('original','V')):
            rr = [r for r in diagnostics if (r['arm'],r['condition'],r['input_mod']) == (arm,condition,mod)]
            summary[arm][condition+':'+mod] = dict(n=len(rr), **Counter({
                key:sum(bool(r[key]) for r in rr) for key in ('answer_correct','format_error','result_parseable','result_matches_truth',
                'all_candidate_comparisons_parseable','all_rule_parameters_match','all_observations_match','all_comparisons_self_consistent')}))
    (OUT/'operand_diagnostics_summary.json').write_text(json.dumps(dict(summary=summary,
        caveat='Posthoc syntactic coverage, not semantic grading of free prose. Image observation mismatches are against text annotations and require visual review. Float spelling normalized. Final correctness uses unchanged main parser.'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

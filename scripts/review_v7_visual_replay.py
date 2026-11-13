#!/usr/bin/env python3
"""Persist human judgments of this experiment's fixed32recall responses."""
import explore_v7_visual_replay as trial
import run_v7_evaluation as ev
from run_v7_full_experiment import rows, read, save, sha


def main():
    out = trial.OUT
    knowledge = {k['id']: k['fact']['quote'] for k in rows(trial.SOURCE / 'target_knowledge.jsonl')}
    wrong = {
        'scale1k_0087_programmatic': '误答[1,4]，正确区间为[12,16)。',
        'scale1k_0271_programmatic': '把菱形或黄色的OR筛选错答为[1,100]数值条件，并错误使用ALL。',
        'scale1k_0275_programmatic': '误答[1,12]，正确区间为[4,6)。',
        'scale1k_0954_programmatic': '把五边形或白色的OR筛选错答为[1,6]数值条件。',
        'scale1k_0956_programmatic': '把五边形或蓝色的OR筛选错答为[1,4]数值条件。',
        'scale1k_0957_programmatic': '把正方形或红色的OR筛选错答为[1,2]数值条件。',
        'scale1k_0964_programmatic': '误答[1,4]，正确区间为[4,14)。',
        'scale1k_0965_programmatic': '误答[1,6]，正确区间为[9,14)。',
    }
    result, summary = [], {}
    for arm in trial.ARMS:
        folder = out / arm
        assert read(folder / 'eval_state.json')['status'] == 'complete'
        raw = folder / 'predictions.jsonl'
        rr = [r for r in rows(raw) if r['fmt'] == 'card']
        assert len(rr) == 16
        for r in rr:
            correct = arm == 'text_replay' or r['condition'] == 'recall_image'
            reason = '人工逐条检查：精确条件、ANY/ALL、区间端点与原结构规则一致，整段自洽。' if correct else wrong[r['qid']]
            result.append({'arm': arm, 'id': r['id'], 'qid': r['qid'], 'condition': r['condition'],
                'core_facts_correct': correct, 'whole_response_consistent': correct, 'reason_zh': reason,
                'response': r['response'], 'finish_reason': r['finish_reason'],
                'expected_structured_rule': knowledge[r['qid']], 'source_file': str(raw),
                'source_sha256': sha(raw), 'response_sha256': ev.digest(r['response']),
                'method': 'Human semantic inspection of all32fixed responses; explicit judgments, not an automatic general scorer.'})
        summary[arm] = {}
        for c in ('recall', 'recall_image'):
            group = [r for r in result if r['arm'] == arm and r['condition'] == c]
            assert len(group) == 8
            summary[arm][c] = {'n': 8, 'correct': sum(r['core_facts_correct'] for r in group),
                               'truncated': sum(r['finish_reason'] == 'length' for r in group)}
    assert len(result) == 32
    ev.save_rows(out / 'semantic_reviews.jsonl', result)
    save(out / 'semantic_review_summary.json', summary)


if __name__ == '__main__':
    main()

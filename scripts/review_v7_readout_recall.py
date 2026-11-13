#!/usr/bin/env python3
"""Store explicit human judgments for all120fixed no-image recall outputs."""
import explore_v7_readout_protocol as trial
import run_v7_evaluation as ev
from run_v7_full_experiment import rows, read, save, sha


def main():
    out = trial.OUT
    knowledge = {k['id']: k['fact']['quote'] for k in rows(trial.SOURCE / 'target_knowledge.jsonl')}
    ids = set(knowledge)
    correct = {a: {v: set() for v in ('plain', 'page_memory', 'aligned_plain')} for a in trial.ARMS}
    correct['text']['plain'] = {'scale1k_0087_programmatic', 'scale1k_0271_programmatic'}
    correct['text']['page_memory'] = {f'scale1k_{i}_programmatic' for i in ('0087', '0271', '0956', '0957', '0964')}
    correct['text']['aligned_plain'] = ids
    correct['text_replay']['page_memory'] = ids - {'scale1k_0964_programmatic'}
    correct['text_replay']['aligned_plain'] = ids
    reasons = {
        ('text', 'plain', 'scale1k_0275_programmatic'): '把Lervom区间错答成菱形或黄色。',
        ('text', 'plain', 'scale1k_0954_programmatic'): '输出Delve且只有通用ANY说明，缺五边形/白色的具体事实。',
        ('text', 'plain', 'scale1k_0956_programmatic'): 'Tetkil形状误答square，正确pentagon；颜色blue正确但整条规则错误。',
        ('text', 'plain', 'scale1k_0957_programmatic'): 'Lancos误答diamond/white，应square/red。',
        ('text', 'plain', 'scale1k_0964_programmatic'): '把Lamdal区间错答成菱形或黄色。',
        ('text', 'plain', 'scale1k_0965_programmatic'): '把Vevlev区间错答成菱形或白色。',
        ('text', 'page_memory', 'scale1k_0275_programmatic'): 'Lervom误答[4,14)，应[4,6)。',
        ('text', 'page_memory', 'scale1k_0954_programmatic'): 'Deldev形状误答round，应pentagon。',
        ('text', 'page_memory', 'scale1k_0965_programmatic'): '把Vevlev区间错答成菱形或黄色。',
        ('text_replay', 'page_memory', 'scale1k_0964_programmatic'): 'Lamdal误答[4,6)，应[4,14)。',
    }
    reviewed, summary = [], {}
    for arm in trial.ARMS:
        summary[arm] = {}
        sources = [out / arm, out / 'system_alignment' / arm]
        for folder in sources:
            assert read(folder / 'eval_state.json')['status'] == 'complete'
            raw = folder / 'predictions.jsonl'
            for r in rows(raw):
                if r['fmt'] != 'card':
                    continue
                variant = r['condition']
                success = r['qid'] in correct[arm][variant]
                if success:
                    reason = '精确条件、组合方式、边界及整段应用说明与原规则一致。'
                elif (arm, variant, r['qid']) in reasons:
                    reason = reasons[arm, variant, r['qid']]
                elif variant == 'plain':
                    reason = '编造无关游戏/术语或否认规则存在，缺少目标规则核心事实；即使回答很长也不计正确。'
                elif arm == 'warmup':
                    reason = '未学习目标规则，输出无关数值/字符串映射或通用说明；缺目标具体条件。'
                else:
                    reason = '形状颜色OR规则被写成不相干数值条件，或目标区间阈值/边界错误，部分伴随重复输出；不能算完整回忆。'
                reviewed.append({'arm': arm, 'variant': variant, 'id': r['id'], 'qid': r['qid'],
                    'core_facts_correct': success, 'whole_response_consistent': success,
                    'reason_zh': reason, 'response': r['response'], 'finish_reason': r['finish_reason'],
                    'expected_structured_rule': knowledge[r['qid']], 'source_file': str(raw),
                    'source_sha256': sha(raw), 'response_sha256': ev.digest(r['response']),
                    'method': 'Human review of all120saved outputs. Full concise correct responses checked; explicit wrong core facts or unrelated fabrications suffice to reject long outputs. No automatic general semantic scorer.'})
        for variant in ('plain', 'page_memory', 'aligned_plain'):
            group = [r for r in reviewed if r['arm'] == arm and r['variant'] == variant]
            assert len(group) == 8 and {r['qid'] for r in group} == ids
            summary[arm][variant] = {'n': 8, 'correct': sum(r['core_facts_correct'] for r in group),
                'truncated': sum(r['finish_reason'] == 'length' for r in group)}
    assert len(reviewed) == 120
    ev.save_rows(out / 'semantic_reviews.jsonl', reviewed)
    save(out / 'semantic_review_summary.json', summary)


if __name__ == '__main__':
    main()

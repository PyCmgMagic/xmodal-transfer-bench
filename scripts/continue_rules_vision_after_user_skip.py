#!/usr/bin/env python3
"""Continue only repeat10 apply after a user-stopped repeat1; retain partial reporting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def condition_state(arm, phase, records, stopped):
    if arm == 'repeat1' and phase == 'apply':
        assert stopped['status'] == 'stopped_by_user' and len(records) == stopped['saved'] < 4000
        return 'stopped_by_user'
    assert len(records) == 4000
    return 'complete'


def main(root, work, revision):
    sys.path.insert(0, str(work / 'source_snapshots'))
    from entityqa_common import rows, save, sha, now, write_rows
    from run_rules_glm46_vision_suite import verify, recall_summary
    from run_rules_glm46_rule_given import summaries, pair_counts
    from run_rules_glm46_recall_apply import matched_change
    with (work / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((work / 'experiment.json').read_text())
        verify(work, plan)
        stopped = json.loads((work / 'evaluation_apply/repeat1/status.json').read_text())
        assert stopped['status'] == 'stopped_by_user'
        assert sha(work / 'evaluation_apply/repeat1/predictions.jsonl') == stopped['predictions_sha256']
        def status(stage, **extra):
            save(work / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
                updated_at=now(), user_stopped_stage='apply_repeat1', partial_saved=stopped['saved'], **extra))
        while True:
            values = subprocess.check_output(['nvidia-smi', '-i', '0',
                '--query-gpu=memory.free,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15)
            free, used = [int(s.strip()) for s in values.strip().split(',')]
            if free >= 34 * 1024:
                break
            status('waiting_for_free_gpu', next_stage='apply_repeat10',
                   free_mib=free, required_free_mib=34 * 1024)
            time.sleep(30)
        save(revision / 'gpu_admission.json', dict(checked_at=now(), free_mib=free, used_mib=used,
            required_free_mib=34 * 1024, policy='Continue same GLM workload when sufficient free memory exists; never signal other GPU processes',
            other_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader'], text=True)))
        cmd = [str(root / '.venv-glm/bin/python'), '-u',
            str(work / 'source_snapshots/run_rules_glm46_vision_suite.py'),
            '--root', str(root), '--work', str(work), '--stage', 'apply_repeat10']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_PIXELS='262144',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        save(work / 'apply_repeat10_command.json', cmd)
        with (work / 'apply_repeat10.log').open('a') as log:
            child = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status('apply_repeat10', child_pid=child.pid, log=str(work / 'apply_repeat10.log'))
        if child.wait():
            raise RuntimeError('Repeat10 apply failed; outputs preserved; inspect apply_repeat10.log')
        verify(work, plan)
        conditions, changes = {}, {}
        for arm in ['repeat1', 'repeat10']:
            current = {}
            for phase in ['closed', 'apply']:
                output = work / ('evaluation_closed' if phase == 'closed' else 'evaluation_apply') / arm
                records = rows(output / 'predictions.jsonl')
                assert len(records) == len({(r['id'], r['modality']) for r in records})
                state = condition_state(arm, phase, records, stopped)
                pairs, details = pair_counts(records)
                summary = dict(status=state, arm=arm, phase=phase, saved=len(records), total=4000,
                    groups=summaries(records), paired_outcomes=pairs, evaluation_complete=state == 'complete',
                    source_sha256=sha(output / 'predictions.jsonl'))
                if state == 'stopped_by_user':
                    assert sha(output / 'predictions.jsonl') == stopped['predictions_sha256']
                    summary['stopped_at'] = stopped['updated_at']
                conditions[arm + '/' + phase] = summary
                current[phase] = records
            changes[arm] = {mod: matched_change([r for r in current['closed'] if r['modality'] == mod],
                [r for r in current['apply'] if r['modality'] == mod]) for mod in 'TV'}
        memory = recall_summary(work)
        save(work / 'recall_summary.json', memory)
        save(work / 'comparison.json', dict(status='complete', workflow_status='complete_with_user_skipped_stage',
            all_planned_evaluations_complete=False, stopped_stages=['apply_repeat1'],
            completed_at=now(), conditions=conditions, recall=memory,
            paired_change_from_same_adapter_closed_book=changes,
            partial_results_note='repeat1 apply stopped by user; denominator is saved answers only; not the full 4000-task result'))
        lines = ['# GLM规则视觉学习：含用户停止的部分结果', '',
            '| 重复 | 测试 | 状态 | T题数 | T准确率 | T无效率 | V题数 | V准确率 | V无效率 |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for key, item in conditions.items():
            arm, phase = key.split('/')
            t, v = item['groups']['T/all'], item['groups']['V/all']
            state = '用户停止，部分结果' if item['status'] == 'stopped_by_user' else '完成'
            label = '直接闭卷' if phase == 'closed' else '回忆后答题'
            lines.append(f'| {arm} | {label} | {state} | {t["n"]} | {t["accuracy"]:.2%} | {t["invalid_rate"]:.2%} | {v["n"]} | {v["accuracy"]:.2%} | {v["invalid_rate"]:.2%} |')
        lines += ['', '用户要求停止repeat1回忆后答题，已保存回复和训练产物均保留。该组仅统计实际完成题，不能视作全部4000题成绩。',
                  'repeat10回忆后答题沿用原冻结协议；原生thinking关闭，简短解释提示，temperature0，输出1024 token。',
                  '流程结束后进入已排队的Qwen3-VL-32B实验；流程完成状态不表示被停止的一组完成全部评测。', '']
        (work / 'RESULTS_ZH.md').write_text('\n'.join(lines))
        plan.update(status='complete', completed_at=now(), all_planned_evaluations_complete=False,
            workflow_status='complete_with_user_skipped_stage', user_stopped_stages=dict(apply_repeat1=stopped),
            recall_scoring_status=memory['status'])
        save(work / 'experiment.json', plan)
        status('complete', workflow_status='complete_with_user_skipped_stage',
            all_planned_evaluations_complete=False, results=str(work / 'RESULTS_ZH.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--revision', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root.resolve(), args.work.resolve(), args.revision.resolve())
    except Exception as error:
        sys.path.insert(0, str(args.work / 'source_snapshots'))
        from entityqa_common import save, now
        save(args.work / 'status.json', dict(stage='failed', error=repr(error), updated_at=now(),
                                           user_stopped_stage='apply_repeat1'))
        raise

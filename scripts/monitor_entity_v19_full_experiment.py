#!/usr/bin/env python3
"""Silent hourly experiment checks and recovery of failed or stalled runs.

Independent of the frozen training/evaluation controller. Recovery uses the
existing stop/start commands, checkpoints and cached predictions. No messages
or desktop notifications are sent. Files live under <work>/monitor/.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from entity_v19_experiment_data import DEFAULT_WORK, ROOT, now, read, save, evaluation_scope, in_evaluation_scope
from v7_gpu_resources import alive

STOP = threading.Event()


def last_json(path):
    """Only inspect the tail, so growing prediction files need not be loaded."""
    if not path.is_file():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        length = stream.tell()
        stream.seek(max(0, length - 32768))
        block = stream.read()
    for line in reversed(block.splitlines()):
        try:
            result = json.loads(line)
            if isinstance(result, dict):
                return result
        except (ValueError, UnicodeDecodeError):
            continue
    return {}


def snapshot(work, stale_seconds=1800):
    wall = time.time()
    controller = read(work / 'controller.json')
    problems = {}
    running = []
    if controller.get('status') == 'running' and not alive(controller.get('pid')):
        problems['controller_dead'] = '实验控制器标记 running，但进程已退出'
    if controller.get('status') == 'failed':
        problems['controller_failed'] = '实验控制器失败：' + controller.get('error', '')
    jobs = []
    for path in sorted((work / 'jobs').glob('*/*/seed_*/state.json')):
        state = read(path)
        name = str(path.parent.relative_to(work / 'jobs'))
        jobs.append({'job': name, **state})
        if state.get('status') == 'failed':
            problems['failed:' + name] = '训练失败：' + name
    # Current controller's durable job/process states take precedence over
    # historical per-model `stopped` records from the stop/restart self-test.
    for path in sorted(work.rglob('*.process.json')):
        state = read(path)
        if state.get('status') not in ('running', 'waiting_for_memory'):
            continue
        name = str(path.parent.relative_to(work))
        if state['status'] == 'waiting_for_memory':
            running.append({'task': name, 'status': 'waiting_for_memory',
                            'free_mib': state.get('free_mib'), 'required_mib': state.get('required_mib')})
            continue
        pid = state.get('pid')
        if not alive(pid):
            if controller.get('status') == 'running':
                problems['dead:' + name] = '任务进程退出但尚未正常完成：' + name
            continue
        stem = path.name.removesuffix('.process.json')
        log = path.with_name(stem + '.log')
        train_metrics = path.parent / 'checkpoints/logging.jsonl'
        raw = path.with_name(stem + '.raw.jsonl')
        progress_file = raw if raw.exists() else (train_metrics if train_metrics.exists() else log)
        updated = progress_file.stat().st_mtime if progress_file.exists() else path.stat().st_mtime
        idle = max(0., wall - updated)
        if idle > stale_seconds:
            problems['stale:' + name] = f'任务超过 {int(stale_seconds / 60)} 分钟没有进度更新：{name}'
        entry = {'task': name, 'status': 'running', 'pid': pid, 'gpu': state.get('gpu'),
                 'seconds_since_progress': round(idle), 'progress_file': str(progress_file)}
        if train_metrics.exists():
            entry['last_metric'] = last_json(train_metrics)
        if raw.exists():
            entry['raw_bytes'] = raw.stat().st_size
        running.append(entry)
    # Detect model-worker failures belonging to this controller invocation.
    started = controller.get('updated_at', controller.get('started_at', ''))
    for model in ('qwen3', 'glm46'):
        state = read(work / f'{model}.status.json')
        if state.get('status') == 'failed' and state.get('updated_at', '') >= started:
            problems['worker:' + model] = f'{model} 队列失败：' + state.get('error', '')
    complete_evaluations = []
    predictions = 0
    for path in work.rglob('progress.json'):
        if not in_evaluation_scope(path, work):
            continue
        state = read(path)
        if state.get('status') == 'complete':
            complete_evaluations.append(str(path.parent.relative_to(work)))
            predictions += state.get('logical_rows', 0)
    return {'checked_at': now(), 'controller': controller, 'controller_alive': alive(controller.get('pid')),
            'training_jobs': dict(Counter(j.get('status') for j in jobs)),
            'completed_training': sorted(j['job'] for j in jobs if j.get('status') == 'complete'),
            'active': running, 'completed_evaluations': complete_evaluations,
            'completed_prediction_rows': predictions,
            'evaluation_seeds': evaluation_scope(work)['evaluation_seeds'],
            'expected_prediction_rows': evaluation_scope(work)['expected_predictions']['total'],
            'problems': problems}


def changes(previous, current):
    events = []
    for key, value in current['problems'].items():
        if key not in previous.get('problems', {}):
            events.append({'kind': 'problem', 'key': key, 'message': value})
    for key in previous.get('problems', {}):
        if key not in current['problems']:
            events.append({'kind': 'recovered', 'key': key, 'message': '异常已解除：' + key})
    before = len(previous.get('completed_training', []))
    after = len(current['completed_training'])
    if after == 12 and before < 12:
        events.append({'kind': 'milestone', 'key': 'all_training', 'message': '12/12 组训练全部完成，正在评测'})
    for model in ('qwen3', 'glm46'):
        if (sum(p.startswith(model + '/') for p in current['completed_training']) == 6 and
            sum(p.startswith(model + '/') for p in previous.get('completed_training', [])) < 6):
            events.append({'kind': 'milestone', 'key': model + '_training', 'message': model + ' 的 6 组训练全部完成'})
    old_status = previous.get('controller', {}).get('status')
    if current['controller'].get('status') == 'complete' and old_status != 'complete':
        events.append({'kind': 'complete', 'key': 'experiment', 'message': '实体类完整实验已完成，训练、评测和统计报告均已生成'})
    return events


def check(work):
    folder = work / 'monitor'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.check.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read(folder / 'latest.json')
        current = snapshot(work)
        events = changes(previous, current)
        with (folder / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(current, ensure_ascii=False) + '\n')
        for event in events:
            event['at'] = current['checked_at']
            event['notification'] = 'disabled_by_user'
            with (folder / 'alerts.jsonl').open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        save(folder / 'latest.json', current)
        text = ['# 实体类实验巡检', '', '检查时间（UTC）：' + current['checked_at'], '',
                '频率：每小时。按用户要求静默运行，不发送任何通知。故障确认后自动断点恢复；超过两小时无任何日志更新才按停滞恢复。', '',
                f"控制器：{current['controller'].get('status')}；进程存活：{current['controller_alive']}", '',
                f"训练完成：{len(current['completed_training'])}/12；评测 seed：{current['evaluation_seeds']}；评测记录：{current['completed_prediction_rows']:,}/{current['expected_prediction_rows']:,}", '']
        for task in current['active']:
            metric = task.get('last_metric', {})
            text.append(f"- {task['task']}：{task['status']}；step={metric.get('global_step/max_steps', '—')}；loss={metric.get('loss', '—')}")
        text += ['', '异常：'] + (list(current['problems'].values()) or ['无。'])
        (folder / 'status.md').write_text('\n'.join(text) + '\n')
    return current


def recovery_reasons(current):
    if current['controller'].get('status') in ('complete', 'stopped'):
        return []
    reasons = [key for key in current['problems'] if not key.startswith('stale:')]
    for task in current['active']:
        if task.get('status') != 'running' or task.get('seconds_since_progress', 0) < 7200:
            continue
        # A long operation can leave JSON metrics unchanged while printing to
        # its log. Require BOTH outputs to be stale before recycling a process.
        log_dir = Path(task['progress_file']).parent
        if log_dir.name == 'checkpoints':
            log_dir = log_dir.parent
        recent_log = max([p.stat().st_mtime for p in log_dir.glob('*.log')] or [0])
        if time.time() - recent_log >= 7200:
            reasons.append('stalled:' + task['task'])
    return reasons


def recover(work, current):
    reasons = recovery_reasons(current)
    if not reasons:
        return
    # Avoid treating the short exit/state-update window as a failed job.
    if STOP.wait(10):
        return
    confirmed = snapshot(work)
    reasons = recovery_reasons(confirmed)
    if not reasons:
        return
    folder = work / 'monitor'
    script = ROOT / 'scripts/run_entity_v19_full_experiment.py'
    assignment = confirmed['controller'].get('gpu_assignment', {'qwen3': '0', 'glm46': '1'})
    action = {'at': now(), 'reasons': reasons, 'action': 'restart_controller_with_checkpoint_and_prediction_resume',
              'notifications': False, 'parameters_changed': False, 'commands': []}
    # Preserve hyperparameters and data. Existing stop confirms all owned
    # process groups exit; start avoids racing two writers to a checkpoint.
    try:
        for verb in ('stop', 'start'):
            if STOP.is_set():
                break
            cmd = [sys.executable, str(script), verb, '--work', str(work)]
            if verb == 'start':
                cmd += ['--gpu0', assignment['qwen3'], '--gpu1', assignment['glm46']]
            with (folder / 'recovery.log').open('a') as stream:
                result = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=90)
            action['commands'].append({'verb': verb, 'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError(f'Recovery {verb} failed: {result.returncode}')
        action['status'] = 'restarted' if len(action['commands']) == 2 else 'interrupted'
    except Exception as exc:
        action['status'] = 'retry_next_check'
        action['error'] = str(exc)
    finally:
        with (folder / 'recoveries.jsonl').open('a') as stream:
            stream.write(json.dumps(action, ensure_ascii=False) + '\n')


def run(work, interval):
    folder = work / 'monitor'
    folder.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())
    with (folder / '.watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not STOP.is_set():
            try:
                current = check(work)
                save(folder / 'watcher.json', {'status': 'running', 'pid': os.getpid(),
                     'interval_seconds': interval, 'notifications': False, 'auto_recover': True,
                     'checked_at': current['checked_at'],
                     'next_check_at_utc': datetime.fromtimestamp(time.time() + interval,
                         tz=timezone.utc).isoformat()})
                if current['controller'].get('status') in ('complete', 'stopped'):
                    break
                recover(work, current)
            except Exception as exc:
                with (folder / 'errors.log').open('a') as stream:
                    stream.write(f'{now()} {type(exc).__name__}: {exc}\n')
            STOP.wait(interval)
        save(folder / 'watcher.json', {'status': 'stopped', 'pid': os.getpid(), 'ended_at': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'run', 'once', 'status', 'stop'))
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--interval', type=int, default=3600)
    args = parser.parse_args()
    work = args.work.resolve()
    folder = work / 'monitor'
    folder.mkdir(parents=True, exist_ok=True)
    if args.action == 'start':
        state = read(folder / 'watcher.json')
        if state.get('status') == 'running' and alive(state.get('pid')):
            print(json.dumps(state, ensure_ascii=False))
            return
        assert args.interval >= 60
        with (folder / 'watcher.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run',
                '--work', str(work), '--interval', str(args.interval)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'interval_seconds': args.interval, 'directory': str(folder)}))
    elif args.action == 'run':
        run(work, args.interval)
    elif args.action == 'once':
        print(json.dumps(check(work), ensure_ascii=False, indent=2))
    elif args.action == 'status':
        print(json.dumps({'watcher': read(folder / 'watcher.json'), 'latest': read(folder / 'latest.json')}, ensure_ascii=False, indent=2))
    else:
        state = read(folder / 'watcher.json')
        if state.get('status') == 'running' and alive(state.get('pid')):
            os.kill(state['pid'], signal.SIGTERM)
        print('Monitor stop requested; experiment is unaffected.')


if __name__ == '__main__':
    main()

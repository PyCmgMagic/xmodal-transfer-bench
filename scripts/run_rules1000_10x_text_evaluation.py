#!/usr/bin/env python3
"""Queue a matched evaluation for the rules-only TEXT comparison."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/rules1000_10x_20261030_text_comparison'
DATA = ROOT / 'data/rules_repaired_single_image_v7'
BASELINE = ROOT / 'runs/rules_full_sft_ep10_20261026'
LEDGER = ROOT / 'runs/qwen3_lora_cot_eval_20261025'
MODEL = ROOT / 'model_cache/Qwen3-VL-4B-Instruct'
EVALUATION_NAME = 'evaluation_cot_box'
EXPERIMENTS = (
    ('text_original', BASELINE / 'jobs/text/checkpoint/checkpoint-630', ('T2T', 'T2V'), 'text'),
    ('text_multiform10', WORK / 'jobs/multiform10/checkpoint/checkpoint-6250', ('T2T', 'T2V'), 'text'),
    ('text_repeat10', WORK / 'jobs/repeat10/checkpoint/checkpoint-6250', ('T2T', 'T2V'), 'text'),
    ('vision_original', BASELINE / 'jobs/vision/checkpoint/checkpoint-630', ('V2T', 'V2V'), 'vision'),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def parent_pid(pid):
    try:
        for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
            if line.startswith('PPid:'):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def belongs_to_controller(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == os.getpid():
            return True
        seen.add(pid)
        pid = parent_pid(pid)
    return False


def external_gpu_processes():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    processes = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        try:
            pid = int(fields[0])
        except (ValueError, IndexError):
            continue
        if not belongs_to_controller(pid):
            try:
                command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(
                    b'\0', b' ').decode(errors='replace').strip()
            except OSError:
                command = fields[1] if len(fields) > 1 else 'unknown'
            processes.append({'pid': pid, 'memory_mib': fields[2] if len(fields) > 2 else None,
                              'command': command})
    return processes


def wait_for_external_gpu_processes(status, status_path, poll_seconds):
    announced = False
    while True:
        processes = external_gpu_processes()
        if not processes:
            status.update(status='waiting_for_gpu', external_gpu_processes=[], updated_at=now())
            save(status_path, status)
            return
        status.update(status='waiting_for_external_gpu', external_gpu_processes=processes,
                      updated_at=now())
        save(status_path, status)
        if not announced:
            print(f'Waiting for unrelated GPU processes to finish: {processes}', flush=True)
            announced = True
        time.sleep(poll_seconds)


def training_preflight():
    for arm in ('multiform10', 'repeat10'):
        state = read(WORK / f'jobs/{arm}/state.json')
        if state.get('status') != 'complete' or state.get('epoch') != 10.0 or state.get('global_step') != 6250:
            raise RuntimeError(f'{arm} is not a completed 10-epoch run: {state}')
        audit = [json.loads(line) for line in
                 (WORK / f'jobs/{arm}/parameter_audit.jsonl').read_text().splitlines()]
        if not any(record.get('event') == 'step_end' and record.get('step') == 6250 for record in audit):
            raise RuntimeError(f'{arm} has no final parameter update audit')
    for modality in ('text', 'vision'):
        state = read(BASELINE / f'jobs/{modality}/state.json')
        if state.get('status') != 'complete' or state.get('epoch') != 10.0:
            raise RuntimeError(f'Original {modality} baseline is incomplete: {state}')
    for name, checkpoint, _directions, _mode in EXPERIMENTS:
        if not (checkpoint / 'config.json').is_file():
            raise FileNotFoundError(f'{name} checkpoint is missing: {checkpoint}')


def command(checkpoint, directions, mode, output, resume, max_new_tokens, probe_size,
            answer_format='cot-box', write_batch_size=16):
    result = [sys.executable, str(ROOT / 'scripts/run_v7_evaluation.py'),
              '--data-root', str(DATA), '--model', str(MODEL),
              '--directions', *directions, '--answer-format', answer_format,
              '--native-thinking', 'false', '--require-valid-probe',
              '--probe-size', str(probe_size),
              '--max-new-tokens', str(max_new_tokens), '--max-pixels', '1003520',
              '--batch-size', '1', '--write-batch-size', str(write_batch_size), '--seed', '42',
              '--output-dir', str(output)]
    result.extend(['--text-adapter' if mode == 'text' else '--visual-adapter', str(checkpoint)])
    if resume:
        result.append('--resume')
    return result


def summarize():
    reports = []
    for name, _checkpoint, directions, _mode in EXPERIMENTS:
        for direction in directions:
            path = WORK / EVALUATION_NAME / name / direction / 'report.json'
            if path.is_file():
                reports.append({'experiment': name, **read(path)})
    save(WORK / EVALUATION_NAME / 'summary.json', {'updated_at': now(), 'reports': reports})
    with (WORK / EVALUATION_NAME / 'summary.csv.tmp').open('w', newline='') as stream:
        fields = ('experiment', 'direction', 'status', 'total', 'matched', 'valid', 'correct',
                  'accuracy', 'accuracy_on_valid_only', 'empty_prediction', 'format_violation',
                  'missing_prediction')
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reports)
    os.replace(WORK / EVALUATION_NAME / 'summary.csv.tmp', WORK / EVALUATION_NAME / 'summary.csv')
    lines = ['# Rules TEXT augmentation: evaluation', '',
             'Each column uses 2,000 rule questions. Invalid and missing answers remain in the full accuracy denominator.', '',
             '| Learning checkpoint | Direction | Accuracy | Valid | Invalid |',
             '|---|---|---:|---:|---:|']
    for row in reports:
        invalid = (row.get('empty_prediction', 0) + row.get('format_violation', 0) +
                   row.get('missing_prediction', 0))
        lines.append(f"| {row['experiment']} | {row['direction']} | {row.get('accuracy', 0):.2%} | "
                     f"{row.get('valid', 0)}/{row.get('total', 0)} | {invalid} |")
    (WORK / EVALUATION_NAME / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    return reports


def main():
    global WORK, EVALUATION_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--evaluation-name', default=EVALUATION_NAME,
                        help='Separate output subdirectory for a changed inference protocol')
    parser.add_argument('--answer-format', choices=('strict', 'cot-box'), default='cot-box',
                        help='strict: answer letter only; cot-box: brief CoT and terminal box')
    parser.add_argument('--max-new-tokens', type=int, default=6144,
                        help='Maximum generated tokens per answer (default: 6144)')
    parser.add_argument('--write-batch-size', type=int, default=16,
                        help='Flush every N answers; use 1 for immediate per-question saving')
    parser.add_argument('--probe-size', type=int, default=12,
                        help='Initial format probes; halt if all are invalid (default: 12)')
    args = parser.parse_args()
    WORK = args.work.resolve()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if args.probe_size < 1:
        parser.error('--probe-size must be positive')
    if args.write_batch_size < 1:
        parser.error('--write-batch-size must be positive')
    if Path(args.evaluation_name).name != args.evaluation_name or args.evaluation_name in ('.', '..'):
        parser.error('--evaluation-name must be a directory name')
    EVALUATION_NAME = args.evaluation_name
    training_preflight()
    eval_root = WORK / EVALUATION_NAME
    eval_root.mkdir(parents=True, exist_ok=True)
    status_path = eval_root / 'status.json'
    previous = read(status_path) if status_path.exists() else {}
    if previous and (previous.get('answer_format', 'cot-box') != args.answer_format or
                     previous.get('max_new_tokens') != args.max_new_tokens):
        parser.error('Inference protocol changed; use a new --evaluation-name')
    if previous.get('status') == 'complete':
        summarize()
        return
    status = {'status': 'waiting_for_gpu', 'started_at': previous.get('started_at', now()),
              'updated_at': now(), 'experiments': [row[0] for row in EXPERIMENTS],
              'protocol': ('direct answer only: exactly one uppercase A/B/C/D; no explanation'
                           if args.answer_format == 'strict' else
                           'explicit CoT with brief reasoning; do not overthink; boxed answer preferred, unique terminal answer normalized'),
              'answer_format': args.answer_format,
              'max_new_tokens': args.max_new_tokens,
              'write_batch_size': args.write_batch_size,
              'probe_size': args.probe_size,
              'native_thinking': False,
              'gpu_budget_mib': 20000, 'rule_rows_per_direction': 2000,
              'do_not_overlap_existing_compute_processes': True}
    save(status_path, status)
    from v7_gpu_resources import reserve
    wait_for_external_gpu_processes(status, status_path, args.poll_seconds)
    with reserve(LEDGER, '0', 'rules1000_10x_text_evaluation', 20000,
                 poll=args.poll_seconds) as lease:
        lease.bind(os.getpid())
        for name, checkpoint, directions, mode in EXPERIMENTS:
            wait_for_external_gpu_processes(status, status_path, args.poll_seconds)
            output = eval_root / name
            report_paths = [output / direction / 'report.json' for direction in directions]
            if all(path.is_file() and read(path).get('matched') == 2000 for path in report_paths):
                continue
            output.mkdir(parents=True, exist_ok=True)
            resume = (output / 'manifest.json').is_file()
            cmd = command(checkpoint, directions, mode, output, resume,
                          args.max_new_tokens, args.probe_size,
                          args.answer_format, args.write_batch_size)
            status.update(status='running', experiment=name, updated_at=now(), command=cmd)
            save(status_path, status)
            log_path = output / 'controller.log'
            env = {**os.environ, 'PYTHONUNBUFFERED': '1',
                   'PATH': str(ROOT / '.venv/bin') + os.pathsep + os.environ.get('PATH', ''),
                   'HF_HOME': str(ROOT / '.runtime/cache/huggingface'),
                   'HF_HUB_OFFLINE': '1'}
            with log_path.open('a') as log:
                result = subprocess.run(cmd, cwd=ROOT, env=env,
                                        stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            summarize()
            if result.returncode:
                status.update(status='failed', experiment=name, returncode=result.returncode,
                              updated_at=now(), log=str(log_path))
                save(status_path, status)
                raise RuntimeError(f'{name} evaluation failed; inspect {log_path}')
        reports = summarize()
        complete = all(any(r['experiment'] == name and r['direction'] == direction and
                           r.get('matched') == 2000 for r in reports)
                        for name, _checkpoint, directions, _mode in EXPERIMENTS
                        for direction in directions)
        status.update(status='complete' if complete else 'incomplete', updated_at=now(),
                      completed_directions=sum(r.get('matched') == 2000 for r in reports),
                      expected_directions=sum(len(item[2]) for item in EXPERIMENTS))
        save(status_path, status)
        if not complete:
            raise RuntimeError('Evaluation ended without all eight 2,000-row directions complete')


if __name__ == '__main__':
    main()

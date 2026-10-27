"""Shared admission budgets for this experiment's training and evaluation."""
import fcntl
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

# MiB, including allocator/cache headroom. Inference includes image batches.
TRAIN_PEAKS = {'qwen3': {'text': 14000, 'vision': 17000},
               'qwen25': {'text': 22000, 'vision': 27000},
               'glm46': {'text': 31000, 'vision': 36000}}
EVAL_PEAKS = {'qwen3': 20000, 'qwen25': 28000, 'glm46': 38000}
RESERVE_MIB = 4000


def alive(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().split(') ')[1].split()[0] != 'Z'
    except (OSError, ValueError, IndexError, TypeError):
        return False


def parent(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def owned_memory(entries, processes):
    owned = {key: 0 for key in entries}
    roots = {int(e['pid']): key for key, e in entries.items() if e.get('pid')}
    for pid, mib in processes:
        seen = set()
        while pid > 1 and pid not in seen:
            if pid in roots:
                owned[roots[pid]] += mib
                break
            seen.add(pid)
            pid = parent(pid)
    return owned


def available_mib(total, used, entries, owned):
    # Used includes unrelated processes. Reserve the unallocated portion of
    # every admitted job's budget, so simultaneous model loads cannot race.
    return total - RESERVE_MIB - used - sum(
        max(0, e['peak_mib'] - owned.get(key, 0)) for key, e in entries.items())


def snapshot(gpu):
    cmd = ['nvidia-smi', '-i', str(gpu)]
    total, used = map(int, subprocess.check_output(cmd + [
        '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(','))
    output = subprocess.check_output(cmd + ['--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    processes = []
    for line in output.splitlines():
        try:
            processes.append(tuple(map(int, line.split(','))))
        except ValueError:
            continue
    return total, used, processes


@contextmanager
def ledger(work):
    work = Path(work)
    with (work / '.gpu_admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = work / 'gpu_reservations.json'
        entries = json.loads(path.read_text()) if path.exists() else {}
        entries = {k: e for k, e in entries.items()
                   if alive(e.get('pid') or e.get('owner_pid'))}
        yield entries
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(entries, indent=2) + '\n')
        tmp.replace(path)


class Reservation:
    def __init__(self, work, key):
        self.work, self.key = work, key

    def bind(self, pid):
        with ledger(self.work) as entries:
            entries[self.key]['pid'] = pid


def can_admit(work, gpu, peak_mib):
    with ledger(work) as entries:
        same_gpu = {k: e for k, e in entries.items() if e['gpu'] == str(gpu)}
        total, used, processes = snapshot(gpu)
        return available_mib(total, used, same_gpu, owned_memory(same_gpu, processes)) >= peak_mib


@contextmanager
def reserve(work, gpu, key, peak_mib, adopted_pid=None, poll=10):
    work = Path(work)
    while True:
        accepted = False
        with ledger(work) as entries:
            prior = entries.get(key)
            if prior and prior.get('pid') != adopted_pid:
                raise RuntimeError(f'GPU reservation already active: {key}')
            same_gpu = {k: e for k, e in entries.items() if e['gpu'] == str(gpu) and k != key}
            if adopted_pid:
                accepted = True
            else:
                total, used, processes = snapshot(gpu)
                accepted = available_mib(total, used, same_gpu, owned_memory(same_gpu, processes)) >= peak_mib
            if accepted:
                entries[key] = {'gpu': str(gpu), 'peak_mib': peak_mib,
                                'owner_pid': os.getpid(), 'pid': adopted_pid}
        if accepted:
            break
        time.sleep(poll)
    try:
        yield Reservation(work, key)
    finally:
        with ledger(work) as entries:
            entry = entries.get(key, {})
            # If the coordinator exits while its child is alive, keep its
            # budget until a replacement coordinator adopts that process.
            if not alive(entry.get('pid')):
                entries.pop(key, None)

#!/usr/bin/env python3
"""Reversible, journaled workspace archival; never deletes file contents."""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '废弃/20261022_目录整理'
KEEP_RUNS = {
    'v7_full_experiment': '完整v7三模型LoRA实验；同时承载活动GPU资源账本',
    'v7_fullparam_sft_ep10_20261019': '完整1000规则、TEXT/VISION、10epoch全量微调',
    'v7_process_scale64_20261021': '仍在运行：64规则、TEXT学习、三个seed',
    'v7_cot100_20261021': '64规则模型的100题/模态诊断；正式结果在vllm子目录',
    'v7_process_confirmation_20261020': '8规则前置实验；其manifest是活动64规则实验的冻结依赖',
    'v7_process_image_diagnostic_20261021': '活动64规则工作流依赖其完成标记',
    'entity1020_merged1109_qwen3_base_20261020': '当前1109组实体数据的基座评测',
}
DATASETS = ['rules_repaired_single_image_v7', 'entity1020/merged_1109']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot():
    result = {}
    for base in DATASETS:
        p = ROOT / base
        # All dataset images and all active JSON/JSONL/labels are protected.
        files = sorted(x for x in p.rglob('*') if x.is_file() and 'runs' not in x.relative_to(p).parts
                       and ('images' in x.relative_to(p).parts
                            or x.suffix.lower() in {'.json', '.jsonl', '.txt'}))
        result[base] = {str(x.relative_to(p)): sha(x) for x in files}
    return result


def plan():
    moves = []

    def add(src, section, reason, destination=None):
        src = ROOT / src
        if not src.exists():
            return
        dst = ARCHIVE / section / (destination or src.name)
        assert not dst.exists(), dst
        moves.append(dict(source=str(src.relative_to(ROOT)), destination=str(dst.relative_to(ROOT)), reason=reason))

    for p in sorted(ROOT.iterdir()):
        if p.is_dir() and (p.name == 'rules' or p.name.startswith(('rules_corrected', 'rules_heldout', 'rules_repaired'))):
            if p.name != DATASETS[0]:
                add(p.name, '数据集/规则类', '历史规则版本；当前仅使用v7')
    add('entity_v19_1478', '数据集/实体类', '历史实体版本；当前使用merged_1109')
    for p in sorted((ROOT / 'entity1020').iterdir()):
        if p.name not in {'merged_1109', 'CURRENT_DATASET.md'}:
            add(str(p.relative_to(ROOT)), '数据集/实体类/entity1020历史内容', '旧实体数据或配套历史文件')
    for p in sorted(ROOT.glob('*.zip')):
        add(p.name, '旧压缩包', '旧数据打包文件')
    for n in ['packages', 'manifest.json', 'UPLOAD_README.md']:
        add(n, '旧压缩包', '1572组实体上传包及说明；不是当前1109组版本')
    for p in sorted((ROOT / 'runs').iterdir()):
        if p.is_dir() and p.name not in KEEP_RUNS:
            add(str(p.relative_to(ROOT)), '运行结果', '历史实验或辅助诊断；保留全部日志、权重、预测')
    add('rules_repaired_single_image_v7/runs', '运行结果/数据集内历史结果', '旧v7构造验证结果移出活动数据目录', 'rules_v7_runs')
    for n in ['diagnostics', 'rules_sft', '.tmp_options_pilot', '.tmp_rules_learning', '.tmp_rules_pilot']:
        add(n, '历史诊断与试验', '历史诊断、训练派生数据或临时试验')
    for n in ['实验历程与结果总整理.md', '实验完整台账.md', '实验数据证据附录.md', '实验整理证据',
              '规则类实验PPT逐页稿.md', '规则类实验与尝试总整理.md', '规则类实验数据证据附录.md',
              '规则类实验整理证据', '规则类实验汇报.pptx', 'entity_v19_full_experiment_plan.md']:
        add(n, '历史汇报与证据', '保留历史汇报原件；通过归档入口查阅')
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--rollback', action='store_true')
    args = ap.parse_args()
    if args.rollback:
        journal = ARCHIVE / '执行记录.jsonl'
        for line in reversed(journal.read_text().splitlines()):
            row = json.loads(line)
            src, dst = ROOT / row['source'], ROOT / row['destination']
            if dst.exists():
                if src.exists():
                    raise RuntimeError(f'Cannot overwrite existing path: {src}')
                src.parent.mkdir(parents=True, exist_ok=True)
                dst.rename(src)
        print('Rollback complete; generated index documents remain for review.')
        return
    moves = plan()
    print(json.dumps(dict(moves=len(moves), keep_runs=KEEP_RUNS), ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    assert not ARCHIVE.exists(), 'Archive already exists; inspect journal instead of rerunning.'
    # Validate frozen sources before and after; never edit active experiment code/manifests.
    manifest = json.loads((ROOT / 'runs/v7_process_scale64_20261021/manifest.json').read_text())
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected, path
    before = snapshot()
    save(ARCHIVE / '保留数据迁移前SHA256.json', before)
    save(ARCHIVE / '迁移计划.json', dict(created_at=datetime.now().astimezone().isoformat(), moves=moves, keep_runs=KEEP_RUNS))
    with (ARCHIVE / '执行记录.jsonl').open('a') as journal:
        for row in moves:
            src, dst = ROOT / row['source'], ROOT / row['destination']
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Log intent before the atomic rename, so interrupted execution is recoverable.
            journal.write(json.dumps(row, ensure_ascii=False) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            src.rename(dst)
    after = snapshot()
    assert before == after, 'Retained dataset contents changed!'
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected, path
    for row in moves:
        assert not (ROOT / row['source']).exists()
        assert (ROOT / row['destination']).exists()
    save(ARCHIVE / '迁移核验.json', dict(status='pass', moved_entries=len(moves), dataset_hashes_unchanged=True,
         frozen_workflow_sources_unchanged=True, kept_runs=KEEP_RUNS,
         protected_files={base: len(files) for base, files in before.items()}))
    lines = ['# 2026-10-22 目录整理归档', '', '这里只归档，不删除。废弃表示退出当前使用，不表示结果无效。', '',
             '## 迁移清单', '', '| 原路径 | 归档路径 | 原因 |', '|---|---|---|']
    lines += [f"| `{r['source']}` | `{r['destination']}` | {r['reason']} |" for r in moves]
    lines += ['', '## 回滚', '', '先查看迁移计划；停止可能写入相同路径的任务后，可执行：', '',
              '```bash', '.venv/bin/python scripts/organize_workspace_20261022.py --rollback', '```', '',
              '遇到同名新文件会停止，不覆盖。归档文件内部的旧绝对路径保留作历史证据；重跑旧实验需恢复路径或显式指定新路径。']
    (ARCHIVE / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'VERIFIED: {len(moves)} entries moved; dataset hashes and active frozen sources unchanged.', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Print or execute a single-GPU ms-swift full-SFT job inside Slurm."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from common import ARMS, MODALITIES, digest, dump, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--modality', choices=MODALITIES, required=True)
    ap.add_argument('--epochs', type=float, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--swift', default='swift', help='ms-swift CLI executable')
    ap.add_argument('--smoke', action='store_true', help='Two optimizer steps only; separate output required')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--execute', action='store_true', help='Default is command preview; execution requires Slurm')
    args = ap.parse_args()
    root, work, model, output = (p.resolve() for p in (args.root, args.work, args.model, args.output))
    if args.epochs <= 0:
        ap.error('--epochs must be positive')
    if output == root or output.is_relative_to(root):
        ap.error('--output must be outside the immutable data package')
    prepared = json.loads((work / 'prepared.json').read_text())
    if prepared['package_root'] != str(root) or prepared['manifest_sha256'] != digest(root / 'manifest.json'):
        ap.error('Package moved or changed; re-run prepare.py into a new work directory')
    if prepared['portable_sha256sums_sha256'] != digest(root / 'SHA256SUMS'):
        ap.error('Package checksum list changed')
    relative = f'train/{args.arm}/{args.modality}.jsonl'
    dataset = work / 'data' / relative
    if digest(dataset) != prepared['files'][relative]:
        ap.error('Prepared training data changed')
    expected_model = json.loads((root / 'model_identity.json').read_text())
    for name in ('config.json', 'tokenizer_config.json'):
        if digest(model / name) != expected_model['files'][name]:
            ap.error(f'Model identity differs: {name}')
    train_rows = list(rows(dataset))
    if args.smoke:
        # Use the longest targets, rather than accidentally checking only short rows.
        longest = sorted(train_rows, key=lambda r: -sum(len(m['content']) for m in r['messages']))[:4]
        dataset = work / f'smoke_{args.arm}_{args.modality}.jsonl'
        from common import write_rows
        if args.execute:
            write_rows(dataset, longest)
    accumulation = 1 if args.smoke else 16
    cmd = [args.swift, 'sft', '--model', str(model), '--model_type', 'qwen3_vl',
           '--dataset', str(dataset), '--tuner_type', 'full', '--torch_dtype', 'bfloat16',
           '--attn_impl', 'sdpa', '--freeze_vit', str(args.modality == 'text').lower(),
           '--freeze_aligner', str(args.modality == 'text').lower(),
           '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', str(accumulation),
           '--learning_rate', '1e-5', '--num_train_epochs', str(args.epochs),
           '--lr_scheduler_type', 'cosine', '--warmup_ratio', '0' if args.smoke else '0.05',
           '--gradient_checkpointing', 'true', '--vit_gradient_checkpointing', 'true',
           '--max_length', '8192', '--max_pixels', '1003520', '--save_strategy', 'steps' if args.smoke else 'epoch',
           '--save_total_limit', '2', '--save_only_model', 'false', '--split_dataset_ratio', '0',
           '--logging_steps', '1' if args.smoke else '5', '--seed', str(args.seed), '--data_seed', str(args.seed),
           '--add_version', 'false', '--output_dir', str(output), '--use_chat_template', 'true',
           '--loss_scale', 'default', '--strict', 'true', '--enable_thinking', 'false',
           '--report_to', 'none', '--dataset_num_proc', '1', '--dataloader_num_workers', '0',
           '--dataloader_persistent_workers', 'false', '--optim', 'adamw_bnb_8bit',
           '--external_plugins', str(root / 'tools/rules_dense_training_audit.py'),
           '--callbacks', 'rules_dense_training_audit']
    if args.smoke:
        cmd += ['--max_steps', '2', '--save_steps', '1']
    if args.resume:
        resume = args.resume.resolve()
        for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            if not (resume / name).is_file():
                ap.error(f'Incomplete resume checkpoint: {name}')
        cmd += ['--resume_from_checkpoint', str(resume), '--load_args', 'false']
    plan = {'arm': args.arm, 'modality': args.modality, 'rows': len(train_rows),
            'epochs': args.epochs, 'seed': args.seed, 'smoke': args.smoke,
            'sample_presentations_per_rule': None if args.smoke else args.epochs * (1 if args.arm == 'original' else 10),
            'estimated_optimizer_steps_single_gpu': 2 if args.smoke else math.ceil(len(train_rows) / 16) * args.epochs,
            'budget': 'same_epochs', 'command': cmd, 'shell_command': shlex.join(cmd),
            'dataset_sha256': prepared['files'][relative], 'package_manifest_sha256': prepared['manifest_sha256']}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    if not os.environ.get('SLURM_JOB_ID'):
        ap.error('Submit via Slurm; --execute is disabled outside an allocation')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        ap.error('This template uses one GPU; multi-GPU requires an explicit batch/schedule design')
    if not shutil.which(args.swift):
        ap.error('ms-swift executable not found; activate the prepared environment first')
    if output.exists() and any(output.iterdir()) and not args.resume:
        ap.error('Output is nonempty; choose a fresh directory or provide --resume')
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'cluster_run_plan.json', plan)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', MAX_PIXELS='1003520', PYTHONUNBUFFERED='1', NPROC_PER_NODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               RULES_DENSE_AUDIT=str(output / 'parameter_audit.jsonl'), RULES_DENSE_MODALITY=args.modality,
               RULES_DENSE_STOP_AT='2' if args.smoke else '', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    # Keep scheduler-provided CUDA_VISIBLE_DEVICES intact.
    subprocess.run(cmd, env=env, cwd=work, check=True)


if __name__ == '__main__':
    main()

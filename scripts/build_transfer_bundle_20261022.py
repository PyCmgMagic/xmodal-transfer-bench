#!/usr/bin/env python3
"""Build a portable snapshot without touching active experiments.

Dataset files are hard-linked into the staging tree when possible, so the
snapshot itself does not consume a second copy before it is archived.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '迁移包_xmodal_20261022'

def link_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def add_tree(src: Path, rel: str, predicate=None):
    target = OUT / rel
    count = 0
    for p in src.rglob('*'):
        if not p.is_file():
            continue
        if predicate and not predicate(p):
            continue
        link_file(p, target / p.relative_to(src))
        count += 1
    return count

def sha(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def run_file(p):
    # Keep reports, manifests, logs, predictions, and source snapshots; omit
    # model/optimizer weights and tokenizer copies from the portable results.
    if any(x in p.parts for x in ('checkpoint','model','smoke')):
        return False
    return p.stat().st_size <= 200_000_000

def main():
    if OUT.exists():
        raise SystemExit(f'{OUT} already exists; remove only if this snapshot is intentionally rebuilt')
    OUT.mkdir()
    # Latest immutable data only.
    add_tree(ROOT/'rules_repaired_single_image_v7', 'data/rules_repaired_single_image_v7')
    add_tree(ROOT/'entity1020/merged_1109', 'data/entity1020/merged_1109')
    # All current scripts and tests are small and are needed by imported modules.
    add_tree(ROOT/'scripts', 'scripts')
    add_tree(ROOT/'tests', 'tests')
    # Current entry-point documentation and reproducibility records.
    for name in ['README.md','CROSS_MODAL_RESEARCH_PROTOCOL.md','instruct.md','second_question_construction_guide.md',
                 'runs/README.md',
                 'MIGRATION_README.md']:
        src=ROOT/name; link_file(src, OUT/name)
    # Preserve useful result summaries and predictions without transferring
    # multi-gigabyte model/optimizer checkpoints.
    keep_runs = [
        'v7_full_experiment','v7_fullparam_sft_ep10_20261019',
        'v7_cot100_20261021','v7_process_confirmation_20261020',
        'v7_process_image_diagnostic_20261021','v7_process_scale64_20261021',
        'entity1020_merged1109_qwen3_base_20261020']
    for name in keep_runs:
        add_tree(ROOT/'runs'/name, 'runs/'+name, run_file)
    # Keep the original repository-relative entry points.  They are symlinks
    # inside the archive, so the data is stored only once.
    (OUT/'rules_repaired_single_image_v7').symlink_to(Path('data/rules_repaired_single_image_v7'))
    (OUT/'entity1020').symlink_to(Path('data/entity1020'))
    # A compact environment record; caches and virtualenv are intentionally
    # excluded because they contain machine-specific absolute paths.
    req = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=False)
    (OUT/'environment_pip_freeze.txt').write_text(req.stdout)
    try:
        smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=False).stdout
    except FileNotFoundError: smi='nvidia-smi unavailable on source host\n'
    (OUT/'environment_nvidia_smi.txt').write_text(smi)
    manifest={
      'schema':'xmodal-portable-transfer-v1','created_at':'2026-10-22',
      'source_root':str(ROOT),'latest_datasets':{
        'rules':'data/rules_repaired_single_image_v7','entities':'data/entity1020/merged_1109'},
      'included':['data','scripts','tests','current documentation','result summaries and predictions <=200MB'],
      'excluded':['model_cache','huggingface_cache','.venv','local_site_packages','optimizer/model checkpoints','废弃'],
      'rebuild_on_target':['create a new venv and install matching torch/transformers/ms-swift',
                           'download or mount the three base models',
                           'update the qwen25 absolute model path in scripts/run_v7_full_experiment.py',
                           'run the smoke commands in MIGRATION_README.md before full training'],
      'active_source_runs_not_stopped':True}
    files=[]
    for p in OUT.rglob('*'):
        if p.is_file() and p.name!='transfer_manifest.json': files.append(str(p.relative_to(OUT)))
    manifest['file_count_before_manifest']=len(files)
    manifest['sha256']={x:sha(OUT/x) for x in sorted(files)}
    (OUT/'transfer_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'staging':str(OUT),'files':len(files),'size_bytes':sum((OUT/x).stat().st_size for x in files)},ensure_ascii=False))

if __name__=='__main__': main()

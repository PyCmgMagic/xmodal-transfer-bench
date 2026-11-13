#!/usr/bin/env python3
"""Compare saved recall experiments without running models or altering inputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('runs/v7_cot100_20261021')
NEW = Path('runs/qwen3_lora_recall_apply_v2_20261026')
TRAIN = Path('runs/qwen3_lora_learning_ep5_20261024')
OUT = ROOT / 'runs/experiment_summary_20261027'


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    with (ROOT / path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--school-snapshot', type=Path,
                        default=OUT / 'school_readonly_snapshot_20261027.json')
    args = parser.parse_args()
    old_manifest = read(OLD / 'manifest.json')
    new_manifest = read(NEW / 'manifest.json')
    old_summary = read(OLD / 'vllm/summary.json')
    new_summary = read(NEW / 'summary.json')
    old_raw = rows(OLD / 'vllm/trained/predictions.jsonl')
    new_raw = rows(NEW / 'jobs/rules_sft_text/raw.jsonl')
    old_knowledge = {r['id']: r for r in rows(OLD / 'vllm/knowledge.jsonl')}
    knowledge = {r['content']['id']: r['content'] for r in rows(
        Path('data/rules_repaired_single_image_v7/audit/rule_knowledge.jsonl'))}
    selection = {r['qid']: r for r in new_manifest['selection']}
    common = sorted(old_knowledge.keys() & selection.keys())
    old_recalls = {r['qid']: r for r in old_raw if r['protocol'] == 'recall'}
    new_recalls = {r['qid']: r for r in new_raw if r['stage'] == 'recall'}
    old_audit = read(OLD / 'vllm/recall_manual_audit.json')
    old_checks = {r['qid']: r for r in old_audit['checks']}
    old_training = rows(Path('runs/v7_process_scale64_20261021/definitions_training.jsonl'))
    new_training = rows(TRAIN / 'data/rules_sft_text.jsonl')
    old_counts = Counter(r['qid'] for r in old_training)
    new_counts = Counter(next(m['content'] for m in r['messages'] if m['role'] == 'user')
                         for r in new_training)
    old_questions = rows(OLD / 'vllm/questions.jsonl')
    new_questions = {r['id']: r for r in rows(NEW / 'data/T2T.jsonl')}
    common_questions = [r for r in old_questions if r['qid'] in common and r['modality'] == 'T']
    paired_content_matches = all(
        r['messages'][-1]['content'].rsplit('\n\nGive a concise explanation', 1)[0]
        == new_questions[r['source_id']]['question'] for r in common_questions)
    examples = []
    for qid in common:
        before, after = old_knowledge[qid], knowledge[qid]
        examples.append({
            'qid': qid, 'name': selection[qid]['name'],
            'same_learning_text': before['material']['learning']['text'] == after['material']['learning']['text'],
            'same_fact': json.loads(before['fact']['quote']) == json.loads(after['fact']['quote']),
            'reference_fact': json.loads(after['fact']['quote']),
            'old_recall': old_recalls[qid]['response'],
            'current_recall': new_recalls[qid]['response'],
            'old_core_correct_per_saved_manual_audit': old_checks[qid]['core_predicates_and_connective_correct'],
            'current_core_correct_on_manual_review': False,
            'review_reason': '参考规则是条件筛选；当前回答是物品或图形的加权求和，规则类型和条件均不对应。',
        })
    phrase = 'score calculated for one observed collection'
    phrase_families = Counter(selection[q]['family'] for q, r in new_recalls.items()
                             if phrase in r['response'])
    repair = [r for r in read(Path('runs/answer_output_repair_20261026/summary.json'))['cells']
              if r['experiment'] == 'recall_v2']
    paths = [OLD / 'manifest.json', OLD / 'vllm/manifest.json', OLD / 'vllm/summary.json',
             OLD / 'vllm/report_zh.md', OLD / 'vllm/recall_manual_audit.json',
             OLD / 'vllm/trained/predictions.jsonl', OLD / 'vllm/knowledge.jsonl',
             OLD / 'vllm/questions.jsonl', NEW / 'manifest.json', NEW / 'summary.json',
             NEW / 'jobs/rules_sft_text/raw.jsonl', NEW / 'data/T2T.jsonl',
             TRAIN / 'manifest.json', TRAIN / 'data/rules_sft_text.jsonl',
             Path('runs/v7_process_scale64_20261021/definitions_training.jsonl'),
             Path('runs/v7_process_scale64_20261021/seed42/definitions/train.log'),
             Path('runs/answer_output_repair_20261026/summary.json'),
             Path('data/rules_repaired_single_image_v7/audit/rule_knowledge.jsonl')]
    hashes = {str(p): digest(ROOT / p) for p in paths}
    transfer = read(Path('transfer_manifest.json'))
    transferred = {p: value == hashes[p] for p, value in transfer['sha256'].items() if p in hashes}
    school = {'status': 'not_authenticated', 'host': 'school',
              'project_root': '/mnt/yunpan2/tanjiyang/xmodal',
              'note': '本地旧产物与迁移清单一致；尚不能确认远端最新实验。'}
    if args.school_snapshot.is_file():
        school = json.loads(args.school_snapshot.read_text())
        school['local_hash_matches'] = {
            p: digest(ROOT / p) == value for p, value in school['hashes'].items()
            if (ROOT / p).is_file()}
        adapter_path = ('runs/v7_process_scale64_20261021/seed42/definitions/'
                        'checkpoint/checkpoint-3840/adapter_model.safetensors')
        school['adapter_matches_eval_manifest'] = (
            school['hashes'][adapter_path] == old_manifest['adapter']['sha256']['adapter_model.safetensors'])
        assert all(school['local_hash_matches'].values())
        assert school['adapter_matches_eval_manifest']
    audit = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'school': school, 'transfer_snapshot': {
            'created_at': transfer['created_at'], 'source_root': transfer['source_root'],
            'hash_matches': transferred},
        'old_trained_metrics': old_summary['arms']['trained']['metrics'],
        'current_reports': new_summary['reports'],
        'current_answer_repair': repair,
        'training': {
            'old': {'rules': len(old_counts), 'rows': len(old_training), 'epochs': 3,
                    'rows_per_rule_histogram': dict(Counter(old_counts.values())),
                    'channel_counts': dict(Counter(r['channel'] for r in old_training)),
                    'sample_presentations_per_rule': 240, 'rank': 32, 'alpha': 64,
                    'learning_rate_from_train_log': 3e-5, 'checkpoint_step': 3840},
            'current': {'unique_rule_prompts': len(new_counts), 'rows': len(new_training),
                        'epochs': 5, 'rows_per_prompt_histogram': dict(Counter(new_counts.values())),
                        'sample_presentations_per_rule': 5, **read(TRAIN / 'manifest.json')['config']},
            'caveat': '样本呈现数不是完整定义次数或token数；多个训练与推理变量同时变化，不能隔离因果。'},
        'common_rules': examples,
        'common_text_questions': {'n': len(common_questions), 'same_question_and_options': paired_content_matches},
        'current_recall_phrase_audit': {
            'phrase': phrase, 'matching': sum(phrase_families.values()), 'n': len(new_recalls),
            'by_true_family': dict(phrase_families),
            'non_scalar_matches': sum(n for f, n in phrase_families.items() if f != 'rule:scalar_index_rule'),
            'caveat': '字面短语诊断，不是完整语义回忆准确率。'},
        'source_sha256': hashes,
    }
    assert len(common) == 4 and paired_content_matches
    assert all(r['same_learning_text'] and r['same_fact'] for r in examples)
    assert all(r['old_core_correct_per_saved_manual_audit'] for r in examples)
    assert set(old_counts.values()) == {80} and set(new_counts.values()) == {1}
    assert transferred and all(transferred.values())
    assert len(new_recalls) == 90 and sum(phrase_families.values()) == 51
    assert [old_summary['arms']['trained']['metrics'][p]['correct'] for p in
            ['direct_cot_T', 'self_recall_cot_T', 'direct_cot_V', 'self_recall_cot_V']] == [43, 88, 32, 55]
    assert [(r['direct_boxed']['correct'], r['self_recall_boxed']['correct'])
            for r in new_summary['reports']] == [(50, 46), (50, 42), (14, 4)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'recall_school_comparison.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# school 与当前 L40 的回忆实验对比', '',
        '现有证据不支持“同一个实验换服务器后准确率下降”。已保存的两轮使用不同训练检查点、规则范围和推理协议。当前模型存在明确的规则回忆错误，错误文本随后被原样加入答题提示。', '',
        '## 数据来源', '',
        f"旧快照来源：`{transfer['source_root']}`，迁移日期 {transfer['created_at']}。本次核对的 {len(transferred)} 个旧文件与迁移清单哈希全部一致。",
        (f"远端 SSH 已只读核对（{school['checked_at']}）：20个文件已散列；18个本地有副本的文件全部哈希一致。远端adapter权重实测哈希也与旧评测manifest一致。指定目录未发现recall v2等L40新实验目录。远端清单和哈希见配套JSON。" if school.get('status') == 'verified'
         else '远端 SSH 认证尚未成功；以下旧实验结论来自可验证的迁移快照，不能替代远端最新目录核查。'), '',
        '## 成绩口径', '',
        '| 轮次 / TEXT训练模型 | 直接文字 | 回忆后文字 | 直接图片 | 回忆后图片 |',
        '|---|---:|---:|---:|---:|',
        '| 旧 cot100，50规则，每模态100题 | 43/100（43%） | 88/100（88%） | 32/100（32%） | 55/100（55%） |',
        '| 当前 recall v2，90规则，每方向180题 | 50/180（27.78%） | 46/180（25.56%） | 50/180（27.78%） | 42/180（23.33%） |', '',
        '无效回答计错，分母不删除无效样本。当前v2整体暂停，完成3/10方向；本表只比较已完成的TEXT学习模型。VISION→文字是另一模型，原始14/180→4/180，且无效112→164条，追加读取仍有276条待完成。', '',
        '## 为什么不能直接比较', '',
        '| 配置 | 旧 cot100 | 当前 recall v2 |',
        '|---|---|---|',
        '| 训练规则数 | 64 | 1000 |',
        '| 每规则每epoch的训练样本 | 80条：48完整事实、24原子事实、8边界逻辑 | 1条完整定义 |',
        '| epoch / 每规则样本呈现 | 3 / 240 | 5 / 5 |',
        '| LoRA rank / alpha / LR | 32 / 64 / 3e-5 | 8 / 32 / 1e-4 |',
        '| 检查点 | definitions/checkpoint-3840 | rules_sft_text/checkpoint-315（5epoch LoRA） |',
        '| 测试范围 | 4个结构子族，50规则 | 9个规则大类，90规则 |',
        '| 推理引擎 | vLLM，独立BF16合并权重 | ms-swift TransformersEngine，挂载LoRA |',
        '| 回忆 / 答题最大tokens | 768 / 3072 | 1024 / 1536 |',
        '| 答案要求 | FINAL: A/B/C/D | boxed，兼容显式终答解析 |', '',
        '240对5比较的是样本呈现数，不是完整规则重复次数或token数。多个变量同时变化，尚不能将差异单独归因于训练量、rank、引擎或服务器。当前回忆实验也不是最新10epoch全参数模型的评测。', '',
        '## 两阶段设计', '',
        '先只给规则名，让同一检查点自由回忆；回忆时没有正确规则、学习图片、测试题、选项或gold。再将未经校正的自身回忆与原题/原测试图一起输入同一模型。直接答和回忆后答在各自实验内采用相同题目、生成参数和计分方式；不按回忆对错筛样。', '',
        '旧轮回忆提示要求120词内说明精确条件与逻辑，当前要求具体条件、阈值、映射及记录编码。两轮都不是检索正确知识后再答，因此回忆错了会把错误条件带入后续推理。', '',
        '## 同规则证据', '',
        '两轮仅4条规则重合。这4条的参考规则文本和结构事实完全一致，8道文字题的题干和选项也完全一致。旧模型4条核心规则均回忆正确；当前4条均串成无关的加权计分。原文保存在配套JSON，不能将这4条扩展成全部90条的语义准确率。', '',
        '| 规则 | 正确核心规则 / 旧回忆 | 当前回忆 |',
        '|---|---|---|',
        '| Merves | sample pouch AND plastic AND white | circle×0.5 + triangle×1.5 + square×0.25 + 1 |',
        '| Cedlis | sports bag AND plastic AND yellow | storage box×0.5 + sample jar×1.5 + 1 |',
        '| Lerkot | storage bin AND plastic AND yellow | circle×0.5 + triangle×1.5 + square×0.25 + 2 |',
        '| Vedsod | 闭区间 1≤value≤4 筛选 | sample jar×0.5 + sample bottle×1.5 + 1 |', '',
        f"当前TEXT模型90条回忆中，{sum(phrase_families.values())}条含同一计分短语，其中43条真实类型不是scalar_index_rule。这是结构性串规则的线索，不是完整语义正确率。旧人工审计48/50（96%）只覆盖核心条件与连接词，不包含完整记录编码。", '',
        '## 答案解析是否足以解释下降', '',
        '| 当前TEXT模型 / 方向 | 原直接→回忆 | 纯解析后 | 追加约束读取后（额外推理） |',
        '|---|---|---|---|',
        '| T2T | 50/180→46/180 | 50/180→46/180 | 53/180→46/180 |',
        '| T2V | 50/180→42/180 | 50/180→42/180 | 53/180→45/180 |', '',
        '这两个TEXT方向追加读取后无效均为0，回忆后成绩仍低于直接答。格式修复不能反转这轮结果；追加读取是再次给同模型原题及自身回答并约束输出A–D，须与纯解析分开。配对原始结果为T2T修好23题/弄错27题，T2V修好18题/弄错26题，样本较小，不据此认定方法普遍有害。', '',
        '## 核对边界与剩余问题', '',
        ('已确认指定school目录保存的cot100正式结果与迁移副本一致，模型权重身份也一致。未对指定目录之外的位置作全盘搜索。' if school.get('status') == 'verified'
         else '尚待核对school最新目录及远端文件哈希。'), '',
        '若要隔离服务器因素，需要固定同一基座与adapter哈希、同一题集和协议做控制。目前不能从这些异条件结果单独推断硬件影响，也不能把当前小幅下降推广为回忆方法整体失效。此对比仅整理已有产物，没有启动或恢复训练/推理。', '',
        '逐文件来源、完整回忆原文、配置计数及远端核对状态见 [recall_school_comparison.json](recall_school_comparison.json)。', '',
        '重新生成：`python scripts/compare_recall_experiments.py`；成功读取远端后可传 `--school-snapshot <只读快照JSON>`。',
    ]
    (OUT / 'recall_school_comparison_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'common_rules': len(common), 'matched_transferred_files': len(transferred),
                      'school_status': school['status'], 'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

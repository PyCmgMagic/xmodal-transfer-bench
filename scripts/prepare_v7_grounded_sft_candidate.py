#!/usr/bin/env python3
"""Audit absent VISION target facts and prepare a separate explicit-target candidate."""
import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, read, rows, save, sha


def prepare(out):
    data, source = ROOT / 'rules_repaired_single_image_v7', ROOT / 'runs/v7_full_experiment/data'
    knowledge = {r['content']['id']: r['content'] for r in rows(data / 'audit/rule_knowledge.jsonl')}
    vision = rows(source / 'sft_vision.jsonl')
    text = {r['id']: r for r in rows(source / 'sft_text.jsonl')}
    assert len(vision) == len(text) == len(knowledge) == 1000
    inputs = {name: sha(source / name) for name in ('sft_vision.jsonl', 'sft_text.jsonl')}
    candidate, missing, counts = [], [], Counter()
    for old in vision:
        qid = old['id']
        new = copy.deepcopy(old)
        explicit = text[qid]['messages'][-1]['content']
        assert explicit == knowledge[qid]['material']['learning']['text']
        new['messages'][-1]['content'] = explicit
        assert new['messages'][:-1] == old['messages'][:-1] and new['images'] == old['images']
        candidate.append(new)
        k = knowledge[qid]
        if k['fact']['relation'] != 'binary_predicate':
            continue
        for c in json.loads(k['fact']['quote'])['conditions']:
            if not isinstance(c['value'], str):
                continue
            pattern = r'(?<!\w)' + re.escape(c['value']) + r'(?!\w)'
            if re.search(pattern, explicit, re.I):
                counts[c['field'] + '_explicit_in_text'] += 1
                if not re.search(pattern, old['messages'][-1]['content'], re.I):
                    counts[c['field'] + '_absent_in_vision_target'] += 1
                    missing.append({'id': qid, **c})
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'manifest.json').exists(), 'Use a fresh candidate directory'
    ev.save_rows(out / 'sft_vision_explicit.jsonl', candidate)
    save(out / 'target_audit.json', {'scope': 'Literal string coverage in binary-predicate assistant targets; not a complete semantic/image audit.',
        'binary_rules': 104, 'rules_with_missing_literal_values': len({r['id'] for r in missing}),
        'missing_literal_values': len(missing), 'counts': dict(counts), 'details': missing})
    old_diagnostic = rows(ROOT / 'runs/v7_learning_diagnostic_20261014_v2/raw.jsonl')
    conditions = ['base_application_blind', 'base_application_material', 'adapter_application_blind', 'adapter_application_material']
    by_relation = {}
    for relation in sorted({k['fact']['relation'] for k in knowledge.values()}):
        group = [r for r in old_diagnostic if knowledge[r['qid']]['fact']['relation'] == relation]
        by_relation[relation] = {c: {'n': sum(r['condition'] == c for r in group),
                                   'correct': sum(r['correct'] for r in group if r['condition'] == c)} for c in conditions}
    save(out / 'existing_diagnostic_by_relation.json', by_relation)
    images = rows(data / 'image_hashes.jsonl')
    assert len(images) == 3000
    for item in images:
        assert item['source_sha256'] == item['repaired_sha256'] == sha(data / item['path'])
    for filename, digest in inputs.items():
        assert sha(source / filename) == digest
    save(out / 'manifest.json', {'status': 'prepared_not_trained_or_validated', 'rows': len(candidate),
        'candidate_sha256': sha(out / 'sft_vision_explicit.jsonl'), 'original_inputs_sha256': inputs,
        'knowledge_sha256': sha(data / 'audit/rule_knowledge.jsonl'), 'script_sha256': sha(Path(__file__)),
        'change': 'Only SFT VISION assistant targets become the corresponding explicit learning text; all user/image inputs identical.',
        'training_label_source': 'Existing learning material only; no evaluation questions or labels used to create training targets.',
        'original_dataset_modified': False, 'images_verified': 3000,
        'limits': ['Not a validated repair; no checkpoint or accuracy claim.',
                   'Explicit targets supply visual grounding labels; they do not solve TEXT recall/application failures.',
                   'Only two original reference images were visually spot-checked this turn; not a full image-content audit.',
                   'PT raw-material control is not changed.'], 'created_at': ev.now()})
    lines = ['# 训练无提升：监督目标核查与候选修复', '',
        '当前证据不足以认为数据已修好。此次只准备一个VISION SFT候选，没有重新训练，也不覆盖原v7或历史实验。', '',
        '## 新确认的监督缺口', '',
        f"在104条二值规则中，{len({r['id'] for r in missing})}条的VISION答案至少缺少一个TEXT答案里明确出现的具体条件值，共{len(missing)}项：物体类型83项、颜色37项、形状21项、纹理6项。这里是字面覆盖检查，不能证明模型完全不能从图中学习。",
        '例如Nornat的原目标只要求与参考图同类型、同颜色，没有输出tray、blue。抽看原图确为蓝色托盘；Cemsid的两行优先级图片也与已有文字材料的顺序对应。两例抽查不能替代1000图的完整核验。',
        '这种目标允许模型复述“按图比较”的说明，而不必准确说出图中的具体规则，因此缺少直接检验视觉内容提取的监督。它是需要验证的设计缺口，尚未证明是VISION低分的唯一原因，也不能解释TEXT低分。', '',
        '候选文件sft_vision_explicit.jsonl共1000条，图片路径和user提示逐项不变，仅将assistant目标换为同一规则的明确文字知识。标签来自已有学习材料，没有读取评测题生成训练答案。PT原始材料对照保持原定义。', '',
        '## 既有72题诊断按九类拆分', '',
        '| 规则类型 | 基础模型不给材料 | 基础模型给材料 | 原SFT不给材料 | 原SFT给材料 |',
        '|---|---|---|---|---|---|']
    for relation, group in by_relation.items():
        values = [group[c] for c in conditions]
        lines.append('| ' + relation + ' | ' + ' | '.join(f"{v['correct']}/{v['n']}" for v in values) + ' |')
    lines += ['', '每类仅4条规则、8道题，是开发诊断而非独立测试；不能用这个样本给九类做确定排名。此前多轮增强训练只试了binary_predicate，不能据此判断其余896条的学习机制。', '',
        '## 接下来的验证顺序', '',
        '1. 先从给材料能够解答的子类做训练验证，例如集合筛选、比例分档。本样本中二者给材料均8/8；先扩大独立开发样本确认，不根据旧评测题制作训练样本。',
        '2. 将训练监督拆成具体事实/图中规则复述、单步应用、原任务形式的选择题。新观察由规则结构独立生成，训练和验证按观察组合及模板分开。先证明训练集内掌握，再检验迁移；固定训练预算下比较方案，不把增加重复暴露冒充单一因素。',
        '3. VISION用原参考图配明确事实答案，先核验图片和目标对应；TEXT单独检查名称到事实的记忆。各自通过后再测跨模态方向。',
        '4. 保留SFT与原始材料PT的不同定义；修复GLM格式统计，但分别报告严格口径和补充解析。增大LoRA、延长训练都作为后续可检验变量，不声称必然有效。', '',
        '该候选不提供准确率保证，未运行新训练。没有更改原图片、原题、标签或历史实验。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'candidate_rows': len(candidate), 'missing_values': len(missing),
                      'rules_with_missing_values': len({r['id'] for r in missing}),
                      'status': 'prepared_not_validated', 'report': str(out / 'report_zh.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args().out.resolve())

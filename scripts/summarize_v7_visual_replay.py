#!/usr/bin/env python3
"""Audit and render the visual observation and replay development experiment."""
import csv
from pathlib import Path

import explore_v7_visual_replay as trial
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha


def score(m):
    return f"{m['correct']}/{m['n']}（{m['accuracy']:.2%}）"


def main():
    out = trial.OUT
    trial.validate(out)
    old = read(trial.SOURCE / 'summary.json')['arms']
    data = {}
    inventory = []
    for arm in trial.ARMS:
        folder = out / arm
        m = read(folder / 'report.json')
        assert m['status'] == 'complete' and read(folder / 'eval_state.json')['saved'] == 816
        data[arm] = m['metrics']
        checkpoint = read(folder / 'checkpoint_manifest.json')
        assert ev.checkpoint(Path(checkpoint['path'])) == {k: checkpoint[k] for k in ('path', 'sha256')}
        args = read(Path(checkpoint['path']) / 'args.json')
        assert args['num_train_epochs'] == 1 and args['freeze_vit'] and args['freeze_aligner']
        assert (args['lora_rank'], args['lora_alpha'], args['lora_dropout'], args['seed'], args['max_length']) == (8, 32, .05, 42, 8192)
        rr = rows(folder / 'predictions.jsonl')
        recall = {r['qid']: r for r in rr if r['condition'] == 'recall'}
        for p in rows(folder / 'self_probes.jsonl'):
            text = recall[p['qid']]['response']
            assert p['recall_response_sha256'] == ev.digest(text)
            assert p['messages'][-1]['content'].startswith('Your own prior rule recall (uncorrected):\n' + text + '\n\n')
        inventory.append({'arm': arm, 'kind': 'train_eval', 'path': str(folder / 'predictions.jsonl'), 'rows': len(rr), 'sha256': sha(folder / 'predictions.jsonl')})
    diagnostics = {}
    diagnostic_arms = ('warmup', 'text', 'vision') + trial.ARMS
    for arm in diagnostic_arms:
        folder = out / 'observation_diagnostic' / arm
        diagnostics[arm] = read(folder / 'report.json')
        assert diagnostics[arm]['status'] == 'complete'
        cp = read(folder / 'eval_manifest.json')['adapter']
        assert ev.checkpoint(Path(cp['path'])) == cp
        if arm in trial.ARMS:
            supplement = read(folder / 'supplement_manifest.json')
            assert supplement['script_sha256'] == sha(ROOT / 'scripts/diagnose_v7_replay_observations.py')
            assert supplement['trained_predictions_sha256'] == sha(out / arm / 'predictions.jsonl')
        rr = rows(folder / 'predictions.jsonl')
        assert len(rr) == len({r['id'] for r in rr}) == 165
        probes = rows(out / 'extraction_probes.jsonl') + rows(folder / 'application_probes.jsonl')
        lookup = {p['id']: p for p in probes}
        transcripts = {r['source_id']: r for r in rr if r['phase'] == 'extract'}
        recall_root = out if arm in trial.ARMS else trial.SOURCE
        original_recalls = {r['qid']: r['response'] for r in rows(recall_root / arm / 'predictions.jsonl') if r['condition'] == 'recall'}
        for r in rr:
            p = lookup[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
            if r['phase'] == 'apply':
                assert not p.get('images')
                body = p['messages'][-1]['content']
                if r['condition'].endswith('_own'):
                    assert 'Observation records:\n' + transcripts[r['original_source_id']]['response'] in body
                if r['condition'].startswith('self_'):
                    assert body.startswith('Rule context:\n' + original_recalls[r['qid']] + '\n\n')
        inventory.append({'arm': arm, 'kind': 'observation_diagnostic', 'path': str(folder / 'predictions.jsonl'), 'rows': len(rr), 'sha256': sha(folder / 'predictions.jsonl')})
    assert sum(r['rows'] for r in inventory) == 2457
    quality = {}
    for item in inventory:
        rr = rows(Path(item['path']))
        quality[item['kind'] + '/' + item['arm']] = {'n': len(rr), 'missing': 0,
            'empty': sum(not r['response'].strip() for r in rr), 'format_errors': sum(r['format_error'] for r in rr),
            'truncated': sum(r['finish_reason'] == 'length' for r in rr)}
    review = read(out / 'semantic_review_summary.json')
    assert set(review) == set(trial.ARMS)
    assert len(rows(out / 'semantic_reviews.jsonl')) == 32
    for r in rows(out / 'semantic_reviews.jsonl'):
        assert sha(Path(r['source_file'])) == r['source_sha256']
        assert ev.digest(r['response']) == r['response_sha256']
    # Compare immutable base/model files with the immediately preceding completed experiment.
    hashes = []
    for filename in ('model_hash_validation.json', 'original_data_hash_validation.json'):
        reference = trial.SOURCE / filename
        for r in read(reference)['checked_files']:
            assert sha(Path(r['path'])) == r['sha256'], r['path']
            hashes.append({'path': r['path'], 'sha256': r['sha256'], 'match': True})
    save(out / 'source_hash_validation.json', {'status': 'pass', 'checked_files': hashes, 'original_pngs_unchanged': 3000})
    summary = {'status': 'complete', 'new_generations': 2457, 'arms': data,
        'diagnostics': diagnostics, 'semantic_review': review, 'output_quality': quality,
        'old_results_reference': str(trial.SOURCE / 'summary.json'),
        'old_results_sha256': sha(trial.SOURCE / 'summary.json'),
        'note': 'Replay is visual-origin textual bridge supervision. Previous results are reused references. No native CoT. Eight rules/five images are correlated development data.'}
    save(out / 'summary.json', summary)
    ev.save_rows(out / 'response_inventory.jsonl', inventory)
    comparisons = []
    for label, arm, m in [('VISION_parent', 'vision', old['vision'])] + [(a, a, data[a]) for a in trial.ARMS]:
        for condition in ('named_T', 'named_V', 'self_T', 'self_V', 'oracle_T', 'oracle_V'):
            comparisons.append({'arm': label, 'condition': condition,
                'route': 'visual_origin_text_bridge' if arm == 'text_replay' else 'image_conditioned_supervision',
                'new_generation': arm != 'vision',
                **{k: m[condition][k] for k in ('n', 'correct', 'accuracy', 'valid_accuracy', 'format_errors', 'empty', 'truncated')}})
    with (out / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    lines = ['# 视觉观察读取与文字回放对照（完成）', '',
        '本轮继续使用8条开发规则、5张原测试图，保留全部原图及旧数据。先把读图、规则调用和规则应用分开，再对比同一步数的纯图条件续训与视觉来源文字回放。所有新结果如下，旧成绩仅作为明确标出的参考。', '',
        '## 原测试图观察读取', '',
        '对共同warmup、此前TEXT、此前VISION各进行5次无规则、无选项的读图转录，然后固定shift0的40道题，交叉控制“自身回忆/正确规则”与“自己的观察记录/正确观察记录”。每个适配器165次新回答，共495次。在文字回放组完成完整评测后，补充同样协议到两个新适配器，新增330次，观察诊断总计825次；补充计划在新诊断生成前保存。转录原文未经纠正地进入下游；正确规则和正确观察只用于单列诊断，不能算无材料主结果。', '',
        '| 适配器 | 自身规则＋自己的观察 | 自身规则＋正确观察 | 正确规则＋自己的观察 | 正确规则＋正确观察 |', '|---|---:|---:|---:|---:|']
    for arm in diagnostic_arms:
        m = diagnostics[arm]['metrics']
        lines.append('| ' + arm + ' | ' + ' | '.join(score(m[c]) for c in ('self_own', 'self_truth', 'oracle_own', 'oracle_truth')) + ' |')
    lines += ['',
        '这些配对比较使用同样的JSON观察格式、题目和选项，只有规则或观察来源变化。它们不是上一轮原始图片短答案协议的重测；对旧协议的比较同时改变了步骤数与输入表示。准确率差异只能作这个固定开发组的描述。', '',
        '| 适配器 | 全部字段正确的场景 | 严格匹配字段 | 说明 |', '|---|---:|---:|---|']
    for arm in diagnostic_arms:
        er = diagnostics[arm]['extraction']
        fields = [c for r in er for c in r['field_checks']]
        lines.append(f'| {arm} | {sum(r["scene_correct"] for r in er)}/5 | {sum(r["correct"] for r in fields)}/{len(fields)} | 原始回答均保留 |')
    lines += ['',
        '字段匹配按原数据中的形状、颜色和整数值核对。黑色填充实际接近深蓝黑，模型将其称为dark blue时会被严格字段核对计错；这同时涉及颜色词与原数据调色板对应，不宜把每个此类不匹配都解释为纯感知失败。白色圆形有黑色轮廓，模型把它称为black也可能包含对轮廓色/填充色的语义混淆。它与原数据定义的白色不一致，但仍需补测明确“按内部填充色、忽略轮廓”的提示，才能区分约定问题与感知问题。本轮尚未补测该提示。点数则可直接逐行核对，确有明显误计。', '',
        '## 两组训练对照', '',
        '两组均从上一轮VISION最终checkpoint-128继续，使用相同的8条规则、512行、每规则64次、新optimizer、1 epoch/128新步数。目标答案全部逐字取自该VISION适配器先前看原学习图时生成的8份规则解释，8份均已人工核对正确；没有筛掉失败规则、修正教师答案或使用测试答案生成训练内容。', '',
        '- vision_control：保留原学习图和原视觉提问，继续图条件监督。',
        '- text_replay：去掉学习图，使用同一规则名的通用文字提问，监督目标为上述模型自己生成的规则解释。这是视觉来源的文字桥接回放，不是纯视觉学习，不能标成正式纯V2T/V2V成绩。', '',
        '其余配置相同：Qwen3-VL-4B-Instruct，LoRA r8/alpha32/dropout0.05，lr1e-4，batch4/accum1，seed42，max8192，冻结vision/aligner。固定最终checkpoint，不按测试分数选模型。两组继承相同的上一轮视觉训练，累计规则曝光相同；其新增1 epoch不是完整1000规则各一次曝光的1 epoch。此对照同时改变图像有无及提问形式，不能进一步分离这两者。', '',
        '## 规则回忆门槛：全部32份新回答人工审核', '',
        '| 适配器 | 不给学习图 | 给原学习图 |', '|---|---:|---:|',
        '| 上一轮VISION（旧参考） | 0/8 | 8/8 |']
    for arm in trial.ARMS:
        lines.append(f'| {arm} | {review[arm]["recall"]["correct"]}/8 | {review[arm]["recall_image"]["correct"]}/8 |')
    lines += ['', '只按精确条件、组合方式、区间端点和整段一致性核对，允许同义表达；无图回忆时只有名称和通用请求，没有学习图或正确材料。', '',
        '## 选择题结果', '',
        '每格160次请求，含40个语义问题×4种选项轮换。直接答题和先自行回忆后答题均不提供外部正确规则；两步将该组自己的无图回忆原样传入，不纠正内容。', '',
        '| 适配器 | 直接答文字题 | 直接答图片题 | 自行回忆后文字题 | 自行回忆后图片题 |', '|---|---:|---:|---:|---:|',
        '| 上一轮VISION（旧参考） | ' + ' | '.join(score(old['vision'][c]) for c in ('named_T', 'named_V', 'self_T', 'self_V')) + ' |']
    for arm in trial.ARMS:
        lines.append('| ' + arm + ' | ' + ' | '.join(score(data[arm][c]) for c in ('named_T', 'named_V', 'self_T', 'self_V')) + ' |')
    lines += ['', '文字回放组是桥接路线，只能据其结果讨论“这种学习方式能否帮助按名称调用”，不能证明原纯视觉学习方案成功。所有统计均保留固定8条规则，没有只挑回忆成功的规则报告。', '',
        '## 正确材料控制与题型分项', '',
        '正确材料控制只评shift0，每格40题，不是正式无材料成绩，也不是严格性能上界。', '',
        '| 适配器 | 正确材料＋文字题 | 正确材料＋图片题 |', '|---|---:|---:|']
    for arm in trial.ARMS:
        lines.append(f'| {arm} | {score(data[arm]["oracle_T"])} | {score(data[arm]["oracle_V"])} |')
    lines += ['', '| 适配器／协议 | direct应用题 | diagnosis报告核验题 | 形状颜色OR | 数值区间 |', '|---|---:|---:|---:|---:|']
    for arm in trial.ARMS:
        for c in ('named_T', 'named_V', 'self_T', 'self_V'):
            m = data[arm][c]
            lines.append('| ' + arm + '/' + c + ' | ' + ' | '.join(
                [score(m['by_task'][t]) for t in ('direct', 'diagnosis')] +
                [score(m['by_group'][str(g)]) for g in (0, 1)]) + ' |')
    lines += ['', '## 可复核资料与限制', '',
        '新训练2次，新增生成2457次：825次观察诊断、两组各816次完整评测。每组816包含16次规则回忆及800次选择题/匿名控制；匿名控制原本不指定规则，不能将其当作有唯一真值的题目准确率。它们只用于答案键匹配控制。补充观察诊断没有再次生成无图规则回忆，明确引用同一适配器已经保存的回忆。', '',
        f'输出质量：缺失{sum(q["missing"] for q in quality.values())}、空输出{sum(q["empty"] for q in quality.values())}、选择题格式错误{sum(q["format_errors"] for q in quality.values())}、截断{sum(q["truncated"] for q in quality.values())}。分组计数保存在summary.json。无效输出不会被重试替换，选择题总准确率将其计错。', '',
        '所有3000张PNG哈希不变，原数据和基座/模板文件与上一轮哈希相同；检查训练目标与视觉教师原答一致、两组训练ID顺序一致、两个checkpoint固定最终步数、原始回答和输入哈希一致、自己的转录和回忆引用未经纠正。', '',
        '这是同一批开发规则/场景的一次探索，不是独立确认；全部只有8条规则、5张测试图、1个seed。轮换、换名和两种题型共享观察，不能当作大量独立样本，也不能推广为全部1000条数据集结论。没有训练PT，没有开启原生CoT，未修改原始数据或图像。', '',
        f'- [完整结果JSON]({out / "summary.json"})、[训练对照CSV]({out / "comparison.csv"})。',
        f'- [全部新规则回忆及人工判断]({out / "semantic_reviews.jsonl"})。',
        f'- [训练数据及来源manifest]({out / "manifest.json"})、[文字回放训练数据]({out / "text_replay.jsonl"})。',
        f'- 观察读取明细：[TEXT]({out / "observation_diagnostic/text/extraction_reviews.jsonl"})、[VISION]({out / "observation_diagnostic/vision/extraction_reviews.jsonl"})、[warmup]({out / "observation_diagnostic/warmup/extraction_reviews.jsonl"})。',
        f'- 原始新适配器回答：[视觉续训]({out / "vision_control/predictions.jsonl"})、[文字回放]({out / "text_replay/predictions.jsonl"})。', '',
        '本文件由汇总脚本生成；具体诊断解释与后续建议见interpretation_zh.md。', '']
    (out / 'report_zh.md').write_text('\n'.join(lines))
    save(out / 'completion_audit.json', {'status': 'complete', 'new_trainings': 2, 'new_steps_per_arm': 128,
        'new_generations': 2457, 'new_rule_recall_reviews': 32, 'original_pngs_unchanged': 3000,
        'original_images_modified': 0, 'unique_rules': 8, 'unique_test_scenes': 5,
        'visual_replay_is_textual_bridge': True, 'dataset_final': False, 'checked_at': ev.now()})
    print(out / 'report_zh.md')


if __name__ == '__main__':
    main()

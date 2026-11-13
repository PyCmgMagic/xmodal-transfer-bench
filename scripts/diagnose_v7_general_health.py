#!/usr/bin/env python3
"""Paired everyday/reasoning smoke check of the final five-epoch v7 adapter."""
import argparse
import copy
import dataclasses
import json
import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, read, save, sha
from v7_gpu_resources import reserve


SYSTEM = (
    '你是一名乐于助人的助手。需要计算或推理时，请用简短、可核查的步骤解释，再给出明确答案。'
    '日常问题请给出自然、实用的回答。通常不超过八句话。'
    '如果用户要求特定输出格式，请优先满足该格式。'
)


def probes():
    specs = [
        ('rest', '日常建议', '我今天有点累，只想花十分钟放松一下，你有什么简单建议？',
         '自然、可行的十分钟放松建议；不串入训练规则。', 'qualitative'),
        ('meal', '日常建议', '家里只有鸡蛋、西红柿和熟米饭，想在二十分钟内吃上饭，怎么做比较方便？',
         '围绕现有食材给出可行做法；注意米饭为熟饭。', 'qualitative'),
        ('umbrella', '日常建议', '天气预报说今天下午下雨的概率是80%，我准备步行出门半小时，要不要带伞？请说一下理由。',
         '建议带伞或轻便雨具；不把80%说成必然下雨。', 'qualitative'),
        ('phishing', '日常常识', '陌生短信说我中奖了，要求点击链接并输入银行卡密码才能领奖。我该怎么办？',
         '不要点击/不要提供密码，识别诈骗，通过官方渠道核实。', 'objective'),
        ('multiply', '算术', '17乘以23等于多少？请说明简短计算步骤。', '391', 'objective'),
        ('decimal', '数值比较', '9.9和9.11哪个更大？请解释一下。', '9.9更大，因为9.90 > 9.11。', 'objective'),
        ('discount', '算术', '一件外套原价200元，先打八折，再立减20元。最后需要付多少钱？请说明计算。', '140元', 'objective'),
        ('time', '时间计算', '一场电影下午2点35分开始，片长1小时50分钟，几点结束？请简要计算。', '下午4点25分，即16:25。', 'objective'),
        ('rectangle', '几何', '一个长方形长8厘米、宽5厘米，它的周长和面积分别是多少？请解释计算。', '周长26厘米，面积40平方厘米。', 'objective'),
        ('logic', '简单逻辑', '已知所有玫瑰都是花，而且有些花是红色的。能据此确定所有玫瑰都是红色的吗？请解释。', '不能，前提没有限定玫瑰都属于红花。', 'objective'),
        ('probability', '概率', '均匀的六面骰子掷一次，点数为偶数的概率是多少？请简要说明。', '3/6=1/2，偶数为2、4、6。', 'objective'),
        ('science', '基础科学', '白天和夜晚交替，主要是因为地球自转还是绕太阳公转？请用简单的话解释。', '地球自转；朝向太阳的一面为白天。', 'objective'),
        ('reading', '阅读理解', '小林原来有3支笔，送给小王1支；小王原来有2支笔。现在两个人各有几支笔？请说明。', '小林2支，小王3支。', 'objective'),
        ('translate', '翻译', '请把这句话翻译成自然的中文：I left my umbrella at home, so I got wet on my way to work.',
         '我把伞落在家里了，所以在上班路上淋湿了；含义完整。', 'objective'),
        ('json', '格式遵循', '只输出合法JSON，不要Markdown或解释。信息：姓名小王，年龄20岁，是学生。键名使用name、age、is_student，年龄用数字，是否学生用布尔值。',
         '{"name":"小王","age":20,"is_student":true}，无额外文字。', 'objective'),
        ('python', '简单代码', 'Python表达式list(range(1, 6, 2))的结果是什么？请简要解释。', '[1, 3, 5]，右端点6不包含。', 'objective'),
        ('fraction', '数值比较', '请从小到大排列3/4、0.8和80%，并说明是否有相等的数。', '3/4 < 0.8 = 80%。', 'objective'),
        ('english', '英文问答', 'Mia has 12 apples. She gives away 4 and then buys 3 more. How many apples does she have now? Briefly explain in English.',
         '11 apples; 12 - 4 + 3 = 11. Answer in English.', 'objective'),
        ('soap', '日常解释', '请用两句话向小学生解释：为什么洗手时用肥皂通常比只用清水更有效？',
         '两句话，通俗解释肥皂帮助去除油污和附着的微生物，配合冲洗；不夸大为全部杀灭。', 'qualitative'),
    ]
    items = []
    for qid, category, question, reference, kind in specs:
        items.append({'id': qid, 'category': category, 'question': question, 'reference': reference,
                      'review_kind': kind, 'messages': [{'role':'system','content':SYSTEM}, {'role':'user','content':question}]})
    items.append({'id':'context', 'category':'对话记忆', 'question':'我的猫叫什么？请用一句话回答。',
                  'reference':'团团。', 'review_kind':'objective',
                  'messages':[{'role':'system','content':SYSTEM},
                              {'role':'user','content':'我叫小周，我的猫叫团团。'},
                              {'role':'assistant','content':'你好，小周！你的猫叫团团。'},
                              {'role':'user','content':'我的猫叫什么？请用一句话回答。'}]})
    return items


def numeric_followup_probes():
    original = next(p for p in probes() if p['id']=='fraction')
    specs = [
        ('decimal_percent', '0.8和80%是否相等？请先把百分数转为小数，再给出结论。', '相等，80%=0.8。'),
        ('three_equal', '比较3/4、0.75和75%：哪些相等？请给出理由。', '三者全部相等。'),
        ('half_equal', '将0.5、1/2和50%从小到大排列；如果有相等的数，请明确写出。', '三者全部相等。'),
        ('judge_equality', '判断这个说法是否正确：80%=0.8，因此它们大小相同。请说明。', '正确，百分数转换结果相同。'),
    ]
    items = [{'id':'fraction_repeat','category':'原题单请求复测','question':original['question'],
              'reference':original['reference'],'review_kind':'objective','messages':copy.deepcopy(original['messages'])}]
    items.extend({'id':qid,'category':'数值等价换问法','question':q,'reference':gold,'review_kind':'objective',
                  'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':q}]} for qid,q,gold in specs)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--suite', choices=['general','numeric_followup'], default='general')
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT/'runs/v7_full_experiment'
    data = ROOT/'rules_repaired_single_image_v7'
    checkpoint_info = read(work/'jobs/qwen3_sft_text/checkpoint_manifest.json')
    cp = Path(checkpoint_info['path'])
    trainer = read(cp/'trainer_state.json')
    assert trainer['epoch'] == 5.0 and trainer['global_step'] == 315
    assert sha(cp/'adapter_model.safetensors') == checkpoint_info['sha256']['adapter_model.safetensors']
    items = probes() if a.suite=='general' else numeric_followup_probes()
    batch_size = 4 if a.suite=='general' else 1
    ev.save_rows(out/'probes.jsonl', items)
    config = {'model': str(MODELS['qwen3']), 'adapter':ev.checkpoint(cp), 'epochs':5,
              'global_step':315, 'conditions':['base','adapter'], 'questions':len(items),
              'generation':{'temperature':0,'max_tokens':768,'seed':42,'max_batch_size':batch_size},
              'reasoning_mode':'prompted_explanation', 'native_thinking':False,
              'model_variant':'Qwen3-VL-4B-Instruct, not a native Thinking checkpoint',
              'external_learning_material':False,'images':False,
              'scope':'Small illustrative capability check, not a general capability benchmark.',
              'prompt_sha256':ev.digest(items),'script_sha256':sha(Path(__file__))}
    manifest = out/'manifest.json'
    if manifest.exists():
        assert read(manifest)['config'] == config, 'Cannot resume changed diagnostic'
    else:
        save(manifest, {'config':config, 'created_at':ev.now()})
    raw = out/'predictions.jsonl'
    records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, records)
    seen = {(r['condition'],r['id']) for r in records}
    save(out/'status.json', {'status':'loading_model','pid':os.getpid(),'gpu':a.gpu,'saved':len(records),'updated_at':ev.now()})
    with reserve(work,a.gpu,str(out)+':general_health',22000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(environment(a.gpu,data))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,
                                    device_map='cuda:0',max_batch_size=batch_size,attn_impl='sdpa')
        engine.template.enable_thinking = False
        tensors = [(n,p) for n,p in engine.model.named_parameters() if 'lora_B' in n]
        verification = {'active_adapters':str(getattr(engine.model,'active_adapters',None)),
                        'lora_B_tensors':len(tensors),
                        'nonzero_lora_B_tensors':sum(bool(torch.count_nonzero(p)) for _,p in tensors),
                        'base_control':'same loaded weights with model.disable_adapter()',
                        'template_enable_thinking':engine.template.enable_thinking}
        assert verification['nonzero_lora_B_tensors'] > 0
        save(out/'adapter_verification.json',verification)
        for condition in ('base','adapter'):
            pending = [p for p in items if (condition,p['id']) not in seen]
            with engine.model.disable_adapter() if condition=='base' else nullcontext():
                for start in range(0,len(pending),batch_size):
                    batch = pending[start:start+batch_size]
                    results = engine.infer([InferRequest(messages=copy.deepcopy(p['messages'])) for p in batch],
                                           RequestConfig(max_tokens=768,temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(results)==len(batch)
                    for p,result in zip(batch,results):
                        if isinstance(result,Exception):
                            raise result
                        choice = result.choices[0]
                        record = {k:p[k] for k in ('id','category','question','reference','review_kind')}
                        record.update(condition=condition,response=choice.message.content or '',
                                      reasoning_content=getattr(choice.message,'reasoning_content',None),
                                      finish_reason=choice.finish_reason,usage=dataclasses.asdict(result.usage),
                                      prompt_sha256=ev.digest(p['messages']),saved_at=ev.now(),
                                      raw_response=dataclasses.asdict(result))
                        with raw.open('a') as f:
                            f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
                            f.flush()
                            os.fsync(f.fileno())
                        records.append(record)
                    save(out/'status.json',{'status':'running','condition':condition,'saved':len(records),
                                           'expected':2*len(items),'updated_at':ev.now()})
                    print(condition, len([r for r in records if r['condition']==condition]), '/', len(items), flush=True)
    counts = Counter(r['condition'] for r in records)
    assert counts == {'base':len(items),'adapter':len(items)}
    assert len(seen | {(r['condition'],r['id']) for r in records}) == 2*len(items)
    save(out/'generation_report.json',{'status':'complete','conditions':dict(counts),
        'empty':sum(not r['response'].strip() for r in records),
        'truncated':sum(r['finish_reason']=='length' for r in records),
        'review_status':'pending_manual_review','finished_at':ev.now()})
    lines = ['# 五epoch适配器日常与简单推理对照：原始回答','',
             '基础模型与Qwen3 SFT TEXT最终5epoch适配器，提示式解释，temperature=0，最多768token。',
             '所选Instruct不是原生Thinking模型；原始回答完整保留，未追加学习材料。不是完整通用能力基准。','']
    index = {(r['condition'],r['id']):r for r in records}
    for p in items:
        lines.extend([f"## {p['id']}：{p['category']}",'',p['question'],'',f"核验要点：{p['reference']}",''])
        for condition,label in [('base','基础模型'),('adapter','5epoch适配器')]:
            r = index[(condition,p['id'])]
            lines.extend([f'### {label}','',r['response'],'',f"结束原因：{r['finish_reason']}",''])
    (out/'answers_zh.md').write_text('\n'.join(lines)+'\n')
    save(out/'status.json',{'status':'generation_complete_pending_review','saved':len(records),'updated_at':ev.now()})


if __name__ == '__main__':
    main()

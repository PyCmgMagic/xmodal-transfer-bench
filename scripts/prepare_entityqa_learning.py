#!/usr/bin/env python3
"""Generate MMKU-style attribute QA using learning-only inputs, never test questions."""
import argparse
import atexit
import fcntl
import json
import os
from pathlib import Path
import re
import sys

from entityqa_common import rows, save, write_rows, sha, digest, now

SCHEMA = {'type': 'object', 'properties': {
    'question': {'type': 'string'}, 'answer': {'type': 'string'},
    'evidence': {'type': 'string'}},
    'required': ['question', 'answer', 'evidence'], 'additionalProperties': False}
SYSTEM = """Create ONE factual attribute question and its answer for supervised learning.
Use ONLY the supplied learning material. Do not use your own world knowledge.
Focus on the target claim, not background biography, identity, name, or occupation.
The question must contain the literal placeholder [SUBJECT] exactly once, instead of
the entity's name. It must be a natural, specific, standalone English question with
enough temporal/event context. The learner will NOT be shown the learning material.
Do not reveal the requested answer in the question. Do not write a multiple-choice,
yes/no, vague 'what happened', or entire-biography question.
The answer must be a short factual phrase or sentence (1-40 words), copied VERBATIM
from BOTH the learning material and the anonymized material. It must answer the
question about the target claim. Prefer a concise phrase over the whole sentence.
Never answer with [SUBJECT] or the subject's name. Preserve numbers, dates and names
of other entities exactly. The evidence must be a VERBATIM span from the learning
material containing the answer and sufficient support for the question.
Examples (use the format, never reuse these facts):
Material: Alex Example joined Example United on 2 April 2026.
Question: Which club did [SUBJECT] join on 2 April 2026?
Answer: Example United
Evidence: Alex Example joined Example United on 2 April 2026.
Material: Dana Sample has won the Example Cup twice, in 2024 and 2026.
Question: How many times has [SUBJECT] won the Example Cup, and in which years?
Answer: twice, in 2024 and 2026
Evidence: Dana Sample has won the Example Cup twice, in 2024 and 2026.
Return JSON with exactly question, answer, evidence. No reasoning or commentary."""


def norm(text):
    return ' '.join(text.casefold().split())


def inputs(data, work):
    # Deliberate whitelist: although entity records also contain nested evaluation
    # metadata, only their names/kinds are read into generation inputs.
    entities = {r['qid']: {k: r[k] for k in ['name', 'kind']}
                for r in rows(data / 'source/entities.jsonl')}
    evidence = {r['qid']: r['new_knowledge']['claim'] for r in rows(data / 'source/evidence.jsonl')}
    visual = {r['qid']: r for r in rows(data / 'shared/train_multimodal.jsonl')}
    records = []
    for row in rows(data / 'source/learning.jsonl'):
        entity, v = entities[row['qid']], visual[row['qid']]
        material = v['messages'][0]['content'].replace('<image>', '').strip()
        ref = re.search(r'the (?:person|entity|place|landmark|ship|spacecraft) '
                        r'(?:shown in the image|marked 1 with a red box)', material, re.I)
        reference = ref[0].lower() if ref else {
            'person': 'the person shown in the image', 'ship': 'the ship shown in the image',
            'spacecraft': 'the spacecraft shown in the image',
            'built_landmark': 'the landmark shown in the image',
            'natural_feature': 'the place shown in the image'}[entity['kind']]
        records.append(dict(id=row['id'], qid=row['qid'], **entity,
            learning=row['text'], visual_learning=material, claim=evidence[row['qid']],
            reference=reference, images=v['images']))
    assert len(records) == len({r['qid'] for r in records}) == 1814
    write_rows(work / 'data/learning_inputs.jsonl', records)
    save(work / 'data/learning_input_manifest.json', dict(created_at=now(),
        knowledge_units=len(records), generation_inputs='learning text, learning images, entity name/kind, target claim only',
        forbidden='No test question, option, answer label or test image is used to generate training QA.',
        source_sha256={f: sha(data/f) for f in ['source/entities.jsonl', 'source/learning.jsonl',
            'source/evidence.jsonl', 'shared/train_multimodal.jsonl']},
        sha256=sha(work/'data/learning_inputs.jsonl')))
    return records


def validate(qa, record):
    errors = []
    if not isinstance(qa, dict) or set(qa) != {'question', 'answer', 'evidence'}:
        return ['invalid_schema']
    q, a, e = (qa[k].strip() for k in ['question', 'answer', 'evidence'])
    if q.count('[SUBJECT]') != 1 or not q.endswith('?') or len(q)>400:
        errors.append('question_form')
    if norm(record['name']) in norm(q) or norm(record['name']) == norm(a.rstrip('.')):
        errors.append('literal_subject_name')
    if not 1 <= len(a.split()) <= 40 or len(a)<2 or len(a)>320:
        errors.append('answer_length')
    if any(a not in record[k] for k in ['learning', 'visual_learning']):
        errors.append('answer_not_exact_learning_and_visual_span')
    if e not in record['learning'] or a not in e:
        errors.append('evidence_not_exact_support_span')
    if re.search(r'(?<!\w)'+re.escape(norm(a))+r'(?!\w)',norm(q)):
        errors.append('answer_in_question')
    if re.search(r"(?:what|which) is \[SUBJECT\](?:'s|’s) (?:name|identity)|according to (?:the|this) (?:text|passage)|(?:^|\n)[A-D]\. ", q, re.I):
        errors.append('identity_or_context_or_options_question')
    if re.match(r'^(?:is|was|does|did|has|had|can|would|will|are|were)\b', q, re.I) or not re.search(r'\b(?:what|which|who|whom|whose|when|where|how)\b',q,re.I):
        errors.append('yes_no_question')
    return errors


def normalize_qa(qa, record):
    """Repair presentation only; never invent an answer or change its factual words."""
    if not isinstance(qa,dict) or set(qa)!={'question','answer','evidence'}:
        return qa,[]
    qa={k:v.strip() for k,v in qa.items()}
    changes=[]
    q=qa['question']
    if '[SUBJECT]' not in q and q.count(record['name'])==1:
        qa['question']=q.replace(record['name'],'[SUBJECT]')
        changes.append('literal_subject_to_placeholder')
    a=qa['answer']
    if any(a not in record[k] for k in ['learning','visual_learning']):
        trimmed=a.rstrip('.,;: ')
        if trimmed and all(trimmed in record[k] for k in ['learning','visual_learning']):
            qa['answer']=trimmed
            changes.append('remove_non_source_terminal_punctuation')
    if qa['evidence'] not in record['learning'] or qa['answer'] not in qa['evidence']:
        if qa['answer'] in record['learning']:
            # The complete source paragraph is a verifiable support reference.
            # This does not certify entailment; semantic QA limitations remain.
            qa['evidence']=record['learning']
            changes.append('evidence_reference_to_complete_original_learning_paragraph')
    return qa,changes


def answer_candidates(record):
    """Learning-claim spans only; no test target is consulted to choose a slot."""
    claim=record['claim']
    tokens=list(re.finditer(r"\w+(?:[-’']\w+)*",claim))
    stop=set('a an the in on at of for and or to as by with his her he she it is was were are has have had be been this that their from who which since until will would part not only won elected appointed became joined played served recorded remains active known'.split())
    subject_words=set(re.findall(r'\w+',record['name'].casefold()))
    candidates=set()
    for i in range(len(tokens)):
        for length in range(1,min(8,len(tokens)-i)+1):
            a=claim[tokens[i].start():tokens[i+length-1].end()]
            words=set(re.findall(r'\w+',a.casefold()))
            if tokens[i][0].casefold() in stop or tokens[i+length-1][0].casefold() in stop:
                continue
            if not words-subject_words-stop or len(a)>140 or len(a)<2:
                continue
            if all(a in record[k] for k in ['learning','visual_learning']) and norm(a)!=norm(record['name']):
                candidates.add(a)
    return sorted(candidates,key=lambda a:(-len(a),a))[:180]


def generate(args):
    data, work = args.data.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock=(work/'qa_generation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records = inputs(data, work)
    if not args.finalize_only:
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, VLLM_WORKER_MULTIPROC_METHOD='spawn',
            HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
        llm = LLM(model=str(args.model), dtype='bfloat16', seed=42, max_model_len=8192,
            max_num_seqs=32, max_num_batched_tokens=8192, gpu_memory_utilization=0.50,
            enforce_eager=True, enable_prefix_caching=True,
            limit_mm_per_prompt={'image':0, 'video':0})
        atexit.register(llm.llm_engine.engine_core.shutdown)
    audit = work / 'data/generation_audit.jsonl'
    accepted = {}
    last_invalid = {}
    by_qid = {r['qid']:r for r in records}
    if audit.exists():
        for r in rows(audit):
            qa,changes=normalize_qa(r.get('qa'),by_qid[r['qid']])
            if qa and r['source_sha256']==digest(by_qid[r['qid']]) and not validate(qa,by_qid[r['qid']]):
                accepted[r['qid']] = qa
                if changes:
                    with (work/'data/normalization_audit.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(qid=r['qid'],original_qa=r['qa'],normalized_qa=qa,changes=changes),ensure_ascii=False)+'\n')
            elif qa:
                last_invalid[r['qid']]={'previous':qa,'errors':validate(qa,by_qid[r['qid']])}
    for r in records:
        if r['qid'] in accepted:
            assert not validate(accepted[r['qid']], r)
    if args.reviewed_overrides:
        seen=set()
        for line in args.reviewed_overrides.open():
            qid,question,answer=line.rstrip('\n').split('\t')
            assert qid not in seen and qid in by_qid
            seen.add(qid)
            r=by_qid[qid]
            qa=dict(question=question,answer=answer,evidence=r['learning'])
            assert not validate(qa,r),(qid,validate(qa,r))
            accepted[qid]=qa
            with audit.open('a') as stream:
                stream.write(json.dumps(dict(qid=qid,attempt='agent_reviewed_learning_only',qa=qa,errors=[],
                    source_sha256=digest(r),overrides_sha256=sha(args.reviewed_overrides),
                    note='Authored from learning material and target claim; no test questions or labels; not human-certified'),ensure_ascii=False)+'\n')
    for attempt in range(0 if args.span_only or args.finalize_only else 8):
        pending = [r for r in records if r['qid'] not in accepted]
        for start in range(0, len(pending), 128):
            batch = pending[start:start+128]
            prompts = []
            for r in batch:
                content = json.dumps({k:r[k] for k in ['name','kind','learning','visual_learning','claim']},ensure_ascii=False)
                if r['qid'] in last_invalid:
                    content += '\nCorrect the following rejected QA. Fix ALL listed errors. '
                    content += 'Copy the shortest appropriate answer phrase exactly from the materials. '
                    content += 'Use [SUBJECT] exactly once and a what/which/when/where/how question. '
                    content += json.dumps(last_invalid[r['qid']],ensure_ascii=False)
                prompts.append(processor.apply_chat_template([
                    {'role':'system','content':SYSTEM}, {'role':'user','content':content}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            outputs = llm.generate(prompts, SamplingParams(max_tokens=640, temperature=0 if attempt==0 else 0.5,
                seed=42+attempt, structured_outputs=StructuredOutputsParams(json=SCHEMA)), use_tqdm=False)
            for r,out in zip(batch,outputs):
                response = out.outputs[0].text
                try:
                    raw_qa = json.loads(response)
                    qa,changes = normalize_qa(raw_qa,r)
                    errors = validate(qa,r)
                except (ValueError,TypeError,AttributeError):
                    qa, errors = None,['invalid_json']
                with audit.open('a') as stream:
                    stream.write(json.dumps(dict(qid=r['qid'],attempt=attempt,qa=qa,errors=errors,
                        response=response,finish_reason=out.outputs[0].finish_reason,
                        source_sha256=digest(r),generator_system_sha256=digest(SYSTEM),
                        normalizations=changes if qa is not None else []),ensure_ascii=False)+'\n')
                if not errors:
                    accepted[r['qid']] = qa
                elif qa:
                    last_invalid[r['qid']]={'previous':qa,'errors':errors}
            save(work/'qa_status.json', dict(stage='qa_generation', accepted=len(accepted),
                total=len(records),attempt=attempt,updated_at=now(),pid=os.getpid()))
            print(f'QA accepted {len(accepted)}/{len(records)}, attempt {attempt}',flush=True)
        if len(accepted)==len(records):
            break
    missing = [r['qid'] for r in records if r['qid'] not in accepted]
    # For stubborn format failures, make source-span copying a decoding
    # constraint. The model still selects a factual slot and writes its question.
    for attempt in range(0 if args.finalize_only else 4):
        pending=[r for r in records if r['qid'] not in accepted]
        if not pending:break
        for start in range(0,len(pending),64):
            batch=pending[start:start+64]
            prompts,params=[],[]
            for r in batch:
                candidates=answer_candidates(r)
                if not candidates:raise RuntimeError(f'No claim/learning common answer spans for {r["qid"]}')
                schema={'type':'object','properties':{
                    'question':{'type':'string','pattern':r'^(What|Which|When|Where|How|Who)[^\n]*\[SUBJECT\][^\n]*\?$'},
                    'answer':{'type':'string','enum':candidates}},
                    'required':['question','answer'],'additionalProperties':False}
                system=('Write a standalone factual attribute question about [SUBJECT] and its answer. '
                    'Choose one supplied answer span about the target claim. The answer is a short fact, not the subject identity. '
                    'Ask a specific what/which/when/where/how question containing [SUBJECT] exactly once. '
                    'Do not include the answer in the question. Do not ask a yes/no question. '
                    'Use only the supplied learning text. The learner will not see that text. '
                    'Return JSON question and answer; the answer must equal one allowed span.')
                content=json.dumps(dict(learning=r['learning'],target_claim=r['claim'],
                    subject_name=r['name'],allowed_answer_spans=candidates),ensure_ascii=False)
                prompts.append(processor.apply_chat_template([{'role':'system','content':system},
                    {'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
                params.append(SamplingParams(max_tokens=320,temperature=0.3+0.2*attempt,seed=57+attempt,
                    structured_outputs=StructuredOutputsParams(json=schema)))
            outputs=llm.generate(prompts,params,use_tqdm=False)
            for r,out in zip(batch,outputs):
                response=out.outputs[0].text
                try:
                    qa=json.loads(response)
                    qa['evidence']=r['learning']
                    qa,changes=normalize_qa(qa,r)
                    errors=validate(qa,r)
                except (ValueError,TypeError,AttributeError):qa,errors=None,['invalid_json']
                with audit.open('a') as stream:
                    stream.write(json.dumps(dict(qid=r['qid'],attempt='span_constrained_'+str(attempt),
                        qa=qa,errors=errors,response=response,source_sha256=digest(r),
                        generation_mode='question with exact learning-claim span enumeration; original paragraph evidence',
                        finish_reason=out.outputs[0].finish_reason),ensure_ascii=False)+'\n')
                if not errors:accepted[r['qid']]=qa
            save(work/'qa_status.json',dict(stage='qa_span_repair',accepted=len(accepted),total=len(records),
                attempt=attempt,updated_at=now(),pid=os.getpid()))
            print(f'Span repair accepted {len(accepted)}/{len(records)}, attempt {attempt}',flush=True)
    missing = [r['qid'] for r in records if r['qid'] not in accepted]
    save(work/'data/unresolved_generation.json', missing)
    if missing:
        raise RuntimeError(f'{len(missing)} unresolved learning QA; no silent dropping/fallback to heldout questions')
    qa_rows, text_rows, vision_rows = [], [], []
    for r in records:
        qa = accepted[r['qid']]
        assert not validate(qa,r)
        qtext = qa['question'].replace('[SUBJECT]',r['name'])
        qvision = qa['question'].replace('[SUBJECT]',r['reference'])
        if qvision.startswith('the '):
            qvision = 'The '+qvision[4:]
        base = dict(id=r['id'],qid=r['qid'])
        text_rows.append(dict(**base,messages=[{'role':'user','content':qtext},
            {'role':'assistant','content':qa['answer']}]))
        vision_rows.append(dict(**base,messages=[{'role':'user','content':'<image>\n'+qvision},
            {'role':'assistant','content':qa['answer']}],
            images=[str((data/p).resolve()) for p in r['images']]))
        qa_rows.append(dict(**base,**qa,question_text=qtext,question_vision=qvision,
            reference=r['reference'],source_sha256=digest(r)))
    write_rows(work/'data/qa_pairs.jsonl',qa_rows)
    write_rows(work/'data/train_text.jsonl',text_rows)
    write_rows(work/'data/train_vision.jsonl',vision_rows)
    save(work/'data/manifest.json',dict(status='complete',created_at=now(),knowledge_units=len(records),
        approach='MMKU-style attribute QA SFT, one QA per knowledge unit, identical answer across modalities',
        automatic_checks='Exact answer spans in learning/visual material, exact evidence, one placeholder, no answer in question',
        limitation='Automatically generated questions and source facts are not independently human-certified',
        generator_model=str(args.model),generator_system=SYSTEM,schema=SCHEMA,
        test_data_used=False,training_image_role='learning only',
        files={p.name:sha(p) for p in sorted((work/'data').glob('*.jsonl'))}))
    save(work/'qa_status.json',dict(stage='complete',accepted=len(records),updated_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gpu',default='1')
    p.add_argument('--span-only',action='store_true',help='Repair cached failures with exact source-span decoding')
    p.add_argument('--finalize-only',action='store_true',help='Build from audited QA and reviewed overrides without loading a model')
    p.add_argument('--reviewed-overrides',type=Path)
    args=p.parse_args()
    try:
        generate(args)
    except Exception as error:
        save(args.work/'qa_failure.json',dict(error=repr(error),time=now()))
        raise

"""Gold-independent multiple-choice readout and constrained finalization payloads."""
import json
import re
import unicodedata

LETTER = r'([A-Da-d])\b(?!\s*(?:[,;/&=]|\band\b|\bor\b)\s*[*_`]*[A-Da-d0-9]\b)'
PATTERNS = [
    ('boxed', re.compile(r'\\boxed\s*\{\s*([A-Da-d])\s*\}')),
    ('final_marker', re.compile(r'\bfinal(?:\s+(?:answer|choice|option|selection|decision))?\s*[:：=]\s*[*_`]*'+LETTER, re.I)),
    ('answer_declaration', re.compile(r'\b(?:the\s+)?(?:final\s+)?(?:correct\s+|accurate\s+)?(?:answer|choice|statement|selection|decision)\s*(?:is|[:：=])\s*(?:option\s+)?[*_`]*'+LETTER, re.I)),
    ('correct_option', re.compile(r'\b(?:the\s+)?(?:final\s+)?correct\s+(?:option|output)\s*(?:is|[:：=])\s*[*_`]*'+LETTER,re.I)),
    ('final_result', re.compile(r'\b(?:the\s+)?final\s+result\s*(?:is|[:：=])\s*[*_`]*'+LETTER,re.I)),
    ('option_assertion', re.compile(r'\boption\s+([A-Da-d])\s+(?:is\s+)?(?:the\s+)?(?:correct|accurate|right|best)\b', re.I)),
    ('option_match', re.compile(r'\b(?:this|that|it)\s+(?:matches|corresponds\s+to)\s+option\s+'+LETTER, re.I)),
    ('choose', re.compile(r'(?:^|[.!?\n]\s*)(?:therefore,?\s+|thus,?\s+|so,?\s+)?(?:I\s+)?(?:choose|select|return)\s+(?:option\s+)?[*_`]*'+LETTER, re.I)),
    ('letter_assertion', re.compile(r'(?:^|[.!?\n]\s*)(?:therefore,?\s+)?(?:only\s+)?(?:option\s+)?([A-Da-d])\s+is\s+(?:the\s+)?correct\s+(?:answer|choice|option)\b', re.I)),
    ('chinese', re.compile(r'(?:最终)?(?:正确)?答案\s*(?:是|为|[:：])\s*[*_`]*'+LETTER)),
]
NEGATED = re.compile(r'\b(?:not|isn\x27t|never|incorrect|wrong|reject|if|unless|whether|suppose|assuming)\b|不是|并非|如果', re.I)
CONDITIONAL = re.compile(r'^\s*(?:if|unless|only\s+if|or\b|and\b)',re.I)


def options_from_question(question):
    matches=list(re.finditer(r'(?m)^\s*([A-D])[.)、]\s+',question))
    if [m[1] for m in matches]!=list('ABCD'):
        return {}
    result={}
    for i,m in enumerate(matches):
        value=question[m.end():matches[i+1].start() if i+1<len(matches) else len(question)]
        value=re.split(r'\n\s*\n',value,maxsplit=1)[0].strip()
        result[m[1]]=value
    return result


def normalized(text):
    text=unicodedata.normalize('NFKC',text).casefold().strip()
    text=re.sub(r'^result\s*:\s*','',text)
    return re.sub(r'[^\w]+',' ',text).strip()


def parse_answer(response, options=None):
    """Return an explicit consistent option, never a gold-selected or guessed letter."""
    if not isinstance(response,str):return None,'non_text'
    if response.count('<think>')!=response.count('</think>'):return None,'unclosed_thinking'
    text=response.rsplit('</think>',1)[-1].strip()
    if not text:return None,'empty'
    # Machine-readable responses and bare options are valid too.
    cleaned=text.strip('` \n')
    if cleaned.startswith('json\n'):cleaned=cleaned[5:]
    try:
        obj=json.loads(cleaned)
    except (ValueError,TypeError):obj=None
    if isinstance(obj,dict) and set(obj) <= {'answer','final_answer','option'}:
        values={str(v).strip().upper() for v in obj.values()}
        if len(values)==1 and next(iter(values)) in set('ABCD'):return next(iter(values)),'json_answer'
    bare=re.fullmatch(r'[\s$\\()\[\]*_`]*([A-Da-d])[\s.!?$\\()\[\]*_`]*',text)
    if bare:return bare[1].upper(),'bare_option'
    findings=[]
    for kind,pattern in PATTERNS:
        for m in pattern.finditer(text):
            # Reject a multi-valued candidate result such as A=KEEP or C and D.
            tail=text[m.end():]
            if re.match(r'\s*[*_`]*(?:=|->|>|<|[,;/&]\s*[A-D]\b|(?:and|or)\s+[A-D]\b)',tail,re.I):continue
            # A lowercase English article ("selection is a separate event")
            # must not be mistaken for option A.
            if m[1].islower() and re.match(r'\s+[A-Za-z]',tail):continue
            start=max(text.rfind('.',0,m.start()),text.rfind('!',0,m.start()),
                      text.rfind('?',0,m.start()),text.rfind('\n',0,m.start()))+1
            prefix=text[start:m.start()]
            negated=NEGATED.search(prefix) if kind not in {'boxed','final_marker'} else re.search(r'\b(?:not|if)\s*$',prefix,re.I)
            if negated or CONDITIONAL.search(tail) or tail.lstrip().startswith('?'):continue
            if kind=='boxed' and re.search(r'\b(?:example|e\.g)\b',prefix,re.I):continue
            findings.append((m.start(),m[1].upper(),kind,m.end()))
    strong=[r for r in findings if r[2] in {'boxed','final_marker'}]
    if strong and re.fullmatch(r'[\s.!?$\\()\[\]*_`]*',text[max(strong)[3]:]):
        if len({r[1] for r in strong})>1:return None,'conflicting_answers'
        # An explicit terminal marker is the final selection; earlier natural
        # candidate results and self-corrections do not override it.
        last=max(strong)
        return last[1],last[2]
    letters={r[1] for r in findings}
    if len(letters)>1:return None,'conflicting_answers'
    if letters:return next(iter(letters)),max(findings)[2]
    # Literal option-content equality only, not semantic guessing from prose.
    if options:
        body=re.sub(r'^(?:the\s+)?(?:final\s+)?answer\s*(?:is|[:：])\s*','',text,flags=re.I)
        hits=[letter for letter,value in options.items() if normalized(body)==normalized(value)]
        if len(hits)==1:return hits[0],'exact_option_content'
    return None,'no_explicit_answer'


FINALIZER_SYSTEM = ('Your task is to finish a multiple-choice answer, not describe facts or explain a procedure. '
                    'Use the original question and your own previous response. Identify the computed result, '
                    'then match it to the COMPLETE content of one answer OPTION. Candidate IDs and answer option '
                    'letters are different: if the selected candidate is D but option B describes that result, output B. '
                    'Do not repeat reasoning, restate the rule, or refer the task back to the user. '
                    'If the previous response is incomplete or uncertain, select the most likely option yourself. '
                    'Return exactly one uppercase letter: A, B, C, or D.')


def remove_output_instructions(text):
    """Remove this project's known CoT suffixes only, preserving the question/options."""
    patterns=[
        r'\n\nReason step by step in a concise explanation \(at most 200 words\)\. '
        r'End with exactly FINAL: followed by one uppercase letter A, B, C, or D\.$',
        r'\n\nReason step by step in at most 200 words\. Finish with one final option '
        r'in \\boxed\{A\}, \\boxed\{B\}, \\boxed\{C\}, or \\boxed\{D\}\.$']
    for pattern in patterns:text=re.sub(pattern,'',text)
    return text


def finalizer_payload(original_payload, response):
    """Whitelist original inputs; labels and audit metadata never enter inference."""
    users=[remove_output_instructions(m['content']) for m in original_payload['messages'] if m['role']=='user']
    payload={'messages':[{'role':'system','content':FINALIZER_SYSTEM},
                         {'role':'user','content':'Original question and context:\n'+'\n\n'.join(users)+
                          '\n\nYour own previous response (uncorrected):\n'+response+
                          '\n\nNow select exactly one answer OPTION: A, B, C, or D.'}]}
    if original_payload.get('images'):payload['images']=list(original_payload['images'])
    return payload


def choice_hook(tokenizer):
    """ms-swift pre-infer hook: constrain one generated token to A/B/C/D."""
    ids=[tokenizer.encode(letter,add_special_tokens=False) for letter in 'ABCD']
    if not all(len(value)==1 for value in ids):raise ValueError('A/B/C/D must each tokenize to one token')
    allowed=[value[0] for value in ids]
    if len(set(allowed))!=4:raise ValueError('Option tokens must be distinct')
    def hook(kwargs):
        kwargs['inputs']['prefix_allowed_tokens_fn']=lambda batch_id,input_ids:allowed
        return kwargs
    return hook

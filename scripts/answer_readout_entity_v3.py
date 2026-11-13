"""Conservative extension for explicit terminal answer-as-option statements."""
import re
from answer_readout_entity_v2 import parse_answer as parse_v2

VERSION='entity-readout-v3-terminal-declaration'
TERMINAL=re.compile(
    r'(?:^|[.!?\n]\s*)(?:This\s+gives\s+|That\s+gives\s+)?'
    r'(?:the\s+)?(?:correct\s+)?answer\s+(?:as|is)\s+(?:option\s+)?'
    r'[*_`]*([A-D])[*_`]*[.!\s]*$',re.I)


def parse_answer(response,options=None):
    prediction,reason=parse_v2(response,options)
    if prediction is not None or reason!='no_explicit_answer' or not isinstance(response,str):
        return prediction,reason
    text=response.rsplit('</think>',1)[-1].strip()
    match=TERMINAL.search(text)
    if match:
        # Do not silently resolve an earlier explicit, different option assertion.
        mentions=set(re.findall(r'\boption\s+([A-D])\b',text,re.I))
        if any(letter.upper()!=match[1].upper() for letter in mentions):
            return None,'conflicting_answers'
        return match[1].upper(),'terminal_answer_as_option'
    return None,reason

"""Accept an unambiguous letter inside common LaTeX text wrappers."""
import re
from answer_readout_entity_v3 import parse_answer as parse_v3

VERSION='entity-readout-v4-boxed-text-letter'
BOXED_TEXT=re.compile(
    r'\\boxed\s*\{\s*\\(?:text|textrm|mathrm|mathbf|mathsf)\s*'
    r'\{\s*([A-Da-d])\s*\}\s*\}')


def parse_answer(response,options=None):
    if isinstance(response,str):
        canonical=BOXED_TEXT.sub(lambda m:'\\boxed{'+m[1]+'}',response)
        if canonical!=response and re.search(r'(?:^|[.!?\n]\s*)\s*(?:if|unless)\b[^.!?\n]*\\boxed',canonical,re.I):
            return None,'conditional_boxed_answer'
        pred,reason=parse_v3(canonical,options)
        if pred is not None and canonical!=response:
            return pred,'boxed_text_letter/'+reason
        return pred,reason
    return parse_v3(response,options)

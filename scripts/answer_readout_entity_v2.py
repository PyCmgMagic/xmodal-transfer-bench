"""Entity readout v2: accept a letter plus its unique complete option content."""
import re

from answer_readout import normalized, parse_answer as parse_v1

VERSION = 'entity-readout-v2-full-option-prefix'


def parse_answer(response, options=None):
    if isinstance(response, str):
        text = response.rsplit('</think>', 1)[-1].strip()
        boxes = re.findall(r'\\boxed\s*\{\s*([A-Da-d])\s*\}', text)
        if len({value.upper() for value in boxes}) > 1:
            return None, 'conflicting_answers'
    if isinstance(response, str) and options:
        prefixed = re.fullmatch(r'[*_`]*([A-D])[*_`]*[.):]\s+(.+)', text, re.S)
        if prefixed and response.count('<think>') == response.count('</think>'):
            hits = [letter for letter, value in options.items()
                    if normalized(prefixed[2]) == normalized(value)]
            if len(hits) > 1:
                return None, 'ambiguous_option_content'
            if len(hits) == 1:
                if hits[0] == prefixed[1]:
                    return hits[0], 'letter_and_exact_option_content'
                return None, 'letter_content_conflict'
    return parse_v1(response, options)

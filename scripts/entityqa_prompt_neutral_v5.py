"""Neutral entity MCQ protocol, versioned separately from legacy boxed examples."""
import re
from pathlib import Path

from entityqa_common import STRICT_SUFFIX

PROMPT_VERSION = 'entity-neutral-box-v5'
COT_SUFFIX = ('\n\nCompare the options with what you know. Explain briefly and do not overthink. '
              'End with the final option letter in \\boxed{}.')


def eval_payload(row, root):
    assert [m['role'] for m in row['messages']] in [['user'], ['system', 'user']]
    text = row['messages'][-1]['content'].rstrip()
    assert re.findall(r'(?m)^([A-D])\.\s', text) == list('ABCD')
    assert text.endswith(STRICT_SUFFIX), 'Unknown output instruction'
    text = text[:-len(STRICT_SUFFIX)].rstrip() + COT_SUFFIX
    payload = {'messages': [{'role': 'user', 'content': text}]}
    if row.get('images'):
        payload['images'] = [str((Path(root) / p).resolve()) for p in row['images']]
        assert all(Path(p).is_file() for p in payload['images'])
    assert text.count('<image>') == len(payload.get('images', []))
    return payload

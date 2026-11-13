"""Small, physical-line-safe utilities for the repaired entity QA experiment."""
import copy
import hashlib
import json
from pathlib import Path
import re
import time


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def render(payload, processor):
    """Only messages/images enter inference. Audit fields and labels are excluded."""
    from PIL import Image
    messages = copy.deepcopy(payload['messages'])
    paths = iter(payload.get('images', []))
    images = []
    for message in messages:
        content = message['content']
        if '<image>' not in content:
            continue
        blocks = []
        for i, piece in enumerate(content.split('<image>')):
            if i:
                with Image.open(next(paths)) as im:
                    images.append(im.convert('RGB'))
                blocks.append({'type': 'image'})
            if piece:
                blocks.append({'type': 'text', 'text': piece})
        message['content'] = blocks
    assert next(paths, None) is None, 'Unconsumed images'
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    assert not text.endswith('<think>\n')
    result = {'prompt': text}
    if images:
        result['multi_modal_data'] = {'image': images}
    return result


STRICT_SUFFIX = ('\n\nAnswer with exactly one uppercase letter (A, B, C, or D). '
                 'Do not include an explanation or any other characters.')
COT_SYSTEM = ('Answer as of 3 November 2026. Solve the multiple-choice question with brief reasoning. Do not overthink. '
              'Select exactly one of the provided options, even if uncertain. '
              'End with exactly one LaTeX box containing the final option letter; '
              'do not put other content after the box.')
COT_SUFFIX = ('\n\nThink step by step briefly, but do not overthink. Put the final answer '
              'in exactly one LaTeX box: \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}. '
              'The box must contain only the answer option letter.')


def eval_payload(row, root):
    assert [m['role'] for m in row['messages']] in [['user'], ['system', 'user']]
    text = row['messages'][-1]['content'].rstrip()
    assert re.findall(r'(?m)^([A-D])\.\s', text) == list('ABCD')
    assert text.endswith(STRICT_SUFFIX), 'Unknown output instruction'
    text = text[:-len(STRICT_SUFFIX)].rstrip() + COT_SUFFIX
    payload = {'messages': [{'role': 'system', 'content': COT_SYSTEM},
                            {'role': 'user', 'content': text}]}
    if row.get('images'):
        payload['images'] = [str((Path(root) / p).resolve()) for p in row['images']]
        assert all(Path(p).is_file() for p in payload['images'])
    assert text.count('<image>') == len(payload.get('images', []))
    return payload


def repeated_suffix(tokens):
    for period in range(8, min(128, len(tokens) // 4) + 1):
        block = tokens[-period:]
        if all(tokens[-period * (i + 1):-period * i] == block for i in range(1, 4)):
            return period
    return None

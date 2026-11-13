"""Gold-independent safeguards and final answer readout for prompted CoT."""
from collections import Counter
import copy
import re
import time

import run_v7_evaluation as ev
from answer_readout import FINALIZER_SYSTEM


def repeated_suffix(tokens, repeats=4, min_period=8, max_period=128):
    """Require four identical consecutive blocks, not merely repeated words."""
    for period in range(min_period, min(max_period, len(tokens) // repeats) + 1):
        block = tokens[-period:]
        if all(tokens[-period * (i + 1):-period * i] == block
               for i in range(1, repeats)):
            return period
    return None


class GenerationGuard:
    """Force EOS after an answer, an exact loop, or a per-sample deadline.

    Used as a Transformers logits processor, on batch size 1. Ordinary logits
    are left unchanged until a stopping condition is observed. Prompt tokens
    are excluded from all answer and repetition checks.
    """
    def __init__(self, tokenizer, prompt_length, eos_token_id, max_seconds=90):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.reason = None
        self.stopped_after_tokens = None

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('GenerationGuard requires batch size 1')
        n = input_ids.shape[-1] - self.prompt_length
        if not self.reason and n > 0:
            # Decode only generated tokens; the prompt itself lists four boxes.
            tail = input_ids[0, max(self.prompt_length, input_ids.shape[-1] - 512):].tolist()
            text = self.tokenizer.decode(tail, skip_special_tokens=False)
            if re.search(r'\\boxed\s*\{\s*[A-Da-d]\s*\}[\s$`.!?]*$', text):
                self.reason = 'terminal_box'
            elif n >= 128 and n % 16 == 0 and repeated_suffix(tail):
                self.reason = 'repeated_token_block'
            elif n >= 128 and n % 32 == 0:
                sentences = [s.strip() for s in re.findall(r'[^.!?\n]+[.!?]', text)]
                if any(count >= 4 and len(sentence) >= 40
                       for sentence, count in Counter(sentences).items()):
                    self.reason = 'repeated_sentence'
            if not self.reason and time.monotonic() - self.started >= self.max_seconds:
                self.reason = 'time_limit'
        if self.reason:
            if self.stopped_after_tokens is None:
                self.stopped_after_tokens = n
            scores.fill_(-float('inf'))
            scores[:, self.eos_token_id] = 0
        return scores


def inference_payload(prepared):
    """Explicit whitelist: neither gold nor audit fields can enter inference."""
    return copy.deepcopy({k: prepared[k] for k in ('messages', 'images') if k in prepared})


def final_readout_payload(strict_prepared, reasoning):
    """Use only the original test input and this model's uncorrected reasoning."""
    source = inference_payload(strict_prepared)
    question = source['messages'][-1]['content']
    if question.endswith(ev.SUFFIX):
        question = question[:-len(ev.SUFFIX)]
    payload = {'messages': [
        {'role': 'system', 'content': FINALIZER_SYSTEM},
        {'role': 'user', 'content': (
            'Original question:\n' + question +
            '\n\nYour own previous reasoning (uncorrected):\n' + reasoning +
            '\n\nFinish the answer now. Select exactly one answer OPTION: A, B, C, or D.')} ]}
    if source.get('images'):
        payload['images'] = source['images']
    return payload

"""Swift plugin: emphasize the decision token for one-letter MCQ targets."""
from swift.loss_scale import loss_scale_map
from swift.loss_scale.base import LossScale
from swift.template import ContextType


class MCQContentOnly(LossScale):
    is_binary = False

    def __call__(self, context_list, context_types, messages, **kwargs):
        assistants = [m for m in messages if m['role'] == 'assistant']
        self._mcq = bool(assistants and assistants[-1]['content'] in ('A', 'B', 'C', 'D'))
        try:
            return super().__call__(context_list, context_types, messages, **kwargs)
        finally:
            self._mcq = False

    def _inner_call(self, context, context_type, query, loss, loss_scale, is_last_round):
        if self._mcq and context_type == ContextType.SUFFIX:
            # Swift's dynamic-EOS pass restores exactly-zero suffix weights.
            return [context], [0.001]
        if self._mcq and context_type == ContextType.RESPONSE:
            # Three supervised tokens in Qwen3: answer, im_end, newline.
            # Its denominator is the count of labels, not the sum of weights.
            return [context], [3.]
        return super()._inner_call(context, context_type, query, loss, loss_scale, is_last_round)


loss_scale_map['v7_mcq_content'] = MCQContentOnly

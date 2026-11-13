"""Equal-row loss ablation with separate answer-content and suffix accounting."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from swift.loss import BaseLoss, loss_map


def components(logits, labels, suffix_ids):
    gold = labels[:, 1:].to(logits.device)
    valid = gold.ne(-100)
    counts = valid.sum(-1)
    assert counts.gt(len(suffix_ids)).all()
    content = valid.clone()
    for i in range(len(gold)):
        positions = valid[i].nonzero().flatten()
        suffix_positions = positions[-len(suffix_ids):]
        assert gold[i, suffix_positions].tolist() == suffix_ids
        content[i, suffix_positions] = False
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                         gold.reshape(-1), ignore_index=-100, reduction='none').reshape(gold.shape)
    suffix = valid & ~content
    all_ce = (ce * valid).sum(-1) / counts
    content_ce = (ce * content).sum(-1) / content.sum(-1)
    suffix_ce = (ce * suffix).sum(-1) / suffix.sum(-1)
    first = content.to(torch.int).argmax(-1)
    first_ce = ce.gather(1, first[:, None]).squeeze(1)
    return all_ce, content_ce, suffix_ce, first_ce, counts


class ContentLoss(BaseLoss):
    content_only = True

    def __call__(self, outputs, labels, *, loss_scale=None, **kwargs):
        assert loss_scale is None
        suffix_ids = json.loads(os.environ['V7_CONTENT_SUFFIX_IDS'])
        all_ce, answer_ce, suffix_ce, first_ce, counts = components(outputs.logits, labels, suffix_ids)
        loss = (answer_ce if self.content_only else all_ce).mean()
        path = os.environ.get('V7_CONTENT_LOSS_AUDIT')
        if path:
            record = {'step_before_update': self.trainer.state.global_step,
                      'content_only': self.content_only,
                      'supervised_tokens': counts.detach().cpu().tolist(),
                      'all_token_ce': all_ce.detach().cpu().tolist(),
                      'answer_content_ce': answer_ce.detach().cpu().tolist(),
                      'suffix_ce': suffix_ce.detach().cpu().tolist(),
                      'first_answer_token_ce': first_ce.detach().cpu().tolist(),
                      'loss': float(loss.detach())}
            with Path(path).open('a') as f:
                f.write(json.dumps(record) + '\n')
        return loss


class SuffixLoss(ContentLoss):
    content_only = False


loss_map['v7_answer_content'] = ContentLoss
loss_map['v7_suffix_control'] = SuffixLoss


def verify():
    torch.manual_seed(42)
    x = torch.randn(2, 7, 6, requires_grad=True)
    y = torch.tensor([[-100, -100, 1, 4, 5, -100, -100], [-100, 2, 3, 1, 4, 5, -100]])
    full, content, suffix, first, counts = components(x, y, [4, 5])
    ce = F.cross_entropy(x[:, :-1].reshape(-1, 6), y[:, 1:].reshape(-1), ignore_index=-100, reduction='none').reshape(2, 6)
    assert torch.allclose(content, torch.stack([ce[0, 1], ce[1, :3].mean()]))
    assert torch.allclose(first, torch.stack([ce[0, 1], ce[1, 0]]))
    assert not torch.allclose(content.mean(), full.mean())
    content.mean().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, 2:].eq(0).all(), 'Suffix and padding must have zero content-loss gradient'
    xx = F.pad(x.detach(), (0, 0, 0, 3))
    yy = F.pad(y, (0, 3), value=-100)
    assert torch.allclose(content, components(xx, yy, [4, 5])[1])
    return {'causal_shift': True, 'padding_invariant': True, 'suffix_gradient_zero': True,
            'equal_rows': True, 'finite_gradients': True}


if __name__ == '__main__':
    print(json.dumps(verify()))

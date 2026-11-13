"""ms-swift external plugin: equal per-example causal-LM loss, padded single GPU only."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from swift.loss import BaseLoss, loss_map


class EqualExampleLoss(BaseLoss):
    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, trainer=None, **kwargs):
        assert loss_scale is None, 'Per-token loss scaling is not part of this experiment'
        assert labels.ndim==2 and outputs.logits.shape[:2]==labels.shape
        # Explicit shift discards the final logit, with no wraparound to first token.
        gold=labels[:,1:].to(outputs.logits.device)
        logits=outputs.logits[:,:-1,:].float()
        valid=gold.ne(-100)
        counts=valid.sum(-1)
        assert counts.gt(0).all(), 'Every example must have a supervised answer'
        ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),gold.reshape(-1),ignore_index=-100,reduction='none').reshape(gold.shape)
        per_example=ce.sum(-1)/counts
        result=per_example.mean()
        path=os.environ.get('V7_EQUAL_LOSS_AUDIT')
        if path:
            record={'step_before_update':trainer.state.global_step if trainer else None,
                'supervised_tokens_including_template_suffix':counts.detach().cpu().tolist(),
                'per_example_mean_loss':per_example.detach().cpu().tolist(),
                'equal_row_weights':[1/labels.shape[0]]*labels.shape[0],
                'default_token_mean_row_weights':(counts/counts.sum()).detach().cpu().tolist(),
                'loss':float(result.detach()),'logits_shape':list(outputs.logits.shape)}
            with Path(path).open('a') as f:f.write(json.dumps(record)+'\n')
        return result


loss_map['v7_equal_example']=EqualExampleLoss


def verify():
    from types import SimpleNamespace
    x=torch.tensor([[[0.,0.],[2.,-1.],[1.,-2.],[0.,0.],[0.,0.]],
                    [[-1.,2.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]]],requires_grad=True)
    y=torch.tensor([[-100,0,0,0,-100],[-100,0,-100,-100,-100]])
    loss=EqualExampleLoss(None,None)(SimpleNamespace(logits=x),y)
    token=F.cross_entropy(x[:,:-1,:].reshape(-1,2),y[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(2,4)
    expected=(token[0,:3].mean()+token[1,0])/2
    assert torch.allclose(loss,expected) and not torch.allclose(loss,token.sum()/4)
    loss.backward();assert torch.isfinite(x.grad).all()
    ypad=F.pad(y,(0,2),value=-100);xpad=F.pad(x.detach(),(0,0,0,2))
    assert torch.allclose(loss.detach(),EqualExampleLoss(None,None)(SimpleNamespace(logits=xpad),ypad))
    return {'equal_example_loss':float(loss.detach()),'default_token_mean':float(token.detach().sum()/4),
            'padding_invariant':True,'finite_gradient':True,'causal_shift_checked':True}


if __name__=='__main__':print(json.dumps(verify()))

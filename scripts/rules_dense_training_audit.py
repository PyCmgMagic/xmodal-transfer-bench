"""Validate dense trainable scopes and real weight updates during SFT."""
import json
import os
from collections import Counter
from pathlib import Path

import torch
from swift.callbacks import TrainerCallback, callbacks_map


class RulesDenseAudit(TrainerCallback):
    def append(self, data):
        with Path(os.environ['RULES_DENSE_AUDIT']).open('a') as stream:
            stream.write(json.dumps(data) + '\n')
            stream.flush()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        groups = {'vision': Counter(), 'language': Counter()}
        self.probes = {}
        for name, parameter in model.named_parameters():
            assert 'lora_' not in name
            scope = 'vision' if '.visual.' in name else 'language'
            groups[scope]['total'] += parameter.numel()
            groups[scope]['trainable'] += parameter.numel() if parameter.requires_grad else 0
            if name.endswith('layers.0.self_attn.q_proj.weight') or name.endswith('visual.patch_embed.proj.weight'):
                self.probes[name] = (parameter, parameter.detach().cpu().clone())
        assert groups['language']['total'] == groups['language']['trainable'] > 0, groups
        expected = groups['vision']['total'] if os.environ['RULES_DENSE_MODALITY'] == 'vision' else 0
        assert groups['vision']['trainable'] == expected, groups
        assert self.probes
        self.append({'event': 'train_begin', 'groups': groups, 'dtype': str(next(model.parameters()).dtype),
                     'optimizer_requested': str(args.optim)})

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        self.step_lr = max(group['lr'] for group in optimizer.param_groups)

    def on_step_end(self, args, state, control, optimizer=None, **kwargs):
        stop = os.environ.get('RULES_DENSE_STOP_AT')
        if stop and state.global_step >= int(stop):
            control.should_save = True
            control.should_training_stop = True
        if state.global_step not in (1, state.max_steps):
            return
        changes = {name: {'changed': not torch.equal(p.detach().cpu(), before),
                          'finite': bool(torch.isfinite(p.detach()).all()),
                          'trainable': p.requires_grad} for name, (p, before) in self.probes.items()}
        assert all(v['finite'] for v in changes.values())
        if self.step_lr > 0:
            assert any(v['changed'] for name, v in changes.items() if '.visual.' not in name), changes
        for name, v in changes.items():
            if '.visual.' in name:
                if os.environ['RULES_DENSE_MODALITY'] == 'text':
                    assert not v['changed'], changes
                elif self.step_lr > 0:
                    assert v['changed'], changes
        self.append({'event': 'step_end', 'step': state.global_step, 'probes': changes,
                     'optimizer_step_lr': self.step_lr,
                     'cuda_peak_mib': torch.cuda.max_memory_allocated()/1024**2})


callbacks_map['rules_dense_training_audit'] = RulesDenseAudit

"""Record actual trainable scopes and FP32 master weights for dense SFT."""
import json
import os
from collections import Counter
from pathlib import Path

import torch
from swift.callbacks import TrainerCallback, callbacks_map


class FullParameterAudit(TrainerCallback):
    def append(self, record):
        path=os.environ.get('V7_PARAMETER_AUDIT')
        if path:
            with Path(path).open('a') as f:f.write(json.dumps(record)+'\n')

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        groups={}
        for name,p in model.named_parameters():
            scope='vision' if '.visual.' in name or name.startswith('visual.') else 'language'
            group=groups.setdefault(scope,Counter())
            group['parameters']+=p.numel();group['trainable']+=p.numel() if p.requires_grad else 0
        self.append({'event':'train_begin','groups':groups,'trainable_tensors':[
            {'name':n,'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in model.named_parameters() if p.requires_grad]})

    def on_step_end(self,args,state,control,optimizer=None,**kwargs):
        if state.global_step not in (1,state.max_steps):return
        chain=[];masters=[];opt=optimizer
        for _ in range(6):
            if opt is None:break
            chain.append(type(opt).__name__)
            for p in getattr(opt,'single_partition_of_fp32_groups',[]):
                masters.append({'dtype':str(p.dtype),'device':str(p.device),'numel':p.numel()})
            opt=getattr(opt,'optimizer',None)
        self.append({'event':'step_end','step':state.global_step,'optimizer_chain':chain,
            'master_weights':masters,'cuda_max_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})


callbacks_map['v7_full_parameter_audit']=FullParameterAudit

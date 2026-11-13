"""Equal-example loss plus an assertion that each optimizer step sees the planned labels."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from v7_equal_example_loss import EqualExampleLoss
from swift.loss import loss_map


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


class CurriculumLoss(EqualExampleLoss):
    def __call__(self,outputs,labels,*,trainer=None,**kwargs):
        if not hasattr(self,'plan'):
            self.plan=json.loads(Path(os.environ['V7_CURRICULUM_PLAN']).read_text())
        step=trainer.state.global_step
        expected=self.plan[step]
        gold=labels[:,1:]
        actual=[digest(row[row.ne(-100)].detach().cpu().tolist()) for row in gold]
        assert actual==expected['supervised_ids_sha256'],(step,actual,expected)
        result=super().__call__(outputs,labels,trainer=trainer,**kwargs)
        record={'step_before_update':step,'planned_channel':expected['channel'],
                'planned_row_ids':expected['row_ids'],'observed_supervised_ids_sha256':actual,
                'matches_planned_labels':True}
        with Path(os.environ['V7_CURRICULUM_ORDER_AUDIT']).open('a') as f:
            f.write(json.dumps(record)+'\n')
        return result


loss_map['v7_curriculum_equal']=CurriculumLoss

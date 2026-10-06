"""Same semantic roles/raw author tasks; two training-only increment controls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_global_task_role as previous
from trifusion.global_task_role_heads import GlobalTaskRoleHeads
from trifusion.incremental_role_objectives import batch_ratio_loss, repair_keep_loss

SCHEMA='trifusion-incremental-role-objective-v1'
OBJECTIVES=('md_batch_ratio','repair_keep')
inner=previous.previous.base.entry.entry
original_build_core=previous.build_core
original_condition=previous.condition
original_loss_values=previous.loss_values
original_training_batch=inner.runner._training_batch
ENVIRONMENTS={}
CURRENT_MODEL=None


class IncrementalObjectiveHeads(GlobalTaskRoleHeads):
    def forward(self,batch,*,return_aux=False):
        output=super().forward(batch,return_aux=return_aux)
        if return_aux:
            return {**output,'retrieval_environments':batch['retrieval_environments']}
        return output


def training_batch(raw):
    batch,labels=original_training_batch(raw)
    batch['retrieval_environments']=torch.tensor(
        [ENVIRONMENTS[Path(path).name] for path in raw[4]],device=labels.device)
    return batch,labels


def build_core(args,protocol):
    global CURRENT_MODEL
    assert args.variant=='semantic' and args.objective in OBJECTIVES
    model,cfg,binding=original_build_core(args,protocol)
    CURRENT_MODEL=model
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
        incremental_objective=args.objective,incremental_weight=1.0,
        repair_margin=0.1,repair_keep_cell_weights=[0.5,0.5],
        added_model_parameters=0,
        scope='Unchanged semantic evidence/raw author global+role tasks and1536 deployment. Training-only objective comparison; not complete P1/P2/P3 or novel loss claim.')
    return model,cfg,binding


def condition(args):
    return {**original_condition(args),'incremental_objective':args.objective,
        'incremental_weight':1.0,'repair_margin':0.1,'repair_keep_cell_weights':[0.5,0.5],
        'auxiliary_tasks':'incremental_role_objective_training_only',
        'relation_environment':'actual_protocol_scene_MSVR310_camera_others'}


def loss_values(args,output,labels,cameras,loss_fn):
    original,stats=original_loss_values(args,output,labels,cameras,loss_fn)
    if args.objective=='md_batch_ratio':
        increment,activity=batch_ratio_loss(output['shared_global'],output['correction'],output['raw_fused'],labels)
    else:
        increment,activity=repair_keep_loss(output['shared_global'],output['fused'],labels,output['retrieval_environments'])
    stats.update(activity,incremental_loss=float(increment.detach()))
    if args.mode=='m0':
        parameters=[(name,p) for name,p in CURRENT_MODEL.named_parameters()
            if name.startswith(('evidence_model.roles.query_projections.','evidence_model.roles.key_projections.'))]
        assert len(parameters)==6
        gradients=torch.autograd.grad(increment,[output['shared_global'],output['correction']]+[p for _,p in parameters],
            retain_graph=True,allow_unused=True)
        assert gradients[0] is None
        assert all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        stats['incremental_isolated_shared_global_gradient_absent']=True
        stats['incremental_isolated_correction_gradient_norm']=float(gradients[1].norm()) if gradients[1] is not None else 0.0
        stats['incremental_isolated_query_key_gradient_norms']={name:float(g.norm()) if g is not None else 0.0
            for (name,_),g in zip(parameters,gradients[2:])}
    return original+increment,stats


def configure():
    previous.configure()
    inner.AuthorHeadEvidence=IncrementalObjectiveHeads
    inner.runner._training_batch=training_batch
    inner.loss_values=loss_values
    base=previous.previous.base
    base.SCHEMA=SCHEMA;base.build_core=build_core;base.configure()
    inner.condition=condition;inner.foundation.condition=condition


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('RGBNT201','MSVR310','RGBNT100'),required=True)
    parser.add_argument('--objective',choices=OBJECTIVES,required=True)
    parser.add_argument('--mode',choices=('prepare','m0','train','evaluate'),required=True)
    for name in ('protocol','signal-source','clip-weight','initialization','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.variant=args.recipe='semantic';args.seed=42;args.epochs=50
    for name in ('protocol','signal_source','clip_weight','initialization','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    args.baseline_sha256=inner.runner.sha256(args.clip_weight)
    protocol=inner.runner.read_protocol(args.protocol,args.dataset)
    for records in protocol['records'].values():
        for row in records:
            key=Path(row['paths'][0]).name
            environment=row['scene'] if args.dataset=='MSVR310' else row['camera']
            assert key not in ENVIRONMENTS or ENVIRONMENTS[key]==environment
            ENVIRONMENTS[key]=environment
    configure()
    if args.mode=='prepare':
        assert not args.initialization.exists()
        _model,_cfg,binding=build_core(args,protocol)
        args.initialization.parent.mkdir(parents=True,exist_ok=True)
        args.initialization.write_text(json.dumps(dict(schema=SCHEMA,status='INITIALIZATION_VERIFIED',binding=binding,
            prepared_at=datetime.now().astimezone().isoformat()),indent=2)+'\n')
    else:
        (inner.foundation.evaluate if args.mode=='evaluate' else inner.train)(args,protocol)


if __name__=='__main__':
    main()

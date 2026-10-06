"""Pinned author SIM versus an active all-patch interaction control; fresh public CLIP."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_foundation_recipe as foundation

SCHEMA='trifusion-signal-selection-reference-v1'
original_core=foundation.build_core
original_train=foundation.train
original_optimization=foundation.optimization
original_training_batch=foundation.runner._training_batch
last_built_model=None
batch_log_path=None
batch_step=0
m0_parameter_before=None

def parameter_sha(parameter):
    return hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def training_batch(raw):
    global batch_step
    batch,labels=original_training_batch(raw)
    batch_step+=1
    with batch_log_path.open('a') as stream:
        stream.write(json.dumps(dict(global_step=batch_step,labels=raw[1].tolist(),cameras=raw[2].tolist(),
                                     view_ids=raw[3].tolist(),rgb_basenames=list(raw[4])))+'\n')
    return batch,labels

def all_patch(_module,rgb,nir,tir,_rgb_global,_nir_global,_tir_global):
    return rgb,nir,tir

def reference_forward(self,batch,*,return_aux=False):
    if return_aux:
        output=self.signal(batch['images'],cam_label=batch['camera_ids'],training=True)
        assert output[0]==2
        heads=[(output[i],output[i+1]) for i in range(1,len(output),2)]
        assert len(heads)==(2 if self.signal.direct else 4)
        raw=torch.cat([feature for _score,feature in heads],dim=1)
        assert raw.shape[1]==3072
        return {'heads':heads,'fused':raw}
    raw=self.signal(batch['images'],cam_label=batch['camera_ids'],training=False)
    assert raw.shape[1]==3072
    return F.normalize(raw.float(),dim=1)

def partition_signal(signal):
    encoder=signal.clip_vision_encoder
    visual=encoder.base
    assert torch.cuda.device_count()==2 and len(visual.transformer.resblocks)==12
    assert not visual.prompt_sign and not visual.adapter_sign
    before={name:id(parameter) for name,parameter in signal.named_parameters()}
    for parameter in (visual.class_embedding,visual.positional_embedding,encoder.cv_embed):
        parameter.data=parameter.data.to('cuda:1')
    visual.conv1.to('cuda:1')
    visual.ln_pre.to('cuda:1')
    for block in visual.transformer.resblocks[:6]:block.to('cuda:1')
    encoder.register_forward_pre_hook(lambda _module,inputs:(inputs[0].to('cuda:1'),*inputs[1:]))
    visual.transformer.resblocks[6].register_forward_pre_hook(lambda _module,inputs:(inputs[0].to('cuda:0'),*inputs[1:]))
    assert before=={name:id(parameter) for name,parameter in signal.named_parameters()}

def build_core(args,protocol):
    global last_built_model
    model,cfg,plain_binding=original_core(args,protocol)
    from modeling.AddModule.useA import Select_Interactive_Module
    from modeling.meta_arch import weights_init_classifier,weights_init_kaiming
    signal=model.signal
    assert not hasattr(signal,'SIM') and not hasattr(signal,'AlignM') and signal.feat_dim==512
    has_sim=args.selection!='global_only'
    if has_sim:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            signal.SIM=Select_Interactive_Module(512,k=int(cfg.MODEL.TOPK)).cuda()
            signal.classifier_var=nn.Linear(1536,signal.num_classes,bias=False)
            signal.classifier_var.apply(weights_init_classifier)
            signal.classifier_var.cuda()
            signal.bottleneck_var=nn.BatchNorm1d(1536)
            signal.bottleneck_var.bias.requires_grad_(False)
            signal.bottleneck_var.apply(weights_init_kaiming)
            signal.bottleneck_var.cuda()
        for name in ('W_q','W_k','W_v'):
            getattr(signal.SIM.token_selection,name).requires_grad_(False)
        if args.selection=='all_patch':
            signal.SIM.token_selection.forward=MethodType(all_patch,signal.SIM.token_selection)
        model.forward=MethodType(reference_forward,model)
    signal.use_A=has_sim
    signal.use_B=False
    cfg=cfg.clone();cfg.defrost();cfg.MODEL.USE_A=has_sim;cfg.MODEL.USE_B=False;cfg.freeze()
    signal.cfg=cfg
    partition_signal(signal)
    binding=dict(architecture=SCHEMA,dataset=args.dataset,selection=args.selection,seed=42,
                 plain_foundation_binding=plain_binding,
                 public_clip_sha256=args.baseline_sha256,protocol_sha256=foundation.runner.sha256(args.protocol),
                 visual_initial_sha256=plain_binding['visual_initial_sha256'],
                 camera_initial_sha256=plain_binding['camera_initial_sha256'],
                 initial_model_state_sha256=foundation.runner._module_state_sha256(model),
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 trainable_parameter_tensors=sum(p.requires_grad for p in model.parameters()),
                 frozen_selector_tensors=6 if has_sim else 0,selection_topk=int(cfg.MODEL.TOPK),
                 feature_width=3072 if has_sim else 1536,
                 batch_size=cfg.SOLVER.IMS_PER_BATCH,num_instances=cfg.DATALOADER.NUM_INSTANCE,
                 cfg_yaml=cfg.dump(),entry_sha256=foundation.runner.sha256(Path(__file__)),
                 boundary='Pinned SIM source; CPU-private seed42 new SIM/head initialization, not full author initialization sequence. Discrete-only selector projections retained and frozen. No roles/shared adapters/AlignM/ReID state.')
    last_built_model=model
    return model,cfg,binding

def build(args,protocol):
    model,cfg,binding=build_core(args,protocol)
    witness=json.loads(args.initialization.read_text())
    assert witness['schema']==SCHEMA and witness['binding']==binding
    return model,cfg,binding

def condition(args):
    return dict(selection=args.selection,seed=42,epochs=50,feature_width=1536 if args.selection=='global_only' else 3072,
                public_clip_sha256=args.baseline_sha256,
                initialization_sha256=foundation.runner.sha256(args.initialization))

def optimization(args,model,cfg):
    global m0_parameter_before
    torch.cuda.reset_peak_memory_stats(1)
    optimizer,scheduler,loss_fn=original_optimization(args,model,cfg)
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))
    assert set(ids)=={id(p) for p in model.parameters() if p.requires_grad}
    if args.mode=='m0':
        m0_parameter_before={name:parameter_sha(p) for name,p in model.named_parameters() if p.requires_grad}
    return optimizer,scheduler,loss_fn

def train(args,protocol):
    global batch_log_path,batch_step
    batch_log_path=args.output_dir/'training_batch_metadata.jsonl'
    assert not batch_log_path.exists()
    batch_step=0
    foundation.runner._training_batch=training_batch
    original_train(args,protocol)
    receipt_path=args.output_dir/'training.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['selection_reference']={
        'feature_width':1536 if args.selection=='global_only' else 3072,'selection':args.selection,
        'frozen_selector_tensors':0 if args.selection=='global_only' else 6,
        'peak_allocated_bytes_per_device':{str(i):torch.cuda.max_memory_allocated(i) for i in (0,1)},
        'memory_boundary':'Process through original train/M0 strict reload; two-card placement, not power or temperature.'}
    if args.mode=='m0':
        signal=last_built_model.signal
        names=('bottleneck',) if signal.direct else ('bottleneck_r','bottleneck_n','bottleneck_t')
        names=names if args.selection=='global_only' else (*names,'bottleneck_var')
        counts={name:int(getattr(signal,name).num_batches_tracked) for name in names}
        assert all(count==8 for count in counts.values())
        receipt['selection_reference']['m0_bn_counts']=counts
        changed={name:parameter_sha(p)!=m0_parameter_before[name]
                 for name,p in last_built_model.named_parameters() if p.requires_grad}
        receipt['selection_reference']['m0_parameter_changed']=changed
        if args.selection!='global_only':
            interaction=[name for name in changed if name.startswith('signal.SIM.modal_interactive.')]
            assert len(interaction)==12 and all(changed[name] for name in interaction)
            assert changed['signal.classifier_var.weight'] and changed['signal.bottleneck_var.weight']
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')

def main():
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('RGBNT201','RGBNT100','MSVR310'),required=True)
    parser.add_argument('--selection',choices=('global_only','masked','all_patch'),required=True)
    parser.add_argument('--mode',choices=('prepare','m0','train','evaluate'),required=True)
    for name in ('protocol','signal-source','clip-weight','initialization','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.recipe='author';args.seed=42;args.epochs=50
    for name in ('protocol','signal_source','clip_weight','initialization','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    args.baseline_sha256=foundation.runner.sha256(args.clip_weight)
    protocol=foundation.runner.read_protocol(args.protocol,args.dataset)
    foundation.SCHEMA=SCHEMA
    foundation.build_core=build_core
    foundation.build=build
    foundation.condition=condition
    foundation.optimization=optimization
    if args.mode=='prepare':
        assert not args.initialization.exists()
        _model,_cfg,binding=build_core(args,protocol)
        args.initialization.parent.mkdir(parents=True,exist_ok=True)
        args.initialization.write_text(json.dumps(dict(schema=SCHEMA,status='INITIALIZATION_VERIFIED',binding=binding,
            prepared_at=datetime.now().astimezone().isoformat()),indent=2)+'\n')
    else:
        (foundation.evaluate if args.mode=='evaluate' else train)(args,protocol)

if __name__=='__main__':main()

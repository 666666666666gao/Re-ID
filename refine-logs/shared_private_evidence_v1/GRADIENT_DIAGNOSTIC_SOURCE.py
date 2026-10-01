"""Eight real batches; replay private adapter local derivatives in FP32 at batch8."""
import json
from pathlib import Path
import sys
from datetime import datetime

import torch
import torch.nn.functional as F

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(ROOT))
from tools import run_shared_private_evidence as run
from tools.check_shared_private_initialization import options
from tools.queue_correspondence_roles import PROTOCOLS

OUT = ROOT / 'logs/shared_private_gradient_diagnostic_20261001_v1.json'
assert not OUT.exists()
run.activate('separated_roles')
args = options('RGBNT201', 'separated_roles')
protocol = run.control.runner.read_protocol(PROTOCOLS / 'RGBNT201.json', 'RGBNT201')
model, _, config, binding = run.control.build(args, protocol)
model.train()
named = {name:p for name,p in model.named_parameters() if p.requires_grad}
visual = model.backbone.signal.clip_vision_encoder.base
visual_names = {'backbone.signal.' + run.control.VISUAL_PREFIX + name for name,_ in visual.named_parameters()}
new_lr = config['OPTIMIZATION']['NEW_MODULE_LR']
multiplier = run.control.runner.learning_rate_multiplier(1, max_epochs=50, warmup_epochs=5)
optimizer = torch.optim.AdamW([
    {'params':[p for name,p in named.items() if name not in visual_names], 'lr':new_lr*multiplier},
    {'params':list(visual.parameters()), 'lr':run.control.VISUAL_LR*multiplier}],
    weight_decay=config['OPTIMIZATION']['WEIGHT_DECAY'])
scaler = torch.amp.GradScaler('cuda', init_scale=256.0)
loader = run.control.runner.loader_for(protocol, run.control.runner.records_for(protocol, 'train'),
                                       training=True, method='PLAIN_V8', seed=42)
samples = {}
handles = []
live = set()
losses = []
replayed = []
for index, raw in enumerate(loader):
    if index == 8: break
    if index == 7:
        for stage, bank in enumerate(model.backbone.private_adapters):
            for role, adapter in enumerate(bank):
                key = f'backbone.private_adapters.{stage}.{role}'
                samples[key] = []
                def capture(_module, inputs, output, *, key=key):
                    row = {'input':inputs[0].detach(), 'output_dtype':str(output.dtype), 'gradient':None}
                    samples[key].append(row)
                    output.register_hook(lambda gradient: row.update(gradient=gradient.detach()))
                handles.append(adapter.register_forward_hook(capture))
    batch, labels = run.control.runner._training_batch(raw)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(batch, return_aux=True)
        loss = F.cross_entropy(output['logits'], labels, label_smoothing=0.1) + run.control.runner._batch_hard_triplet(output['fused'], labels, margin=0.3)
    assert torch.isfinite(loss)
    scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    live.update(name for name,p in named.items() if p.grad is not None and bool(p.grad.abs().sum()>0))
    losses.append(float(loss.detach()))
    if index == 7:
        for handle in handles: handle.remove()
        for stage, bank in enumerate(model.backbone.private_adapters):
            for role, adapter in enumerate(bank):
                key = f'backbone.private_adapters.{stage}.{role}'
                params = list(adapter.named_parameters())
                totals = [torch.zeros_like(p) for _,p in params]
                assert len(samples[key]) == 3 and all(row['gradient'] is not None for row in samples[key])
                for row in samples[key]:
                    with torch.autocast('cuda', enabled=False):
                        predicted = adapter(row['input'].float())
                        gradients = torch.autograd.grad(predicted, [p for _,p in params],
                            grad_outputs=row['gradient'].float()/scale)
                    for total,gradient in zip(totals,gradients): total.add_(gradient)
                for (name,param),total in zip(params,totals):
                    replayed.append({'parameter':key+'.'+name,
                        'original_gradient_max_abs':float(param.grad.abs().max()) if param.grad is not None else None,
                        'fp32_local_replay_gradient_max_abs':float(total.abs().max()),
                        'input_dtype':str(samples[key][0]['input'].dtype),
                        'output_dtype':samples[key][0]['output_dtype']})
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() >= scale
prior = ROOT / 'trained-model/shared_private_preflight_20261001_v1_RGBNT201_separated_roles_m0/training_steps.jsonl'
reference = [json.loads(line)['loss'] for line in prior.read_text().splitlines()]
assert len(reference) == len(losses) == 8
result = {'status':'COMPLETE', 'completed_at':datetime.now().astimezone().isoformat(),
    'binding':binding, 'losses':losses, 'original_loss_max_abs_difference':max(abs(a-b) for a,b in zip(losses,reference)),
    'missing_nonzero_gradients':sorted(set(named)-live),'batch8_private_local_replay':replayed,
    'scope':'One explicit eight-real-batch diagnostic. FP32 local derivative replay uses same captured input/incoming gradient and pre-step weights; no formal train/eval/checkpoint/metric or M0 gate change.'}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'missing':len(result['missing_nonzero_gradients']),
    'loss_max_abs_difference':result['original_loss_max_abs_difference'],
    'zero_to_nonzero_fp32_replays':sum(row['original_gradient_max_abs']==0 and row['fp32_local_replay_gradient_max_abs']>0 for row in replayed),
    'output':str(OUT)}))

"""CPU algebra/gradient checks; this does not replace the production M0."""
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_absolute_pretrained_value as producer
from trifusion.absolute_pretrained_value import absolute_value_readout, AbsolutePretrainedValueTriFusion
from trifusion.pretrained_value_delta import value_delta_readout

torch.set_num_threads(2)
torch.manual_seed(42)
roles=nn.Module()
roles.query_projections=nn.ModuleList([nn.Linear(128,128,bias=False) for _ in range(3)])
roles.key_projections=nn.ModuleList([nn.Linear(128,128,bias=False) for _ in range(3)])
processed=tuple(torch.randn(2,3,16,128,requires_grad=True) for _ in range(3))
reference=tuple(torch.randn(2,3,16,128) for _ in range(3))
keys=tuple(torch.randn(2,3,128,128) for _ in range(3))
values=torch.randn(2,3,128,512)
correction,detail=absolute_value_readout(roles,processed,reference,keys,values)
direct=(detail['attention_weights']@values[:,None]).mean(dim=(1,3)).flatten(1)
assert torch.allclose(correction,direct,atol=1e-6,rtol=1e-6)
assert torch.allclose(detail['attention_weights'].sum(-1),torch.ones(2,3,3,16),atol=1e-6,rtol=0)
old,_=value_delta_readout(roles,processed,reference,keys,values,'uniform_reference')
mean=values.mean(2).flatten(1)
assert torch.allclose(correction,old+mean,atol=1e-6,rtol=1e-6)
offset=torch.randn(2,3,1,512)
shifted,_=absolute_value_readout(roles,processed,reference,keys,values+offset)
assert torch.allclose(shifted-correction,offset.squeeze(2).flatten(1),atol=1e-6,rtol=1e-6)
correction.square().mean().backward()
assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
           and bool(torch.count_nonzero(p.grad)) for p in roles.parameters())
assert all(bool(torch.isfinite(p.grad).all()) and bool(torch.count_nonzero(p.grad)) for p in processed)
producer.configure()
assert producer.previous.inner.RawFeatureSemanticTriFusion is AbsolutePretrainedValueTriFusion
assert producer.previous.previous.previous.base.SCHEMA==producer.SCHEMA
assert producer.previous.previous.previous.base.build_core is producer.build_core
print(json.dumps(dict(status='CPU_COMPONENT_AND_ENTRY_CONFIGURATION_PASS',
    direct_weighted_value_max_abs_difference=float((correction-direct).abs().max().detach()),
    common_value_shift_max_abs_difference=float((shifted-correction-offset.squeeze(2).flatten(1)).abs().max().detach()),
    trainable_projection_tensors=len(list(roles.parameters())),full_models=0,
    production_m0=False,formal_training=False)))

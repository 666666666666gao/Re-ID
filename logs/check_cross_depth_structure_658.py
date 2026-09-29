from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root / 'modeling'))
import torch
from trifusion.correspondence_context_identity import ContextSelectionRoles
from trifusion.cross_depth_role_state import CrossDepthStateRoles
from trifusion.experts.mamba import TinySequenceMixer

torch.set_num_threads(2)
torch.manual_seed(42)
kwargs = dict(grid=(16,8), width=128, anchor_side=4, m1=True, m2=True, mamba_factory=TinySequenceMixer)
old = ContextSelectionRoles(**kwargs).eval()
old.depth_logits.requires_grad_(False)
models = {}
for mode in ('mixed_once','depth_mean','depth_recurrent'):
    model = CrossDepthStateRoles(depth_mode=mode, **kwargs).eval()
    model.load_state_dict(old.state_dict(), strict=True)
    models[mode] = model
named = [{name for name,p in m.named_parameters() if p.requires_grad} for m in models.values()]
assert named[0] == named[1] == named[2]
base_stages = torch.randn(3,3,2,3,129,768)
context = torch.nn.functional.normalize(torch.randn(2,512), dim=1)
with torch.no_grad():
    reference = old(base_stages, context)
    control = models['mixed_once'](base_stages, context)
    differences = {name:float((getattr(reference,name)-getattr(control,name)).abs().max())
                   for name in ('cnn','transformer','mamba','positions')}
assert max(differences.values()) == 0.0
rows = []
outputs = {}
for mode,model in models.items():
    stages = base_stages.clone().requires_grad_(True)
    evidence = model(stages,context)
    outputs[mode] = evidence
    assert torch.equal(evidence.positions, reference.positions)
    target = torch.randn_like(evidence.cnn)
    loss = sum((getattr(evidence,role)*target).mean() for role in ('cnn','transformer','mamba'))
    loss.backward()
    assert torch.isfinite(stages.grad).all()
    depth_gradients = [float(stages.grad[d].abs().sum()) for d in range(3)]
    assert min(depth_gradients) > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    rows.append({'depth_mode':mode, 'role_shape':list(evidence.cnn.shape),
                 'trainable_parameter_tensors':len(named[0]),
                 'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'input_gradient_sum_by_depth':depth_gradients})
assert (outputs['depth_mean'].cnn-outputs['depth_recurrent'].cnn).abs().max() > 0
path = root / '.git/cross_depth_structure_658_20260929.json'
assert not path.exists()
record = {'status':'SYNTHETIC_CPU_STRUCTURE_PASS', 'at':datetime.now().astimezone().isoformat(),
          'source_sha256':hashlib.sha256((root/'modeling/trifusion/cross_depth_role_state.py').read_bytes()).hexdigest(),
          'mixed_once_reference_maximum_difference':differences, 'rows':rows,
          'boundary':'Synthetic inputs and existing TinySequenceMixer; tests exact control parity/common '
                     'addresses/parameter boundary/depth gradients. Not production Mamba M0 or ReID metrics.'}
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))

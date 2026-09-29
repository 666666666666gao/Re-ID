from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path('/data/gaob/Re-ID/Trifusion')
torch.set_num_threads(1)
sys.path.insert(0, str(ROOT / 'modeling'))
path = ROOT / '.git/correspondence_context_identity_652.py'
spec = importlib.util.spec_from_file_location('trifusion.correspondence_context_identity', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from trifusion.correspondence_evidence_readout import ContentSelectionRoles
from trifusion.state import MODALITY_ORDER

sources = json.loads((ROOT / '.git/m3_source_sha_640.json').read_text())
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in sources.items())


class TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.transformer = nn.Module()
        self.base.transformer.resblocks = nn.ModuleList(nn.Identity() for _ in range(12))
        self.projection = nn.Linear(768, 512, bias=False)

    def forward(self, tokens, *, cam_label, view_label):
        value = tokens.permute(1, 0, 2)
        for block in self.base.transformer.resblocks:
            value = block(value)
        return value.permute(1, 0, 2), self.projection(value[0])


class TokenSignal(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_vision_encoder = TokenEncoder()


def build(query, target):
    torch.manual_seed(42)
    return module.ContextIdentityTriFusion(
        TokenSignal(), query_mode=query, auxiliary_target=target, num_classes=7,
        grid=(16, 8), width=128, m1=True, m2=True, m3=False,
        mamba_factory=lambda width: nn.Linear(width, width, bias=False),
    )


conditions = (('static', 'none'), ('context', 'none'), ('static', 'local'),
              ('context', 'local'), ('context', 'global'))
models = [build(*condition) for condition in conditions]
first = models[0].state_dict()
assert all(set(m.state_dict()) == set(first) and all(torch.equal(v, m.state_dict()[k]) for k, v in first.items()) for m in models)
state_digest = hashlib.sha256()
for name, value in sorted(first.items()):
    state_digest.update(name.encode())
    state_digest.update(value.numpy().tobytes())

torch.manual_seed(123)
batch = {'images': {name: torch.randn(4, 129, 768) for name in MODALITY_ORDER},
         'camera_ids': torch.zeros(4, dtype=torch.long)}
labels = torch.tensor([0, 1, 2, 3])
for model in models:
    model.eval()
with torch.no_grad():
    outputs = [m(batch, return_aux=True) for m in models]
assert all(torch.equal(outputs[0]['fused'], r['fused']) for r in outputs)
assert all(r['fused'].shape == r['joint_local'].shape == (4, 1536) and r['logits'].shape == (4, 7) for r in outputs)

roles = models[0].roles
with torch.random.fork_rng(devices=[]):
    old = ContentSelectionRoles(selection='query', grid=roles.grid, width=128, m1=True, m2=True,
                                mamba_factory=lambda width: nn.Linear(width, width, bias=False))
old.load_state_dict({k: v for k, v in roles.state_dict().items() if not k.startswith('context_queries.')}, strict=True)
old.eval()
stages = torch.randn(3, 3, 4, 3, 129, 768)
with torch.no_grad():
    prior = old(stages)
    candidate = roles(stages, torch.randn(4, 512))
    parity = max(float((getattr(prior, n) - getattr(candidate, n)).abs().max()) for n in ('cnn', 'transformer', 'mamba', 'positions'))
assert parity < 1e-6, parity

diagnostic_roles = models[1].roles
with torch.no_grad():
    diagnostic_roles.context_queries.weight.normal_(std=0.01)
    q1 = diagnostic_roles.context_queries(F.normalize(torch.randn(4, 512), dim=1)).reshape(4, 3, 128)
    q2 = diagnostic_roles.context_queries(F.normalize(torch.randn(4, 512), dim=1)).reshape(4, 3, 128)
    positions = torch.zeros(4, 3, 16, 2)
    patches = torch.randn(4, 3, 128, 128)
    selected1 = diagnostic_roles.sample_context(patches, positions, q1, 0)
    selected2 = diagnostic_roles.sample_context(patches, positions, q2, 0)
    changed = float((selected1 - selected2).abs().max())
    constant = torch.ones_like(patches)
    same1 = diagnostic_roles.sample_context(constant, positions, q1, 0)
    same2 = diagnostic_roles.sample_context(constant, positions, q2, 0)
assert changed > 1e-6
assert torch.allclose(same1, torch.ones_like(same1), atol=1e-6) and torch.allclose(same1, same2, atol=1e-6)

gradient_rows = []
for condition in conditions:
    model = build(*condition).train()
    named = {k: v for k, v in model.named_parameters() if v.requires_grad}
    optimizer = torch.optim.AdamW(named.values(), lr=0.00035)
    live = set()
    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, return_aux=True)
        loss = F.cross_entropy(output['logits'], labels, label_smoothing=0.1)
        if condition[1] != 'none':
            loss = loss + F.cross_entropy(output['auxiliary_logits'], labels, label_smoothing=0.1)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in named.values() if p.grad is not None)
        live.update(k for k, p in named.items() if p.grad is not None and bool(p.grad.abs().sum() > 0))
        optimizer.step()
    assert live == set(named), sorted(set(named) - live)
    gradient_rows.append({'query_mode':condition[0], 'auxiliary_target':condition[1],
                          'trainable_parameters':sum(v.numel() for v in named.values()),
                          'trainable_tensors':len(named), 'nonzero_gradient_tensors':len(live)})
assert gradient_rows[0]['trainable_parameters'] == gradient_rows[1]['trainable_parameters']
assert len({r['trainable_parameters'] for r in gradient_rows[2:]}) == 1
assert not torch.cuda.is_initialized()
report = {'status':'CPU_SYNTHETIC_CONTEXT_QUERY_AND_IDENTITY_CONTRACT_PASS',
          'at':datetime.now().astimezone().isoformat(),
          'model_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'common_initial_state_sha256':state_digest.hexdigest(),
          'original_query_role_max_abs_difference':parity,
          'nonzero_context_changes_selection_max_abs_difference':changed,
          'constant_candidate_values_are_not_global_augmented':True,
          'common_zero_initialized_fused_output_equal':True,
          'gradient_rows':gradient_rows,
          'unchanged_existing_science_files':len(sources),
          'boundary':'Synthetic token encoder and linear Mamba substitute on CPU, not production M0 or genuine CLIP/Mamba/retrieval. No real images, labels, official scores or GPU use. Real source M0/reload, new training/evaluation/queue and code review are pending.'}
out = ROOT / '.git/context_identity_cpu_652_20260929.json'
assert not out.exists()
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

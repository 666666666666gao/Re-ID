"""CPU witness of loss ownership and persistent BN state, without real images."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modeling'))
from trifusion.evidence_author_heads import AuthorHeadEvidence
from trifusion.global_task_role_heads import GlobalTaskRoleHeads


class Core(nn.Module):
    def __init__(self, direct):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.signal = nn.Module()
        signal = self.backbone.signal
        signal.feat_dim, signal.num_classes, signal.direct = 512, 4, direct
        names = (('', 1536),) if direct else tuple((suffix, 512) for suffix in ('_r', '_n', '_t'))
        for suffix, width in names:
            neck = nn.BatchNorm1d(width)
            neck.bias.requires_grad_(False)
            setattr(signal, 'bottleneck' + suffix, neck)
            setattr(signal, 'classifier' + suffix, nn.Linear(width, 4, bias=False))
        self.encoder = nn.Linear(16, 1536)
        self.role = nn.Linear(1536, 1536)
        self.readout_gain = nn.Parameter(torch.tensor(0.1))
        self.neck, self.classifier = nn.Identity(), nn.Identity()

    def forward_features(self, batch):
        global_feature = self.encoder(batch['x'])
        correction = self.role(global_feature.detach())
        raw = global_feature + self.readout_gain * correction
        return {'shared_global': global_feature, 'correction': correction,
                'raw_fused': raw, 'fused': F.normalize(raw, dim=1)}


def task(heads, labels):
    return sum(F.cross_entropy(score, labels) for score, _feature in heads)


def witness(direct):
    torch.manual_seed(42)
    core = Core(direct)
    original = AuthorHeadEvidence(deepcopy(core)).train()
    model = GlobalTaskRoleHeads(core).train()
    assert set(model.state_dict()) == set(original.state_dict())
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in model.state_dict().items())
    batch, labels = {'x': torch.randn(8, 16)}, torch.arange(8) % 4
    result = model(batch, return_aux=True)
    control = original(batch, return_aux=True)
    assert torch.equal(result['raw_fused'], control['raw_fused'])
    assert torch.equal(result['fused'], control['fused'])
    assert all(torch.equal(left[0], right[0]) for left, right in zip(result['heads'], control['heads']))
    names, parameters = zip(*[(name, p) for name, p in model.named_parameters() if p.requires_grad])
    global_grad = torch.autograd.grad(task(result['global_heads'], labels), parameters, retain_graph=True, allow_unused=True)
    fused_grad = torch.autograd.grad(task(result['heads'], labels), parameters, allow_unused=True)
    for name, left, right in zip(names, global_grad, fused_grad):
        is_role = '.role.' in name or name.endswith('.readout_gain')
        active, absent = (right, left) if is_role else (left, right)
        assert absent is None, name
        assert active is not None and torch.isfinite(active).all() and active.abs().max() > 0, name
    assert all(int(getattr(model.signal, neck).num_batches_tracked) == 1 for neck, _ in model.head_names)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)
    for _ in range(7):
        optimizer.zero_grad(set_to_none=True)
        result = model(batch, return_aux=True)
        (task(result['global_heads'], labels) + task(result['heads'], labels)).backward()
        optimizer.step()
    counters = {name: int(getattr(model.signal, name).num_batches_tracked) for name, _ in model.head_names}
    assert all(value == 8 for value in counters.values())
    return {'direct': direct, 'state_and_capacity_unchanged': True, 'initial_fused_and_head_logits_exact': True,
            'global_and_fused_loss_gradients_disjoint': True, 'author_bn_forward_count': counters,
            'trainable_tensors': len(parameters)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    record = {'status': 'CPU_HEAD_OWNERSHIP_WITNESS_PASS', 'rows': [witness(True), witness(False)],
              'boundary': 'Synthetic identity-loss structural witness only; not full CLIP, production M0, metric-loss behavior, retrieval benefit or seed stability.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()

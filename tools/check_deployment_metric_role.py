"""Synthetic CPU witness using the sealed author's actual Triplet implementation."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_global_task_role_heads import Core
from trifusion.global_task_role_heads import GlobalTaskRoleHeads
from trifusion.deployment_metric_role import metric_heads
from tools.run_deployment_metric_role import loss_values


def witness(direct, triplet):
    torch.manual_seed(42)
    core = Core(direct)
    model = GlobalTaskRoleHeads(core).train()
    control = deepcopy(model).train()
    batch, labels = {'x': torch.randn(8, 16)}, torch.arange(8) % 4
    output = model(batch, return_aux=True)
    original = control(batch, return_aux=True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    heads = metric_heads(output)
    assert all(score is old_score for (score, _), (old_score, _) in zip(heads, output['heads']))
    assert all(feature is output['fused'] for _score, feature in heads)
    assert all(torch.equal(a, b) for a, b in zip(before.values(), model.state_dict().values()))
    assert torch.equal(output['raw_fused'], original['raw_fused'])
    assert torch.equal(output['fused'], original['fused'])
    assert all(torch.equal(a[0], b[0]) for a, b in zip(heads, original['heads']))
    assert all(torch.equal(a[1], b[1]) for a, b in zip(output['global_heads'], original['global_heads']))
    assert torch.allclose(output['fused'].norm(dim=1), torch.ones(8), atol=1e-6, rtol=0)
    global_task = sum(.25 * F.cross_entropy(score, labels) + triplet(feature, labels)[0]
                      for score, feature in output['global_heads'])
    role_task = sum(.25 * F.cross_entropy(score, labels) + triplet(feature, labels)[0]
                    for score, feature in heads)
    expected = sum(.25 * F.cross_entropy(score, labels) for score, _ in heads)
    expected = expected + len(heads) * triplet(output['fused'], labels)[0]
    torch.testing.assert_close(role_task, expected)
    actual, values = loss_values(argparse.Namespace(variant='semantic'), output, labels, torch.zeros(8),
        lambda score, feat, target, target_cam: .25 * F.cross_entropy(score, target) + triplet(feat, target)[0])
    torch.testing.assert_close(actual, global_task + role_task)
    assert abs(values['global_loss'] - float(global_task.detach())) < 1e-6
    assert abs(values['fused_role_loss'] - float(role_task.detach())) < 1e-6
    original_role = sum(.25 * F.cross_entropy(score, labels) + triplet(feature, labels)[0]
                        for score, feature in original['heads'])
    assert abs(float(role_task.detach() - original_role.detach())) > 1e-4
    names, parameters = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    global_grad = torch.autograd.grad(global_task, parameters, retain_graph=True, allow_unused=True)
    role_grad = torch.autograd.grad(role_task, parameters, allow_unused=True)
    for name, left, right in zip(names, global_grad, role_grad):
        is_role = '.role.' in name or name.endswith('.readout_gain')
        active, absent = (right, left) if is_role else (left, right)
        assert absent is None, name
        assert active is not None and torch.isfinite(active).all() and active.abs().max() > 0, name
    assert set(model.state_dict()) == set(control.state_dict())
    assert all(int(getattr(model.signal, neck).num_batches_tracked) == 1 for neck, _ in model.head_names)
    return dict(direct=direct, role_metric_shape=[8, 1536], original_metric_width=1536 if direct else 512,
                actual_runner_loss_matches_reference=True, raw_logits_and_global_task_unchanged=True, gradient_ownership_disjoint=True,
                original_role_loss=float(original_role.detach()), deployment_role_loss=float(role_task.detach()),
                original_head_sum_preserved=True, joint_triplet_multiplicity=len(heads), persistent_bn_updates=1,
                parameter_count=sum(p.numel() for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signal-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    path = args.signal_source / 'layers/triplet_loss.py'
    spec = importlib.util.spec_from_file_location('sealed_author_triplet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(1)
    result = dict(status='CPU_DEPLOYMENT_METRIC_ROLE_WITNESS_PASS', rows=[witness(True, module.TripletLoss()), witness(False, module.TripletLoss())],
                  boundary='Synthetic actual-author Triplet witness, not production M0 or retrieval performance. No model-state change. Vehicle metric changes both normalization and joint geometry.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Synthetic CPU head/ownership check; no CLIP, images, optimizer or retrieval."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modeling'))
from trifusion.global_task_role_heads import GlobalTaskRoleHeads
from trifusion.normalized_role_classification import NormalizedRoleClassification

AUTHOR_TRIPLET = 'comparators/Signal-cd1b0a6/layers/triplet_loss.py'
spec = importlib.util.spec_from_file_location('sealed_author_triplet', ROOT / AUTHOR_TRIPLET)
author_triplet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author_triplet)


class SyntheticEvidence(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.backbone = nn.Module()
        signal = nn.Module()
        signal.feat_dim, signal.direct, signal.num_classes = 512, True, 2
        signal.bottleneck = nn.BatchNorm1d(1536)
        signal.bottleneck.bias.requires_grad_(False)
        signal.classifier = nn.Linear(1536, 2, bias=False)
        with torch.no_grad():
            signal.classifier.weight.zero_()
            signal.classifier.weight[:, :2] = torch.tensor([[0.7, -0.2], [-0.4, 0.9]])
        self.backbone.signal = signal
        self.neck = nn.Identity()
        self.classifier = nn.Linear(1536, 2, bias=False)
        g = torch.zeros(4, 1536)
        c = torch.zeros_like(g)
        g[:, :2] = torch.tensor([[2., .2], [1.2, .8], [-1.8, -.4], [-1.1, .9]])
        c[:, :2] = torch.tensor([[.1, .3], [-.15, .25], [.12, -.3], [.08, -.2]])
        c[:, 1] += shift
        self.shared = nn.Parameter(g)
        self.correction = nn.Parameter(c)
        self.readout_gain = nn.Parameter(torch.tensor(1.))
        self.calls = 0

    def forward_features(self, _batch):
        self.calls += 1
        raw = self.shared + self.readout_gain * self.correction
        return {'shared_global': self.shared, 'correction': self.correction,
                'raw_fused': raw, 'fused': F.normalize(raw, dim=1)}


def check_case(wrapper, shift):
    torch.manual_seed(42)
    model = wrapper(SyntheticEvidence(shift)).cpu().train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    bn = model.signal.bottleneck
    output = model({}, return_aux=True)
    assert model.evidence_model.calls == 1
    raw = model.evidence_model.shared.detach() + model.evidence_model.correction
    assert torch.equal(output['raw_fused'], raw)
    assert torch.equal(output['heads'][0][1], raw)
    assert torch.equal(output['global_heads'][0][1], model.evidence_model.shared)
    assert torch.equal(output['fused'], F.normalize(raw, dim=1))
    assert int(bn.num_batches_tracked) == 1
    torch.testing.assert_close(bn.running_mean, .1 * model.evidence_model.shared.detach().mean(0), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(bn.running_var, .9 + .1 * model.evidence_model.shared.detach().var(0), rtol=1e-6, atol=1e-6)
    labels = torch.tensor([0, 0, 1, 1])
    ce = F.cross_entropy(output['heads'][0][0], labels, label_smoothing=.1)
    role_targets = [model.evidence_model.correction, model.evidence_model.readout_gain,
                    model.evidence_model.shared, bn.weight, model.signal.classifier.weight]
    role_grads = torch.autograd.grad(ce, role_targets, retain_graph=True, allow_unused=True)
    assert torch.isfinite(role_grads[0]).all() and role_grads[0].abs().max() > 0
    assert torch.isfinite(role_grads[1]) and role_grads[1].abs() > 0
    assert all(value is None for value in role_grads[2:])
    global_ce = F.cross_entropy(output['global_heads'][0][0], labels, label_smoothing=.1)
    global_targets = [model.evidence_model.shared, bn.weight, model.signal.classifier.weight,
                      model.evidence_model.correction, model.evidence_model.readout_gain]
    global_grads = torch.autograd.grad(global_ce, global_targets, allow_unused=True)
    assert all(value is not None and torch.isfinite(value).all() and value.abs().max() > 0 for value in global_grads[:3])
    assert all(value is None for value in global_grads[3:])
    tri, ap, an = author_triplet.TripletLoss()(output['heads'][0][1], labels)
    for name, value in model.named_parameters():
        assert torch.equal(before[name], value.detach())
    saved_bn = {name: value.detach().clone() for name, value in bn.named_buffers()}
    model.eval()
    assert torch.equal(model({}), output['fused'].detach())
    assert model.evidence_model.calls == 2
    assert all(torch.equal(saved_bn[name], value) for name, value in bn.named_buffers())
    return dict(role_logits=output['heads'][0][0].detach(), global_logits=output['global_heads'][0][0].detach(),
                raw=raw.detach(), fused=output['fused'].detach(), ce=float(ce.detach()),
                triplet=float(tri.detach()), hardest_positive=ap.detach(), hardest_negative=an.detach(),
                common_shift_ce_gradient_max=float(role_grads[0].sum(0).abs().max()),
                correction_gradient_max=float(role_grads[0].abs().max()), gain_gradient=float(role_grads[1]),
                state_signature=[(name, list(value.shape)) for name, value in model.state_dict().items()],
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                bn_batches_tracked=int(bn.num_batches_tracked),
                role_global_head_gradient_isolated=True, one_evidence_forward_per_call=True,
                raw_metric_feature_exact=True, deployment_vector_exact=True, parameters_unchanged=True,
                global_only_running_statistics_exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(2)
    cases = {name: [check_case(wrapper, shift) for shift in (0., 2.)]
             for name, wrapper in [('raw', GlobalTaskRoleHeads), ('normalized_ce', NormalizedRoleClassification)]}
    raw, candidate = cases['raw'], cases['normalized_ce']
    for left, right in zip(raw, candidate):
        for key in ('raw', 'fused', 'global_logits', 'hardest_positive', 'hardest_negative'):
            assert torch.equal(left[key], right[key]), key
        for key in ('triplet', 'state_signature', 'total_parameters', 'trainable_parameters', 'bn_batches_tracked'):
            assert left[key] == right[key], key
    raw_logit_delta = float((raw[0]['role_logits'] - raw[1]['role_logits']).abs().max())
    candidate_logit_delta = float((candidate[0]['role_logits'] - candidate[1]['role_logits']).abs().max())
    assert raw_logit_delta < 1e-5
    assert abs(raw[0]['ce'] - raw[1]['ce']) < 1e-5
    assert abs(raw[0]['triplet'] - raw[1]['triplet']) < 1e-5
    assert candidate_logit_delta > .01 and abs(candidate[0]['ce'] - candidate[1]['ce']) > .001
    assert raw[0]['common_shift_ce_gradient_max'] < 1e-5
    assert candidate[0]['common_shift_ce_gradient_max'] > .001
    result = dict(schema='normalized-role-classification-synthetic-cpu-v1', status='PASS_SYNTHETIC_HEAD_ONLY',
                  torch_version=torch.__version__, dtype='float32', device='cpu', seed=42,
                  raw_common_shift_logit_delta=raw_logit_delta,
                  normalized_ce_common_shift_logit_delta=candidate_logit_delta,
                  fixed_common_shift=[0., 2.], feature_dim=1536, batch_size=4, label_smoothing=.1,
                  role_classification='l2(raw) before existing detached train-BN/classifier', metric='raw',
                  counts=dict(synthetic_evidence_calls=8, training_head_calls=8, classification_vjp_calls=8,
                              full_clip_or_role_model_calls=0, optimizer_updates=0, real_data_records=0,
                              real_initializations=0, M0_runs=0, formal_training_runs=0, retrieval_evaluations=0,
                              weights_written=0, weights_deleted=0),
                  cases={name: [{key: value for key, value in case.items() if not torch.is_tensor(value)}
                                for case in rows] for name, rows in cases.items()},
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                      'modeling/trifusion/global_task_role_heads.py', 'modeling/trifusion/evidence_author_heads.py',
                      'modeling/trifusion/normalized_role_classification.py', AUTHOR_TRIPLET,
                      'tools/check_normalized_role_classification_cpu.py')},
                  limits=['Synthetic head fixture, not the real CLIP/roles, production M0 or retrieval gain.',
                          'Finite witness breaks this selected common-shift CE symmetry; not all blind directions.',
                          'BN after L2 remains: this is not direct cosine classification or NormFace replication.'])
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'raw_common_shift_logit_delta',
                                                  'normalized_ce_common_shift_logit_delta', 'counts')}))


if __name__ == '__main__':
    main()

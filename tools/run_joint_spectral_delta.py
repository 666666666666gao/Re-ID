"""Fixed201 raw-author objectives with additive/product/reference controls."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.joint_spectral_roles import JointSpectralSemanticTriFusion
from trifusion.joint_spectral_delta import JointSpectralDelta

SCHEMA = 'trifusion-joint-spectral-delta-v1'
POLICY = previous.POLICY
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
MODE = 'joint_delta'
M0_OWNERSHIP_CHECKED = False


class SpectralM0Diagnostics(inner.M0Diagnostics):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.spectral = dict(model.evidence_model.spectral_readout.named_parameters())
        assert len(self.spectral) == 2 and sum(p.numel() for p in self.spectral.values()) == 589836
        self.spectral_initial = {name: p.detach().clone() for name, p in self.spectral.items()}
        self.spectral_updates = []

    def after_step(self, optimizer, args, kwargs):
        super().after_step(optimizer, args, kwargs)
        self.spectral_updates.append({name: {
            'gradient_finite': bool(torch.isfinite(p.grad).all()),
            'gradient_max_abs': float(p.grad.detach().abs().max()),
            'parameter_delta_max_abs': float((p.detach() - self.spectral_initial[name]).abs().max()),
        } for name, p in self.spectral.items()})

    def result(self):
        value = super().result()
        assert M0_OWNERSHIP_CHECKED and len(self.spectral_updates) == 8
        assert all(row[name]['gradient_finite'] for row in self.spectral_updates for name in self.spectral)
        assert all(any(row[name]['gradient_max_abs'] > 0 for row in self.spectral_updates) for name in self.spectral)
        assert all(self.spectral_updates[-1][name]['parameter_delta_max_abs'] > 0 for name in self.spectral)
        return dict(value, spectral_updates=self.spectral_updates,
                    role_shared_gradient_isolated=True,
                    reference_gradient='unused_control' if MODE == 'joint_absolute' else 'finite_nonzero')


def build_core(args, protocol):
    assert args.dataset == 'RGBNT201' and args.variant == 'semantic'
    model, cfg, binding = original_build_core(args, protocol)
    assert isinstance(model.evidence_model, JointSpectralSemanticTriFusion)
    assert model.evidence_model.spectral_readout.mode == MODE
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
                   spectral_aggregation=MODE, spectral_rank=4,
                   spectral_additional_parameters=589836,
                   scope='Same201 public initialization/raw author objectives; cross-spectral content and differentiable semantic reference. Dense readout and slot pooling are common to all three controls; no extra head/loss/resource.')
    return model, cfg, binding


def condition(args):
    return dict(previous.condition(args), spectral_aggregation=MODE, spectral_rank=4,
                spectral_additional_parameters=589836,
                readout='dense1536_from_role_modality_pooled128')


def loss_values(args, output, labels, cameras, loss_fn):
    global M0_OWNERSHIP_CHECKED
    role_loss, values = previous.previous.loss_values(args, output, labels, cameras, loss_fn)
    global_loss, global_values = previous.previous.original_loss_values(
        args, {**output, 'heads': output['global_heads']}, labels, cameras, loss_fn)
    diagnostics = output['spectral_diagnostics']
    if args.mode == 'm0' and not M0_OWNERSHIP_CHECKED:
        shared_grad, reference_grad = torch.autograd.grad(
            role_loss, (output['shared_global'], diagnostics['reference_descriptor']),
            allow_unused=True, retain_graph=True)
        assert shared_grad is None
        if MODE == 'joint_absolute':
            assert reference_grad is None
        else:
            assert reference_grad is not None and bool(torch.isfinite(reference_grad).all())
            assert int(torch.count_nonzero(reference_grad)) > 0
        M0_OWNERSHIP_CHECKED = True
        print({'event': 'spectral_m0_gradient_ownership', 'mode': MODE,
               'role_to_shared_global': 'unused',
               'reference': 'unused_control' if reference_grad is None else 'finite_nonzero'}, flush=True)
    values.update(global_loss=float(global_loss.detach()), fused_role_loss=float(role_loss.detach()),
                  global_head_losses=global_values['head_losses'])
    for name in ('actual_descriptor', 'reference_descriptor', 'delta_descriptor'):
        values[name + '_norm_mean'] = float(diagnostics[name].detach().norm(dim=-1).mean())
    return global_loss + role_loss, values


def configure():
    previous.configure()
    JointSpectralSemanticTriFusion.aggregation_mode = MODE
    inner.RawFeatureSemanticTriFusion = JointSpectralSemanticTriFusion
    inner.loss_values = loss_values
    inner.M0Diagnostics = SpectralM0Diagnostics
    base = previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


def main():
    global MODE
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--spectral-aggregation', choices=JointSpectralDelta.MODES, required=True)
    own_args, remaining = parser.parse_known_args()
    MODE = own_args.spectral_aggregation
    sys.argv = [sys.argv[0], *remaining]
    configure()
    inner.main()


if __name__ == '__main__':
    main()

"""Keep training unchanged; observe the isolated M0 VJP outside autocast."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_incremental_role_objective as entry

original_build_core = entry.build_core
original_loss_values = entry.loss_values


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    binding.update(entry_sha256=entry.inner.runner.sha256(Path(__file__)),
        m0_auxiliary_vjp_policy='unscaled_autocast_disabled_same_original_targets')
    return model, cfg, binding


def loss_values(args, output, labels, cameras, loss_fn):
    if args.mode != 'm0':
        return original_loss_values(args, output, labels, cameras, loss_fn)
    original, stats = entry.original_loss_values(args, output, labels, cameras, loss_fn)
    if args.objective == 'md_batch_ratio':
        increment, activity = entry.batch_ratio_loss(output['shared_global'], output['correction'], output['raw_fused'], labels)
    else:
        increment, activity = entry.repair_keep_loss(output['shared_global'], output['fused'], labels, output['retrieval_environments'])
    stats.update(activity, incremental_loss=float(increment.detach()))
    parameters = [(name, parameter) for name, parameter in entry.CURRENT_MODEL.named_parameters()
        if name.startswith(('evidence_model.roles.query_projections.', 'evidence_model.roles.key_projections.'))]
    assert len(parameters) == 6
    with torch.autocast('cuda', enabled=False):
        gradients = torch.autograd.grad(increment, [output['shared_global'], output['correction']] + [parameter for _, parameter in parameters],
            retain_graph=True, allow_unused=True)
    assert gradients[0] is None and all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients)
    stats['incremental_isolated_shared_global_gradient_absent'] = True
    stats['incremental_isolated_correction_gradient_norm'] = float(gradients[1].norm()) if gradients[1] is not None else 0.0
    stats['incremental_isolated_query_key_gradient_norms'] = {name: float(gradient.norm()) if gradient is not None else 0.0
        for (name, _), gradient in zip(parameters, gradients[2:])}
    stats['incremental_isolated_vjp_autocast_enabled'] = False
    stats['incremental_isolated_query_key_unused'] = {name: gradient is None for (name, _), gradient in zip(parameters, gradients[2:])}
    return original + increment, stats


if __name__ == '__main__':
    entry.build_core = build_core
    entry.loss_values = loss_values
    entry.main()

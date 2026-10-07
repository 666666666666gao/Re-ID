"""Matched author training with stop-gradient at the role-read boundary only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_native_research as base
from trifusion.role_input_detach import DetachedSemanticTriFusion, DetachedNativeTriFusion

SCHEMA = 'trifusion-role-input-detach-v1'
POLICY = 'detach_stages_context_shared_global_at_role_read_only'
original_build_core = base.build_core
original_loss_values = base.entry.entry.loss_values


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    binding.update(entry_sha256=base.runner.sha256(Path(__file__)), role_input_gradient_policy=POLICY)
    return model, cfg, binding


def loss_values(args, output, labels, cameras, loss_fn):
    loss, values = original_loss_values(args, output, labels, cameras, loss_fn)
    g = output['shared_global'].detach().float()
    c = output['correction'].detach().float()
    h = output['raw_fused'].detach().float()
    values['shared_global_norm_mean'] = float(g.norm(dim=1).mean())
    values['correction_norm_mean'] = float(c.norm(dim=1).mean())
    values['actual_scaled_correction_norm_mean'] = float((h - g).norm(dim=1).mean())
    values['actual_scaled_correction_global_ratio_mean'] = float(((h - g).norm(dim=1) / g.norm(dim=1)).mean())
    return loss, values


def configure():
    inner = base.entry.entry
    inner.RawFeatureSemanticTriFusion = DetachedSemanticTriFusion
    inner.IndependentNativeTriFusion = DetachedNativeTriFusion
    inner.loss_values = loss_values
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()


if __name__ == '__main__':
    configure()
    base.entry.entry.main()

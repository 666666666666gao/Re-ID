"""Same raw author tasks; roles choose actual pretrained value changes."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.pretrained_value_delta import PretrainedValueDeltaTriFusion

SCHEMA = 'trifusion-pretrained-value-delta-v1'
POLICY = previous.POLICY
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
MODE = 'semantic_reference'
M0_CHECKED = False


def build_core(args, protocol):
    assert args.dataset == 'RGBNT201' and args.variant == 'semantic'
    model, cfg, binding = original_build_core(args, protocol)
    assert isinstance(model.evidence_model, PretrainedValueDeltaTriFusion)
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)), value_reference=MODE,
        removed_readout_parameters=589824, added_parameters=0,
        scope='Same public CLIP/raw author/global-role tasks. Existing roles select attention changes over detached actual-QK512 values; old free384-to1536 readout removed. Semantic versus uniform reference, no extra head/loss/teacher or new resource.')
    return model, cfg, binding


def condition(args):
    return dict(previous.condition(args), value_reference=MODE,
        value_source='last_actual_qk_attention_ln_post_visual_projection',
        value_gradient='detached', readout='mean3roles16slots_in_original3x512basis')


def loss_values(args, output, labels, cameras, loss_fn):
    global M0_CHECKED
    loss, values = previous.loss_values(args, output, labels, cameras, loss_fn)
    change = output['value_diagnostics']['attention_change']
    regions = output['value_diagnostics']['regions']
    if args.mode == 'm0' and not M0_CHECKED:
        role_loss, _ = previous.previous.loss_values(args, output, labels, cameras, loss_fn)
        global_grad, change_grad = torch.autograd.grad(role_loss,
            (output['shared_global'], change), allow_unused=True, retain_graph=True)
        assert global_grad is None
        assert change_grad is not None and bool(torch.isfinite(change_grad).all())
        assert int(torch.count_nonzero(change_grad)) > 0
        M0_CHECKED = True
        print({'event':'pretrained_value_m0_ownership','reference':MODE,
               'role_to_shared_global':'unused','attention_change':'finite_nonzero'},flush=True)
    values.update(value_attention_change_abs_mean=float(change.detach().abs().mean()),
        value_attention_change_row_sum_max_abs=float(change.detach().sum(-1).abs().max()),
        value_region_norm_mean=float(regions.detach().norm(dim=-1).mean()))
    return loss, values


class ValueM0Diagnostics(inner.M0Diagnostics):
    def result(self):
        result = super().result()
        assert M0_CHECKED
        return dict(result, pretrained_value_role_global_isolated=True,
            pretrained_value_reference=MODE, attention_change_task_gradient='finite_nonzero')


def configure():
    previous.configure()
    PretrainedValueDeltaTriFusion.reference_mode = MODE
    inner.RawFeatureSemanticTriFusion = PretrainedValueDeltaTriFusion
    inner.loss_values = loss_values
    inner.M0Diagnostics = ValueM0Diagnostics
    base = previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--value-reference', choices=('semantic_reference','uniform_reference'), required=True)
    own, remaining = parser.parse_known_args()
    MODE = own.value_reference
    sys.argv = [sys.argv[0], *remaining]
    configure()
    inner.main()

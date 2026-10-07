"""RAW global/role objectives; patch versus mean-query reconstruction."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.region_evidence_reconstruction import (
    PatchReconstructionTriFusion, MeanReconstructionTriFusion)

SCHEMA = 'trifusion-region-evidence-reconstruction-v1'
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_condition = previous.condition
OriginalDiagnostics = inner.M0Diagnostics


class ReconstructionM0Diagnostics(OriginalDiagnostics):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.detail = {name: p for name, p in model.named_parameters()
                       if '.reconstruction.' in name and p.requires_grad}
        self.initial = {name: p.detach().clone() for name, p in self.detail.items()}

    def result(self):
        assert len(self.updates) == 8
        assert len(self.detail) == 15 and sum(p.numel() for p in self.detail.values()) == 105232
        assert all(any(row[name]['unscaled_gradient_max_abs'] is not None
                       and row[name]['unscaled_gradient_max_abs'] > 0 for row in self.updates)
                   for name in self.detail)
        assert all(self.updates[-1][name]['parameter_delta_from_initial_max_abs'] > 0
                   for name in self.detail)
        tracked = {neck: int(getattr(self.model.signal, neck).num_batches_tracked)
                   for neck, _ in self.model.head_names}
        assert all(value == 8 for value in tracked.values())
        return dict(effective_optimizer_updates=8, optimizer_groups_at_construction=self.groups,
                    author_bn_batches_tracked=tracked, reconstruction_parameters=list(self.detail),
                    reconstruction_updates=self.updates, active_reconstruction_parameters=105232,
                    boundary='Actual cumulative gradient/update and BN support; no retrieval claim.')


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    mode = 'patch' if args.variant == 'semantic' else 'mean'
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
                   reconstruction_query_mode=mode, active_reconstruction_parameters=105232,
                   evidence_source_policy='existing_CNN_role_semantic_patches_no_added_image_path',
                   scope='Visual-only candidate attention before the unchanged CNN region read; patch queries versus repeated image mean, same active parameters. RAW responsibilities and author recipe unchanged.')
    return model, cfg, binding


def condition(args):
    return {**original_condition(args),
            'reconstruction_query_mode': 'patch' if args.variant == 'semantic' else 'mean',
            'active_reconstruction_parameters': 105232,
            'evidence_source_policy': 'existing_CNN_role_semantic_patches_no_added_image_path'}


def configure():
    previous.configure()
    inner.RawFeatureSemanticTriFusion = PatchReconstructionTriFusion
    inner.IndependentNativeTriFusion = MeanReconstructionTriFusion
    inner.M0Diagnostics = ReconstructionM0Diagnostics
    base = previous.previous.base
    base.SCHEMA, base.build_core = SCHEMA, build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    configure()
    inner.main()

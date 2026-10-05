"""Same raw global/role author objectives; active semantic capacity control."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.semantic_capacity_evidence import SemanticCapacityTriFusion

SCHEMA = 'trifusion-semantic-capacity-control-v1'
SOURCE_POLICY = 'CLIP_role_C_patches_active_MLP_128_202_202_128_to_512_candidates'
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_condition = previous.condition


class CapacityM0Diagnostics(inner.M0Diagnostics):
    def result(self):
        assert len(self.updates) == 8
        assert len(self.detail) == 14 and sum(p.numel() for p in self.detail.values()) == 159096
        assert all(any(row[name]['unscaled_gradient_max_abs'] is not None
                       and row[name]['unscaled_gradient_max_abs'] > 0 for row in self.updates)
                   for name in self.detail)
        assert all(self.updates[-1][name]['parameter_delta_from_initial_max_abs'] > 0
                   for name in self.detail)
        tracked = {neck: int(getattr(self.model.signal, neck).num_batches_tracked)
                   for neck, _ in self.model.head_names}
        assert all(value == 8 for value in tracked.values())
        return {'effective_optimizer_updates': 8, 'optimizer_groups_at_construction': self.groups,
                'author_bn_batches_tracked': tracked, 'detail_parameters': list(self.detail),
                'detail_updates': self.updates, 'active_reader_parameters': 159096,
                'boundary': 'Same finite gradient/update/BN support gates; own actual parameter budget. No retrieval claim.'}


def build_core(args, protocol):
    assert args.variant == 'native'  # Existing factory slot; explicit source binding below.
    model, cfg, binding = original_build_core(args, protocol)
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
        evidence_source_policy=SOURCE_POLICY, active_reader_parameters=159096,
        native_reader_parameter_gap=-200,
        scope='Semantic capacity control only; same raw author global/role objectives, detach policy, recipe and deployment. No new images, loss, margin, gain or seed.')
    return model, cfg, binding


def condition(args):
    return {**original_condition(args), 'evidence_source_policy': SOURCE_POLICY,
            'active_reader_parameters': 159096, 'native_reader_parameter_gap': -200}


def configure():
    previous.configure()
    inner.IndependentNativeTriFusion = SemanticCapacityTriFusion
    inner.M0Diagnostics = CapacityM0Diagnostics
    base = previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    configure()
    inner.main()

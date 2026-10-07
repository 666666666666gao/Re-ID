"""RAW duties; slot-specific versus pair-uniform cross-spectrum mass."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.selective_role_transport import SlotMassTriFusion, UniformMassTriFusion

SCHEMA = 'trifusion-selective-role-transport-v1'
MASS_MODES = {'semantic': 'slot_mass', 'native': 'uniform_mass'}
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_condition = previous.condition
OriginalDiagnostics = inner.M0Diagnostics


class TransportM0Diagnostics(OriginalDiagnostics):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.added = {name: p for name, p in model.named_parameters()
                      if '.transport.' in name and p.requires_grad}
        self.detail = self.added
        self.initial = {name: p.detach().clone() for name, p in self.detail.items()}

    def result(self):
        assert len(self.updates) == 8
        assert len(self.added) == 2 and sum(p.numel() for p in self.added.values()) == 32768
        assert all(any(row[name]['unscaled_gradient_max_abs'] is not None
                       and row[name]['unscaled_gradient_max_abs'] > 0 for row in self.updates)
                   for name in self.added)
        assert all(self.updates[-1][name]['parameter_delta_from_initial_max_abs'] > 0
                   for name in self.added)
        tracked = {neck: int(getattr(self.model.signal, neck).num_batches_tracked)
                   for neck, _ in self.model.head_names}
        assert all(value == 8 for value in tracked.values())
        return dict(effective_optimizer_updates=8, optimizer_groups_at_construction=self.groups,
                    author_bn_batches_tracked=tracked, transport_parameters=list(self.added),
                    transport_updates=self.updates, active_transport_parameters=32768,
                    boundary='Cumulative real gradient/update and author BN activity; no retrieval claim.')


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
                   transport_mass_mode=MASS_MODES[args.variant],
                   active_transport_parameters=32768,
                   mamba_sequence_policy='three_private_16_slot_sequences_then_peer_transport',
                   scope='Original RAW-semantic inputs; same-state pair compares local real mass with pair-uniform real mass. No native input, reconstruction, extra loss or external resource.')
    return model, cfg, binding


def condition(args):
    return {**original_condition(args), 'transport_mass_mode': MASS_MODES[args.variant],
            'active_transport_parameters': 32768, 'transport_iterations': 100,
            'transport_null_logit': 1.0,
            'mamba_sequence_policy': 'three_private_16_slot_sequences_then_peer_transport'}


def loss_values(args, output, labels, cameras, loss_fn):
    loss, values = previous.loss_values(args, output, labels, cameras, loss_fn)
    values.update({name: float(value) for name, value in output['transport_diagnostics'].items()})
    return loss, values


def configure():
    previous.configure()
    inner.RawFeatureSemanticTriFusion = SlotMassTriFusion
    inner.IndependentNativeTriFusion = UniformMassTriFusion
    inner.M0Diagnostics = TransportM0Diagnostics
    base = previous.previous.base
    base.SCHEMA, base.build_core = SCHEMA, build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition
    inner.loss_values = loss_values


if __name__ == '__main__':
    configure()
    inner.main()

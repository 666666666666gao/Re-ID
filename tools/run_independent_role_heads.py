"""Independent fused heads; unchanged raw global/role objectives and evidence."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.independent_role_heads import IndependentRoleHeads

SCHEMA = 'trifusion-independent-role-heads-v1'
POLICY = 'global_author_heads_to_global_task_separate_fused_heads_to_role_task'
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_condition = previous.condition


class HeadM0Diagnostics(inner.M0Diagnostics):
    def __init__(self, model, optimizer):
        super().__init__(model, optimizer)
        self.head_parameters = {name: p for name, p in model.named_parameters()
                                if name.startswith('role_heads.') and p.requires_grad}
        self.head_initial = {name: p.detach().clone() for name, p in self.head_parameters.items()}
        self.head_updates = []

    def after_step(self, optimizer, args, kwargs):
        super().after_step(optimizer, args, kwargs)
        self.head_updates.append({name: {
            'unscaled_gradient_max_abs': float(p.grad.detach().abs().max()) if p.grad is not None else None,
            'parameter_delta_from_initial_max_abs': float((p.detach() - self.head_initial[name]).abs().max())
        } for name, p in self.head_parameters.items()})

    def result(self):
        value = super().result()
        assert len(self.head_updates) == 8
        assert len(self.head_parameters) == 2 * len(self.model.head_names)
        assert all(any(row[name]['unscaled_gradient_max_abs'] is not None and
                       row[name]['unscaled_gradient_max_abs'] > 0 for row in self.head_updates)
                   for name in self.head_parameters)
        assert all(self.head_updates[-1][name]['parameter_delta_from_initial_max_abs'] > 0
                   for name in self.head_parameters)
        counters = [int(head['neck'].num_batches_tracked) for head in self.model.role_heads]
        assert all(value == 8 for value in counters)
        return {**value, 'role_head_parameters': list(self.head_parameters),
                'role_head_updates': self.head_updates, 'role_bn_batches_tracked': counters}


def build_core(args, protocol):
    assert args.variant == 'semantic'
    model, cfg, binding = original_build_core(args, protocol)
    added = {name: p for name, p in model.named_parameters() if name.startswith('role_heads.') and p.requires_grad}
    assert len(added) == 2 * len(model.head_names)
    assert sum(p.numel() for p in added.values()) == 1536 * (model.signal.num_classes + 1)
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
        objective_gradient_policy=POLICY, role_head_initialization='deepcopy_original_global_heads_no_rng_consumption',
        role_head_parameters=sum(p.numel() for p in added.values()), role_head_tensors=len(added),
        scope='Semantic evidence unchanged. Independent fused heads train only with raw role task; original heads and shared features train only with global task. Extra trainable head capacity disclosed; not an original research module.')
    return model, cfg, binding


def condition(args):
    return {**original_condition(args), 'objective_gradient_policy': POLICY,
            'role_head_initialization': 'deepcopy_original_global_heads_no_rng_consumption'}


def configure():
    previous.configure()
    inner.AuthorHeadEvidence = IndependentRoleHeads
    inner.M0Diagnostics = HeadM0Diagnostics
    base = previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    configure()
    inner.main()

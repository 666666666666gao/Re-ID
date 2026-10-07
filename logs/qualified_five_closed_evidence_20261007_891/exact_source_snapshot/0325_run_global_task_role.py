"""Matched author objectives with explicit shared/head versus role ownership."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_role_input_detach as previous
from trifusion.global_task_role_heads import GlobalTaskRoleHeads

SCHEMA = 'trifusion-global-task-role-v1'
POLICY = 'author_global_loss_to_shared_and_heads_fused_loss_to_roles_only'
original_build_core = previous.build_core
original_condition = previous.base.entry.condition


def build_core(args, protocol):
    assert args.variant in ('semantic', 'native')
    model, cfg, binding = original_build_core(args, protocol)
    binding.update(entry_sha256=previous.base.runner.sha256(Path(__file__)),
                   objective_gradient_policy=POLICY,
                   scope='Fresh public CLIP/camera/heads and unchanged semantic/native state; separate global and fused author task gradients. No persistent extra head or external data.')
    return model, cfg, binding


def loss_values(args, output, labels, cameras, loss_fn):
    fused_loss, values = previous.loss_values(args, output, labels, cameras, loss_fn)
    global_loss, global_values = previous.original_loss_values(
        args, {**output, 'heads': output['global_heads']}, labels, cameras, loss_fn)
    values.update(global_loss=float(global_loss.detach()), fused_role_loss=float(fused_loss.detach()),
                  global_head_losses=global_values['head_losses'])
    return global_loss + fused_loss, values


def condition(args):
    return {**original_condition(args), 'objective_gradient_policy': POLICY,
            'author_training_objectives': ['shared_global', 'role_corrected_fused']}


def configure():
    previous.configure()
    base = previous.base
    inner = base.entry.entry
    inner.AuthorHeadEvidence = GlobalTaskRoleHeads
    inner.loss_values = loss_values
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    configure()
    previous.base.entry.entry.main()

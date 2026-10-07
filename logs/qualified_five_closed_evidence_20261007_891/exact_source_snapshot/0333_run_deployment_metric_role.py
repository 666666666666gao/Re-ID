"""Role-only Triplet in deployment geometry; original global task unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.deployment_metric_role import metric_heads

SCHEMA = 'trifusion-deployment-metric-role-v1'
METRIC_POLICY = 'role_triplet_joint_1536_l2_h_global_triplet_original_raw_parts'
original_build_core = previous.build_core
original_condition = previous.condition


def build_core(args, protocol):
    model, cfg, binding = original_build_core(args, protocol)
    binding.update(entry_sha256=previous.previous.base.runner.sha256(Path(__file__)),
                   role_metric_policy=METRIC_POLICY,
                   scope='Same initialized state, raw BN logits and global author task; only role Triplet uses actual joint1536 L2(h). Vehicles also change per-modal to joint metric geometry.')
    return model, cfg, binding


def loss_values(args, output, labels, cameras, loss_fn):
    role_output = {**output, 'heads': metric_heads(output)}
    fused_loss, values = previous.previous.loss_values(args, role_output, labels, cameras, loss_fn)
    global_loss, global_values = previous.previous.original_loss_values(
        args, {**output, 'heads': output['global_heads']}, labels, cameras, loss_fn)
    values.update(global_loss=float(global_loss.detach()), fused_role_loss=float(fused_loss.detach()),
                  global_head_losses=global_values['head_losses'], role_metric_width=1536,
                  role_metric_norm_mean=float(output['fused'].detach().norm(dim=1).mean()))
    return global_loss + fused_loss, values


def condition(args):
    return {**original_condition(args), 'role_metric_policy': METRIC_POLICY}


def configure():
    previous.configure()
    base = previous.previous.base
    inner = base.entry.entry
    inner.loss_values = loss_values
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    configure()
    previous.previous.base.entry.entry.main()

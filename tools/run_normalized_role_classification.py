"""One RGBNT201 semantic control changing only the role classifier input."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.normalized_role_classification import NormalizedRoleClassification

SCHEMA = 'trifusion-normalized-role-classification-v1'
POLICY = 'global_raw_author_task_role_l2_before_detached_bn_classifier_metric_raw'
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core


def build_core(args, protocol):
    assert args.dataset == 'RGBNT201' and args.variant == 'semantic'
    model, cfg, binding = original_build_core(args, protocol)
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
                   objective_gradient_policy=POLICY,
                   role_classification_feature='l2_joint_1536_before_existing_detached_author_bn',
                   role_metric_feature='raw_joint_1536',
                   scope='RGBNT201 semantic only; no added head, parameters, losses or retrieval change. Real initialization/M0/training are not qualified by the synthetic head check.')
    return model, cfg, binding


def condition(args):
    return {**previous.condition(args), 'objective_gradient_policy': POLICY,
            'role_classification_feature': 'l2_joint_1536_before_existing_detached_author_bn',
            'role_metric_feature': 'raw_joint_1536'}


def configure():
    previous.configure()
    inner.AuthorHeadEvidence = NormalizedRoleClassification
    base = previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    inner.condition = condition
    inner.foundation.condition = condition


if __name__ == '__main__':
    configure()
    inner.main()

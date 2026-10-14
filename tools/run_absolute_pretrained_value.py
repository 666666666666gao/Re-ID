"""A fixed single-seed test of retaining common value content in role reading."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pretrained_value_delta as previous
from trifusion.absolute_pretrained_value import AbsolutePretrainedValueTriFusion

SCHEMA = 'trifusion-absolute-pretrained-value-v1'
original_condition = previous.condition


def build_core(args, protocol):
    model, cfg, binding = previous.build_core(args, protocol)
    assert isinstance(model.evidence_model, AbsolutePretrainedValueTriFusion)
    binding.update(entry_sha256=previous.inner.runner.sha256(Path(__file__)),
        value_reading='absolute_distribution_retains_image_common_value',
        scope='Same parameters, public CLIP, global/role tasks and initial gain as value951 uniform-reference; restore each image common value to the existing role-selected reading. No scale/seed/head/loss/optimizer change.')
    return model, cfg, binding


def condition(args):
    return dict(original_condition(args), value_reading='absolute_distribution_retains_image_common_value')


def configure():
    previous.MODE = 'uniform_reference'
    previous.configure()
    previous.inner.RawFeatureSemanticTriFusion = AbsolutePretrainedValueTriFusion
    base = previous.previous.previous.base
    base.SCHEMA = SCHEMA
    base.build_core = build_core
    base.configure()
    previous.inner.condition = condition
    previous.inner.foundation.condition = condition


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--value-reference', choices=('uniform_reference',), required=True)
    _, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    configure()
    previous.inner.main()

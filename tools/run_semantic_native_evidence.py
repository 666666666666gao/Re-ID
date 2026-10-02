"""EV1 semantic/semantic-plus-native values using the sealed clean full50 loop."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_clean_clip_joint as clean
from trifusion.semantic_native_evidence import SemanticNativeTriFusion

SCHEMA = 'trifusion-semantic-native-evidence-v1'


def condition(args):
    return dict(clean.condition(args), value_source=args.variant,
                cnn_candidates=512, native_stem_parameters=93248 if args.variant == 'combined' else 0)


def build_core(args, protocol):
    clean.control.GlobalTokenTriFusion = partial(SemanticNativeTriFusion, value_source=args.variant)
    model, cfg, config, binding = clean.build_core(args, protocol)
    binding.update(architecture='semantic_native_evidence_v1', value_source=args.variant,
                   entry_sha256=clean.control.runner.sha256(Path(__file__)),
                   role_source_sha256=clean.control.runner.sha256(ROOT/'modeling/trifusion/semantic_native_evidence.py'))
    return model, cfg, config, binding


def build(args, protocol):
    model, cfg, config, binding = build_core(args, protocol)
    witness = json.loads(args.initialization.read_text())
    assert witness['schema'] == SCHEMA and witness['binding'] == binding
    binding.update(condition=condition(args),
                   initialization_witness_sha256=clean.control.runner.sha256(args.initialization))
    return model, cfg, config, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201','RGBNT100','MSVR310'), required=True)
    parser.add_argument('--mode', choices=('m0','train','evaluate'), required=True)
    parser.add_argument('--variant', choices=('semantic','combined'), required=True)
    for name in ('protocol','signal-source','clip-weight','initialization','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--epochs', type=int, choices=(50,), default=50)
    args = parser.parse_args()
    for name in ('protocol','signal_source','clip_weight','initialization','output_dir'):
        setattr(args, name, getattr(args,name).resolve())
    args.readout = 'roles'
    args.visual_update = 'low_lr'
    args.baseline_sha256 = clean.control.runner.sha256(args.clip_weight)
    protocol = clean.control.runner.read_protocol(args.protocol, args.dataset)
    clean.control.SCHEMA = SCHEMA
    clean.control.condition = condition
    clean.control.build = build
    clean.control.frozen_signal_digest = clean.frozen_signal_digest
    (clean.control.evaluate if args.mode == 'evaluate' else clean.train)(args, protocol)


if __name__ == '__main__':
    main()

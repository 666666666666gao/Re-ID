"""Full50 matched shared/private evidence-flow training, using the verified visual loop."""
import argparse
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_visual_update_control as control
from trifusion.shared_private_evidence import SharedPrivateEvidenceCLIP

SCHEMA = 'trifusion-shared-private-evidence-v1'
FLOWS = ('coupled_roles', 'separated_roles', 'global_only')
BASE_BUILD = control.build


def condition(args, *, flow):
    return {'visual_update': 'low_lr',
            'readout': 'global_only' if flow == 'global_only' else 'roles',
            'visual_lr': control.VISUAL_LR, 'visual_parameter_dtype': 'float32',
            'evidence_flow': flow}


def build(args, protocol, *, flow):
    assert args.visual_update == 'low_lr'
    assert args.readout == ('global_only' if flow == 'global_only' else 'roles')
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    if flow != 'global_only':
        model.backbone = SharedPrivateEvidenceCLIP(
            model.backbone, private_writeback=flow == 'coupled_roles'
        )
    binding.update(architecture='shared_private_evidence_v1', condition=condition(args, flow=flow),
        common_initializer_scope='Original signal/shared-adapter backbone, neck and classifier; private adapters excluded',
        initial_model_state_sha256=control.runner._module_state_sha256(model),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        entry_sha256=control.runner.sha256(Path(__file__)),
        evidence_flow_source_sha256=control.runner.sha256(ROOT / 'modeling/trifusion/shared_private_evidence.py'))
    return model, cfg, config, binding


def activate(flow):
    assert flow in FLOWS
    control.SCHEMA = SCHEMA
    control.condition = partial(condition, flow=flow)
    control.build = partial(build, flow=flow)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201', 'RGBNT100', 'MSVR310'), required=True)
    parser.add_argument('--mode', choices=('m0', 'train', 'evaluate'), required=True)
    parser.add_argument('--evidence-flow', choices=FLOWS, required=True)
    for name in ('protocol', 'signal-source', 'clip-weight', 'baseline-checkpoint', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--epochs', type=int, choices=(50,), default=50)
    args = parser.parse_args()
    args.visual_update = 'low_lr'
    args.readout = 'global_only' if args.evidence_flow == 'global_only' else 'roles'
    for name in ('protocol', 'signal_source', 'clip_weight', 'baseline_checkpoint', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    activate(args.evidence_flow)
    protocol = control.runner.read_protocol(args.protocol, args.dataset)
    (control.evaluate if args.mode == 'evaluate' else control.train)(args, protocol)


if __name__ == '__main__':
    main()

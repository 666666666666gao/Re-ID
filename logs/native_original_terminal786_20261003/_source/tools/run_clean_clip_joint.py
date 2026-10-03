"""Clean public-CLIP joint global/role controls; reuse the sealed full50 loop."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_visual_update_control as control
from tools.build_v12_complete_path_oof_targets import _build_signal_teacher
from tools.official_three_dataset_model import ROLE_CONFIGS
from tools.run_signal_baseline_dev import _configure_signal_source

SCHEMA = 'trifusion-clean-public-clip-joint-v1'
CAMERA_KEY = 'clip_vision_encoder.cv_embed'


def tensor_digest(value):
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def condition(args):
    return {'visual_update': 'low_lr', 'readout': args.readout,
            'visual_lr': control.VISUAL_LR, 'visual_parameter_dtype': 'float32',
            'initialization': 'public_CLIP_fresh_camera_heads_modules',
            'public_clip_sha256': args.baseline_sha256,
            'initialization_witness_sha256': control.runner.sha256(args.initialization),
            'camera_update': 'new_module_lr_0.00035'}


def build_core(args, protocol):
    """Construct directly from public weights. Never load a ReID model state."""
    runner = control.runner
    runner._set_seed(args.seed)
    source_commit = _configure_signal_source(args.signal_source)
    from config import cfg
    cfg.defrost()
    cfg.merge_from_file(str(args.signal_source/'configs'/args.dataset/'Signal.yml'))
    cfg.MODEL.PRETRAIN_PATH_T = str(args.clip_weight)
    cfg.MODEL.USE_A = False
    cfg.MODEL.USE_B = False
    cfg.SOLVER.SEED = args.seed
    cfg.freeze()
    grid = (16, 8) if args.dataset == 'RGBNT201' else (8, 16)
    assert cfg.DATASETS.NAMES == args.dataset
    assert tuple(cfg.INPUT.SIZE_TRAIN) == tuple(cfg.INPUT.SIZE_TEST) == tuple(size*16 for size in grid)
    assert cfg.MODEL.SIE_CAMERA and not cfg.MODEL.SIE_VIEW
    cameras = sorted({row['camera'] for row in protocol['records']['train']})
    assert cameras == list(range(4 if args.dataset == 'RGBNT201' else 8))
    signal = _build_signal_teacher(cfg, num_classes=len(protocol['train_label_map']),
                                   camera_num=len(cameras), view_num=0)
    assert not hasattr(signal, 'SIM') and not hasattr(signal, 'AlignM')
    visual = signal.clip_vision_encoder.base.float()
    assert len(list(visual.parameters())) == 152
    assert all(p.dtype == torch.float32 for p in visual.parameters())
    public = torch.jit.load(str(args.clip_weight), map_location='cpu').state_dict()
    public_visual = {name.removeprefix('visual.'): value for name, value in public.items()
                     if name.startswith('visual.')}
    actual_visual = visual.state_dict()
    assert len(public_visual) == 152 and set(public_visual) == set(actual_visual)
    for name, value in public_visual.items():
        if name == 'positional_embedding':
            # The pinned author constructor resizes on CUDA, not CPU.
            value = value.to(actual_visual[name].device)
            patches = value[1:].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
            patches = F.interpolate(patches, size=grid, mode='bilinear')
            value = torch.cat((value[:1], patches.permute(0, 2, 3, 1).reshape(128, 768)))
        assert torch.equal(actual_visual[name].detach().cpu(), value.float().cpu()), name
    camera = signal.clip_vision_encoder.cv_embed
    assert tuple(camera.shape) == (len(cameras), 1, 768) and bool(torch.isfinite(camera).all())
    camera_initial = tensor_digest(camera)
    runner._set_seed(args.seed)
    initialized = control.GlobalTokenTriFusion(
        signal, num_classes=len(protocol['train_label_map']), grid=grid, width=128,
        m1=True, m2=True, m3=False, mamba_factory=runner.production_mamba_factory,
        token_mode='static', query_mode='context', auxiliary_target='none',
    ).cuda()
    common = {name: runner._module_state_sha256(getattr(initialized, name))
              for name in ('backbone', 'neck', 'classifier')}
    model = control.SharedGlobalOnly(initialized) if args.readout == 'global_only' else initialized
    visual.requires_grad_(True)
    camera.requires_grad_(True)
    assert common == {name: runner._module_state_sha256(getattr(model, name)) for name in common}
    config = json.loads((ROOT/ROLE_CONFIGS[args.dataset]).read_text(encoding='utf-8'))
    assert config['OPTIMIZATION']['NEW_MODULE_LR'] == 0.00035
    assert config['OPTIMIZATION']['WEIGHT_DECAY'] == 0.0001
    binding = {'architecture': 'clean_public_clip_joint_v1', 'dataset': args.dataset,
               'seed': args.seed, 'readout': args.readout, 'public_clip_path': str(args.clip_weight),
               'public_clip_sha256': args.baseline_sha256, 'public_visual_tensors_verified': 152,
               'visual_initial_sha256': runner._module_state_sha256(visual),
               'fresh_camera_initial_sha256': camera_initial, 'fresh_camera_shape': list(camera.shape),
               'author_source_commit': source_commit, 'protocol_sha256': runner.sha256(args.protocol),
               'common_initializer_sha256': common,
               'initial_model_state_sha256': runner._module_state_sha256(model),
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
               'trainable_parameter_tensors': sum(p.requires_grad for p in model.parameters()),
               'entry_sha256': runner.sha256(Path(__file__)),
               'baseline_sha256_semantics': 'Public CLIP input file only; no trained ReID baseline is read.',
               'camera_semantics': 'Fresh author truncated-normal embedding, trainable at new-module LR.',
               'initialization_scope': 'public visual only; fresh camera, unused Signal state, shared adapters, roles and retrieval heads'}
    return model, cfg, config, binding


def build(args, protocol):
    model, cfg, config, binding = build_core(args, protocol)
    witness = json.loads(args.initialization.read_text())
    assert witness['schema'] == SCHEMA and witness['binding'] == binding
    binding.update(condition=condition(args), initialization_witness_sha256=control.runner.sha256(args.initialization))
    return model, cfg, config, binding


def frozen_signal_digest(model, update):
    assert update == 'low_lr'
    digest = hashlib.sha256()
    for name, value in sorted(model.backbone.signal.state_dict().items()):
        if name.startswith(control.VISUAL_PREFIX) or name == CAMERA_KEY:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    CAMERA_OBSERVATIONS.append(tensor_digest(model.backbone.signal.clip_vision_encoder.cv_embed))
    return digest.hexdigest()


CAMERA_OBSERVATIONS = []


def train(args, protocol):
    control.train(args, protocol)
    assert len(CAMERA_OBSERVATIONS) == 2 and CAMERA_OBSERVATIONS[0] != CAMERA_OBSERVATIONS[1]
    receipt_path = args.output_dir/'training.json'
    receipt = json.loads(receipt_path.read_text())
    receipt.update(public_clip_sha256=args.baseline_sha256,
                   baseline_sha256_semantics='Public CLIP input file only; no trained ReID baseline loaded.',
                   fresh_camera_parameters_changed=True,
                   camera_state_before_sha256=CAMERA_OBSERVATIONS[0],
                   camera_state_after_sha256=CAMERA_OBSERVATIONS[1])
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201','RGBNT100','MSVR310'), required=True)
    parser.add_argument('--mode', choices=('prepare','m0','train','evaluate'), required=True)
    parser.add_argument('--readout', choices=('global_only','roles'), required=True)
    for name in ('protocol','signal-source','clip-weight','initialization','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--epochs', type=int, choices=(50,), default=50)
    args = parser.parse_args()
    for name in ('protocol','signal_source','clip_weight','initialization','output_dir'):
        setattr(args, name, getattr(args,name).resolve())
    args.visual_update = 'low_lr'
    # The reused loop's legacy field binds the public file, never a ReID checkpoint.
    args.baseline_sha256 = control.runner.sha256(args.clip_weight)
    protocol = control.runner.read_protocol(args.protocol, args.dataset)
    if args.mode == 'prepare':
        assert not args.initialization.exists()
        model, _cfg, _config, binding = build_core(args, protocol)
        args.initialization.parent.mkdir(parents=True, exist_ok=True)
        args.initialization.write_text(json.dumps({'schema':SCHEMA,'status':'FRESH_PUBLIC_INITIALIZATION_VERIFIED',
            'prepared_at':datetime.now().astimezone().isoformat(),'binding':binding,
            'boundary':'Construction/value parity only; no neural forward, optimizer, M0 or retrieval.'},indent=2)+'\n')
        print(json.dumps({'event':'initialization','dataset':args.dataset,'readout':args.readout,
                          'path':str(args.initialization),'binding':binding}),flush=True)
        return
    control.SCHEMA = SCHEMA
    control.condition = condition
    control.build = build
    control.frozen_signal_digest = frozen_signal_digest
    (control.evaluate if args.mode == 'evaluate' else train)(args, protocol)


if __name__ == '__main__':
    main()

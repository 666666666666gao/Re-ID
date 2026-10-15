"""Initialize the common V8 role model from our exact full50 F1-author best."""
import hashlib
import json
from pathlib import Path

from tools.official_three_dataset_model import ROLE_CONFIGS, sha256


def build_model(protocol, source, clip_weight, manifest, *, seed=42):
    import torch
    from tools.build_v12_complete_path_oof_targets import _build_signal_teacher, _build_v8_experts
    from tools.run_signal_baseline_dev import _configure_signal_source
    from tools.run_signal_preserving_v5 import ARCHITECTURE_V8, _module_state_sha256, _set_seed

    name = protocol['dataset']
    assert name in ('RGBNT100', 'MSVR310')
    manifest = Path(manifest)
    record = next(row for row in json.loads(manifest.read_bytes())['rows'] if row['dataset'] == name)
    semantic = {key: value for key, value in protocol.items() if key != 'dataset_root'}
    semantic_digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert semantic_digest == record['semantic_protocol_sha256']
    path = Path(record['checkpoint_path'])
    assert sha256(path) == record['checkpoint_sha256']
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'trifusion-foundation-recipe-v1' and payload['dataset'] == name
    assert payload['condition']['recipe'] == 'author' and payload['condition']['seed'] == 42
    assert payload['condition']['epochs'] == 50
    assert payload['protocol_sha256'] == record['source_protocol_sha256']
    assert payload['epoch'] == record['selected_epoch'] and payload['metrics'] == record['baseline_metrics']
    assert all(key.startswith('signal.') for key in payload['state'])
    state = {key.removeprefix('signal.'): value for key, value in payload['state'].items()}
    assert not any(part in ('SIM', 'AlignM') for key in state for part in key.split('.'))

    _set_seed(seed)
    source = Path(source)
    _configure_signal_source(source)
    from config import cfg
    cfg.defrost()
    cfg.merge_from_file(str(source/'configs'/name/'Signal.yml'))
    cfg.MODEL.PRETRAIN_PATH_T = str(clip_weight)
    cfg.SOLVER.SEED = seed
    cfg.MODEL.USE_A = False
    cfg.MODEL.USE_B = False
    cfg.freeze()
    assert not cfg.MODEL.ADAPTER and not cfg.MODEL.PROMPT
    assert cfg.DATASETS.NAMES == name
    cameras = sorted({row['camera'] for row in protocol['records']['train']})
    assert cameras == list(range(4 if name == 'RGBNT201' else 8))
    signal = _build_signal_teacher(cfg, num_classes=len(protocol['train_label_map']), camera_num=len(cameras), view_num=0)
    assert not hasattr(signal, 'SIM') and not hasattr(signal, 'AlignM')
    assert set(state) == set(signal.state_dict())
    signal.load_state_dict(state, strict=True)
    signal_digest = _module_state_sha256(signal)
    del state, payload

    root = Path(__file__).resolve().parents[1]
    role_config = json.loads((root/ROLE_CONFIGS[name]).read_text(encoding='utf-8'))
    role_config['DATA']['DATASET_ROOT'] = protocol['dataset_root']
    role_config['SIGNAL'] = dict(SOURCE=str(source), CONFIG=str(source/'configs'/name/'Signal.yml'),
                                 CLIP_WEIGHT=str(clip_weight), BASELINE_WIDTH=1536)
    role_config['MODEL']['ARCHITECTURE'] = ARCHITECTURE_V8
    role_config['OPTIMIZATION']['MAX_EPOCHS'] = 50
    assert role_config['MODEL']['GRID_SIZE'] == ([16, 8] if name == 'RGBNT201' else [8, 16])
    model = _build_v8_experts(signal, role_config, signal_checkpoint_sha256=record['checkpoint_sha256'],
                              num_classes=len(protocol['train_label_map']), use_sim=False)
    assert model.baseline_embedding_width == 1536 and model.fused_embedding_width == 6144
    assert all(not parameter.requires_grad for parameter in model.baseline.parameters())
    binding = dict(baseline_kind='our_F1_author_full50_no_modules',
                   foundation_checkpoint=str(path), foundation_checkpoint_sha256=record['checkpoint_sha256'],
                   foundation_epoch=record['selected_epoch'], foundation_training_epochs=50,
                   foundation_source_protocol_sha256=record['source_protocol_sha256'],
                   semantic_protocol_sha256=semantic_digest, baseline_metrics=record['baseline_metrics'],
                   signal_state_sha256=signal_digest, initial_role_state_sha256=_module_state_sha256(model),
                   baseline_width=1536, fused_width=6144)
    return model, cfg, role_config, binding

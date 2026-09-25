"""Strictly initialize the original V8 roles from a dataset's author Signal weights."""

import hashlib
import json
from pathlib import Path


ROLE_CONFIGS = {
    "RGBNT201": "configs/RGBNT201/TriFusion-signal-preserving-v27-source-style-rtx3090.json",
    "RGBNT100": "configs/RGBNT100/TriFusion-main-v1.json",
    "MSVR310": "configs/MSVR310/TriFusion-source-style-paired-v1-r2.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(protocol, source, clip_weight, checkpoint, expected_sha256, *, seed=42,
                plain_baseline=False):
    import torch

    from tools.build_v12_complete_path_oof_targets import _build_signal_teacher, _build_v8_experts
    from tools.run_signal_baseline_dev import _configure_signal_source
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed

    name = protocol["dataset"]
    _set_seed(seed)
    source = Path(source)
    _configure_signal_source(source)
    from config import cfg

    cfg.merge_from_file(str(source / "configs" / name / "Signal.yml"))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_PATH_T = str(clip_weight)
    cfg.SOLVER.SEED = seed
    if plain_baseline:
        cfg.MODEL.USE_A = False
        cfg.MODEL.USE_B = False
    cfg.freeze()
    assert cfg.DATASETS.NAMES == name
    assert list(cfg.INPUT.SIZE_TRAIN) == list(cfg.INPUT.SIZE_TEST) == (
        [256, 128] if name == "RGBNT201" else [128, 256])
    assert cfg.MODEL.SIE_CAMERA and not cfg.MODEL.SIE_VIEW
    if name == "RGBNT100" and not plain_baseline:
        from modeling.AddModule import useB
        from tools.signal_gram_stable import signal_gram_volume_stable
        useB.volume_computation3 = signal_gram_volume_stable
    path = Path(checkpoint)
    digest = sha256(path)
    assert digest == expected_sha256
    train = protocol["records"]["train"]
    cameras = sorted({row["camera"] for row in train})
    assert cameras == list(range(4 if name == "RGBNT201" else 8))
    signal = _build_signal_teacher(cfg, num_classes=len(protocol["train_label_map"]),
                                   camera_num=len(cameras), view_num=0)
    if plain_baseline:
        assert not hasattr(signal, "SIM") and not hasattr(signal, "AlignM")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if name == "RGBNT100" and not plain_baseline and all(
        key.startswith("module.") for key in state
    ):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    assert set(state) == set(signal.state_dict())
    signal.load_state_dict(state, strict=True)
    signal_hash = _module_state_sha256(signal)
    del state
    root = Path(__file__).resolve().parents[1]
    role_config = json.loads((root / ROLE_CONFIGS[name]).read_text(encoding="utf-8"))
    if name == "RGBNT201":
        from tools.run_signal_preserving_v5 import ARCHITECTURE_V8
        role_config["MODEL"]["ARCHITECTURE"] = ARCHITECTURE_V8
    expected_grid = [16, 8] if name == "RGBNT201" else [8, 16]
    assert role_config["MODEL"]["GRID_SIZE"] == expected_grid
    assert role_config["OPTIMIZATION"]["MAX_EPOCHS"] == 20
    model = _build_v8_experts(signal, role_config, signal_checkpoint_sha256=digest,
                              num_classes=len(protocol["train_label_map"]),
                              use_sim=not plain_baseline)
    assert all(not parameter.requires_grad for parameter in model.baseline.parameters())
    binding = dict(author_checkpoint=str(path), author_checkpoint_sha256=digest,
                   signal_state_sha256=signal_hash,
                   initial_role_state_sha256=_module_state_sha256(model))
    if plain_baseline:
        binding["baseline_kind"] = "independently_trained_module_free_cls"
    return model, cfg, role_config, binding

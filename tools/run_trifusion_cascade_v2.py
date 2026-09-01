#!/usr/bin/env python3
"""Isolated V2 launcher that reuses, but never mutates, the frozen V1 runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_trainable_parameters(
    model: Any,
    *,
    family: str,
    architecture: str = "legacy_parallel",
) -> tuple[list[Any], list[Any]]:
    if architecture != "shared_semantic_cascade_v2":
        raise ValueError("V2 launcher accepts only the cascade V2 architecture")
    if family != "collaborative":
        raise ValueError("cascade V2 requires the collaborative model family")
    pretrained_tokens = (
        "encoder.tokenizer.patch_projection",
        "encoder.tokenizer.positional_embedding",
        "encoder.tokenizer.class_embedding",
        "encoder.tokenizer.pre_norm",
        "encoder.tokenizer.shared_blocks",
        "encoder.tokenizer.post_norm",
        "fusion.semantic_projection",
    )
    pretrained_parameters = []
    new_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            pretrained_parameters
            if any(token in name for token in pretrained_tokens)
            else new_parameters
        )
        target.append(parameter)
    return pretrained_parameters, new_parameters


def _config_path(argv: list[str]) -> Path:
    if "--config" not in argv:
        raise ValueError("cascade V2 launcher requires an explicit config")
    index = argv.index("--config")
    if index + 1 >= len(argv):
        raise ValueError("cascade V2 config path is missing")
    path = Path(argv[index + 1]).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _install_v2_dispatch(config: dict[str, Any]) -> Any:
    import modeling.trifusion.builder as builder_module
    import modeling.trifusion.criterion as criterion_module
    import tools.run_trifusion_experiment as runner
    from modeling.trifusion.cascade_v2 import CascadeV2Criterion
    from modeling.trifusion.cascade_v2_builder import (
        build_trifusion_cascade_v2_from_clip,
    )

    if config.get("MODEL", {}).get("ARCHITECTURE") != "shared_semantic_cascade_v2":
        raise ValueError("V2 launcher config has the wrong architecture")
    original_builder = builder_module.build_trifusion_from_clip

    def build_dispatch(checkpoint: Path | str, **kwargs: Any):
        if kwargs.get("architecture") == "shared_semantic_cascade_v2":
            return build_trifusion_cascade_v2_from_clip(checkpoint, **kwargs)
        return original_builder(checkpoint, **kwargs)

    loss_config = dict(config.get("LOSS", {}))

    class ConfiguredCascadeV2Criterion(CascadeV2Criterion):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                **kwargs,
                label_smoothing=float(loss_config.get("LABEL_SMOOTHING", 0.0)),
                effect_rank_weight=float(
                    loss_config.get("EFFECT_RANK_WEIGHT", 0.0)
                ),
                effect_rank_margin=float(
                    loss_config.get("EFFECT_RANK_MARGIN", 0.05)
                ),
                effect_rank_min_gap=float(
                    loss_config.get("EFFECT_RANK_MIN_GAP", 0.0)
                ),
            )

    original_source_hashes = runner.trifusion_source_hashes

    def v2_source_hashes() -> dict[str, str]:
        hashes = dict(original_source_hashes())
        wrapper = Path(__file__).resolve()
        hashes[str(wrapper.relative_to(PROJECT))] = _sha256(wrapper)
        return hashes

    builder_module.build_trifusion_from_clip = build_dispatch
    criterion_module.TriFusionCriterion = ConfiguredCascadeV2Criterion
    runner._partition_trainable_parameters = _partition_trainable_parameters
    runner.trifusion_source_hashes = v2_source_hashes
    runner.__file__ = str(Path(__file__).resolve())
    return runner


def main() -> int:
    config_path = _config_path(sys.argv[1:])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("cascade V2 config must contain a mapping")
    runner = _install_v2_dispatch(config)
    return int(runner.main())


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["_partition_trainable_parameters", "main"]

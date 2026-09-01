#!/usr/bin/env python3
"""Launch the isolated task-anchored V3 experiment through the audited runner."""

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
    if family != "collaborative" or architecture != "task_anchored_collaborative_v3":
        raise ValueError("V3 launcher accepts only its collaborative architecture")
    pretrained_prefixes = (
        "tokenizer.patch_projection",
        "tokenizer.positional_embedding",
        "tokenizer.class_embedding",
        "tokenizer.pre_norm",
        "tokenizer.shared_blocks",
        "tokenizer.post_norm",
        "tokenizer.output_projection",
        "fusion.residual_projections",
    )
    pretrained = []
    new = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (pretrained if name.startswith(pretrained_prefixes) else new).append(parameter)
    return pretrained, new


def _config_path(argv: list[str]) -> Path:
    if "--config" not in argv:
        raise ValueError("V3 launcher requires an explicit config")
    index = argv.index("--config")
    if index + 1 >= len(argv):
        raise ValueError("V3 config path is missing")
    value = Path(argv[index + 1]).expanduser()
    if not value.is_absolute():
        value = PROJECT / value
    value = value.resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _install_v3_dispatch(config: dict[str, Any]) -> Any:
    import modeling.trifusion.builder as builder_module
    import modeling.trifusion.criterion as criterion_module
    import tools.run_trifusion_experiment as runner
    from modeling.trifusion.task_anchor_v3 import TaskAnchoredV3Criterion
    from modeling.trifusion.task_anchor_v3_builder import (
        build_task_anchored_trifusion_v3_from_clip,
    )

    architecture = config.get("MODEL", {}).get("ARCHITECTURE")
    if architecture != "task_anchored_collaborative_v3":
        raise ValueError("V3 launcher config has the wrong architecture")
    original_builder = builder_module.build_trifusion_from_clip

    def build_dispatch(checkpoint: Path | str, **kwargs: Any):
        if kwargs.get("architecture") == architecture:
            return build_task_anchored_trifusion_v3_from_clip(
                checkpoint,
                residual_scale_init=float(
                    config["MODEL"].get("RESIDUAL_SCALE_INIT", 0.25)
                ),
                **kwargs,
            )
        return original_builder(checkpoint, **kwargs)

    loss = dict(config.get("LOSS", {}))

    class ConfiguredTaskAnchoredV3Criterion(TaskAnchoredV3Criterion):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                **kwargs,
                label_smoothing=float(loss.get("LABEL_SMOOTHING", 0.1)),
                alignment_temperature=float(loss.get("ALIGNMENT_TEMPERATURE", 0.07)),
            )

    original_source_hashes = runner.trifusion_source_hashes

    def v3_source_hashes() -> dict[str, str]:
        hashes = dict(original_source_hashes())
        for path in (
            Path(__file__).resolve(),
            PROJECT / "modeling/trifusion/task_anchor_v3.py",
            PROJECT / "modeling/trifusion/task_anchor_v3_builder.py",
        ):
            hashes[str(path.relative_to(PROJECT))] = _sha256(path)
        return hashes

    builder_module.build_trifusion_from_clip = build_dispatch
    criterion_module.TriFusionCriterion = ConfiguredTaskAnchoredV3Criterion
    runner._partition_trainable_parameters = _partition_trainable_parameters
    runner.trifusion_source_hashes = v3_source_hashes
    runner.__file__ = str(Path(__file__).resolve())
    return runner


def main() -> int:
    config_path = _config_path(sys.argv[1:])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V3 config must contain a mapping")
    return int(_install_v3_dispatch(config).main())


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["_partition_trainable_parameters", "main"]

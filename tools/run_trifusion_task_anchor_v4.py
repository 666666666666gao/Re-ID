#!/usr/bin/env python3
"""Launch task-anchored V4 through the audited experiment runner."""

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
    if family != "collaborative" or architecture != "task_anchored_collaborative_v4":
        raise ValueError("V4 launcher accepts only its collaborative architecture")
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
        raise ValueError("V4 launcher requires an explicit config")
    index = argv.index("--config")
    if index + 1 >= len(argv):
        raise ValueError("V4 config path is missing")
    value = Path(argv[index + 1]).expanduser()
    if not value.is_absolute():
        value = PROJECT / value
    value = value.resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _install_v4_dispatch(config: dict[str, Any]) -> Any:
    import modeling.trifusion.builder as builder_module
    import modeling.trifusion.criterion as criterion_module
    import tools.run_trifusion_experiment as runner
    from modeling.trifusion.task_anchor_v4 import TaskAnchoredV4Criterion
    from modeling.trifusion.task_anchor_v4_builder import (
        build_task_anchored_trifusion_v4_from_clip,
    )

    architecture = config.get("MODEL", {}).get("ARCHITECTURE")
    if architecture != "task_anchored_collaborative_v4":
        raise ValueError("V4 launcher config has the wrong architecture")
    original_builder = builder_module.build_trifusion_from_clip

    def build_dispatch(checkpoint: Path | str, **kwargs: Any):
        if kwargs.get("architecture") == architecture:
            return build_task_anchored_trifusion_v4_from_clip(checkpoint, **kwargs)
        return original_builder(checkpoint, **kwargs)

    loss = dict(config.get("LOSS", {}))

    class ConfiguredTaskAnchoredV4Criterion(TaskAnchoredV4Criterion):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                **kwargs,
                label_smoothing=float(loss.get("LABEL_SMOOTHING", 0.1)),
                alignment_temperature=float(loss.get("ALIGNMENT_TEMPERATURE", 0.07)),
            )

    original_source_hashes = runner.trifusion_source_hashes

    # V4's capacity probe recovered one CLIP-trunk overflow at 512; register the
    # measured-stable initial scaler value so the fail-closed profile stays exact.
    runner.RESOURCE_PROFILES["rtx3090_b32k4"] = {
        **runner.RESOURCE_PROFILES["rtx3090_b32k4"],
        "amp_init_scale": 256.0,
    }

    def v4_source_hashes() -> dict[str, str]:
        hashes = dict(original_source_hashes())
        for path in (
            Path(__file__).resolve(),
            PROJECT / "modeling/trifusion/task_anchor_v3.py",
            PROJECT / "modeling/trifusion/task_anchor_v3_builder.py",
            PROJECT / "modeling/trifusion/task_anchor_v4.py",
            PROJECT / "modeling/trifusion/task_anchor_v4_builder.py",
        ):
            hashes[str(path.relative_to(PROJECT))] = _sha256(path)
        return hashes

    builder_module.build_trifusion_from_clip = build_dispatch
    criterion_module.TriFusionCriterion = ConfiguredTaskAnchoredV4Criterion
    runner._partition_trainable_parameters = _partition_trainable_parameters
    runner.trifusion_source_hashes = v4_source_hashes
    runner.__file__ = str(Path(__file__).resolve())
    return runner


def main() -> int:
    config_path = _config_path(sys.argv[1:])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V4 config must contain a mapping")
    return int(_install_v4_dispatch(config).main())


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["_partition_trainable_parameters", "main"]

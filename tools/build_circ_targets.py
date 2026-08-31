#!/usr/bin/env python3
"""Compile externally scored full-network CIRC interventions into a cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from modeling.trifusion.intervention_targets import (
    compile_circ_targets,
    write_circ_target_cache,
)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    config_sha256 = hashlib.sha256(payload).hexdigest()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        loaded = yaml.safe_load(payload.decode("utf-8"))
    else:
        loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("CIRC config root must be a mapping")
    return loaded, config_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("development", "postfreeze-final"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config, config_sha256 = _load_config(arguments.config)
    rows, receipt = compile_circ_targets(
        config, mode=arguments.mode, config_sha256=config_sha256
    )
    write_circ_target_cache(
        rows, receipt, output_directory=arguments.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

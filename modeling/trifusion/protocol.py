"""Trust anchor for the preregistered CIRC development protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CIRC_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "protocols/circ_target_v1.json"
)
CIRC_PROTOCOL_SHA256 = (
    "3674cdd4716a80e783e5f30993ff8f0f9cb5bbaebda1d22fcf0b9b91400dede5"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_circ_protocol(path: Path | str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    if (
        resolved != CIRC_PROTOCOL_PATH.resolve()
        or not resolved.is_file()
        or sha256_file(resolved) != CIRC_PROTOCOL_SHA256
    ):
        raise ValueError("requires the trusted frozen CIRC protocol")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "circ-protocol-v1":
        raise ValueError("trusted CIRC protocol has an unsupported schema")
    return payload, CIRC_PROTOCOL_SHA256


def trifusion_source_hashes() -> dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    paths = sorted((project / "modeling/trifusion").rglob("*.py"))
    paths.extend(
        [
            project / "tools/build_circ_targets.py",
            project / "tools/run_trifusion_experiment.py",
        ]
    )
    return {
        str(path.relative_to(project)): sha256_file(path)
        for path in paths
    }


__all__ = [
    "CIRC_PROTOCOL_PATH",
    "CIRC_PROTOCOL_SHA256",
    "load_trusted_circ_protocol",
    "sha256_file",
    "trifusion_source_hashes",
]

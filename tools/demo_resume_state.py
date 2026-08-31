#!/usr/bin/env python3
"""Atomic, fail-closed epoch-boundary state for DeMo reproduction runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
VALID_PHASES = frozenset({"post_train", "post_eval"})
BEST_INDEX_KEYS = ("mAP", "Rank-1", "Rank-5", "Rank-10")


@dataclass(frozen=True)
class RunIdentity:
    """Inputs whose equality is required before a run may be resumed."""

    baseline_commit: str
    config_sha256: str
    clip_sha256: str
    recovery_code_sha256: str
    python_version: str
    torch_version: str
    cuda_version: str
    device_name: str
    runtime_sha256: str
    parity_epoch: int
    parity_reference_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ResumeCursor:
    """The last durable epoch boundary returned to the training driver."""

    epoch: int
    phase: str
    best_index: dict[str, float]


@dataclass(frozen=True)
class ResumeAction:
    """One unambiguous action selected from a durable boundary."""

    kind: str
    epoch: int


def next_resume_action(
    cursor: ResumeCursor, *, max_epochs: int, eval_period: int
) -> ResumeAction:
    """Map a durable boundary to exactly one legal driver action."""

    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if eval_period < 1:
        raise ValueError("eval_period must be positive")
    if cursor.epoch > max_epochs:
        raise ValueError(
            f"checkpoint epoch {cursor.epoch} exceeds max_epochs {max_epochs}"
        )
    if cursor.phase == "post_train":
        if cursor.epoch < 1:
            raise ValueError("post_train is invalid at epoch zero")
        kind = "evaluate" if cursor.epoch % eval_period == 0 else "finalize"
        return ResumeAction(kind=kind, epoch=cursor.epoch)
    if cursor.phase != "post_eval":
        raise ValueError(f"invalid resume cursor phase: {cursor.phase!r}")
    if cursor.epoch == max_epochs:
        return ResumeAction(kind="complete", epoch=cursor.epoch)
    return ResumeAction(kind="train", epoch=cursor.epoch + 1)


def _validate_boundary(
    epoch: int, phase: str, best_index: Mapping[str, float]
) -> dict[str, float]:
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}")
    if phase not in VALID_PHASES:
        raise ValueError(f"invalid checkpoint phase: {phase!r}")
    if epoch == 0 and phase == "post_train":
        raise ValueError("post_train is invalid at epoch zero")
    if set(best_index) != set(BEST_INDEX_KEYS):
        raise ValueError(
            "best_index must contain exactly "
            f"{sorted(BEST_INDEX_KEYS)}, got {sorted(best_index)}"
        )
    normalized = {key: float(best_index[key]) for key in BEST_INDEX_KEYS}
    if not all(np.isfinite(value) for value in normalized.values()):
        raise ValueError("best_index values must be finite")
    return normalized


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected:
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from checkpoint RNG state: "
                f"{torch.cuda.device_count()} != {len(cuda_state)}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _validate_adam_state(
    saved_optimizer: Mapping[str, Any],
    optimizer: Any,
    *,
    epoch: int,
) -> None:
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError(
            f"DeMo recovery requires Adam, got {type(optimizer).__name__}"
        )
    live_groups = optimizer.param_groups
    saved_groups = saved_optimizer["param_groups"]
    state_dict_groups = optimizer.state_dict()["param_groups"]
    if len(saved_groups) != len(live_groups) or len(saved_groups) != len(
        state_dict_groups
    ):
        raise ValueError("checkpoint Adam parameter-group count is invalid")

    parameter_by_id = {}
    group_by_id = {}
    saved_parameter_ids = []
    for saved_group, live_group, state_dict_group in zip(
        saved_groups, live_groups, state_dict_groups
    ):
        if set(saved_group) != set(state_dict_group):
            raise ValueError(
                "checkpoint Adam parameter-group keys are incomplete or unexpected"
            )
        saved_ids = saved_group["params"]
        live_parameters = live_group["params"]
        if len(saved_ids) != len(live_parameters):
            raise ValueError("checkpoint Adam parameter-group size is invalid")
        if saved_ids != state_dict_group["params"]:
            raise ValueError(
                "checkpoint Adam parameter IDs differ from the live optimizer"
            )
        saved_parameter_ids.extend(saved_ids)
        parameter_by_id.update(zip(saved_ids, live_parameters))
        group_by_id.update((parameter_id, saved_group) for parameter_id in saved_ids)

    live_parameter_count = sum(len(group["params"]) for group in live_groups)
    if (
        len(saved_parameter_ids) != live_parameter_count
        or len(set(saved_parameter_ids)) != live_parameter_count
        or len(parameter_by_id) != live_parameter_count
    ):
        raise ValueError(
            "checkpoint Adam parameter IDs must be globally unique and complete"
        )

    saved_state = saved_optimizer["state"]
    if epoch == 0:
        if saved_state:
            raise ValueError("checkpoint Adam state must be empty at epoch zero")
        return
    if set(saved_state) != set(parameter_by_id):
        raise ValueError(
            "checkpoint Adam state does not cover every optimizer parameter"
        )

    for parameter_id, parameter_state in saved_state.items():
        parameter = parameter_by_id[parameter_id]
        group = group_by_id[parameter_id]
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad", False):
            expected_keys.add("max_exp_avg_sq")
        if not isinstance(parameter_state, Mapping) or set(
            parameter_state
        ) != expected_keys:
            raise ValueError(
                "checkpoint Adam state keys are incomplete or unexpected for "
                f"parameter {parameter_id}: expected={sorted(expected_keys)} "
                f"actual={sorted(parameter_state) if isinstance(parameter_state, Mapping) else type(parameter_state).__name__}"
            )
        step = parameter_state["step"]
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError("checkpoint Adam step must be scalar")
        elif not isinstance(step, (int, float)):
            raise ValueError("checkpoint Adam step has an invalid type")
        for moment_name in expected_keys - {"step"}:
            moment = parameter_state[moment_name]
            if not torch.is_tensor(moment) or moment.shape != parameter.shape:
                raise ValueError(
                    f"checkpoint Adam {moment_name} shape is invalid for "
                    f"parameter {parameter_id}"
                )


def _validate_nested_training_state(
    payload: Mapping[str, Any], *, optimizer: Any, scheduler: Any
) -> None:
    for state_name in (
        "model",
        "center_criterion",
        "scheduler",
        "scaler",
    ):
        state = payload[state_name]
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"checkpoint {state_name} state is empty or invalid")

    for optimizer_name in ("optimizer", "optimizer_center"):
        optimizer_state = payload[optimizer_name]
        if (
            not isinstance(optimizer_state, Mapping)
            or set(optimizer_state) != {"state", "param_groups"}
            or not isinstance(optimizer_state["state"], Mapping)
            or not isinstance(optimizer_state["param_groups"], list)
            or not optimizer_state["param_groups"]
        ):
            raise ValueError(
                f"checkpoint {optimizer_name} state is incomplete or invalid"
            )
    if payload["epoch"] > 0 and not payload["optimizer"]["state"]:
        raise ValueError("checkpoint optimizer state is empty after training")
    _validate_adam_state(
        payload["optimizer"], optimizer, epoch=payload["epoch"]
    )

    expected_scheduler_keys = set(scheduler.state_dict())
    if set(payload["scheduler"]) != expected_scheduler_keys:
        raise ValueError(
            "checkpoint scheduler state keys are incomplete or unexpected: "
            f"expected={sorted(expected_scheduler_keys)} "
            f"actual={sorted(payload['scheduler'])}"
        )

    rng_state = payload["rng"]
    if not isinstance(rng_state, Mapping) or set(rng_state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("checkpoint RNG state is incomplete")


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_bytes_save(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary_path.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_state_generation(
    payload: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    generation_name = (
        f".{manifest_path.name}.state-{uuid.uuid4().hex}.pt"
    )
    generation_path = manifest_path.parent / generation_name
    temporary_path = generation_path.with_name(
        f".{generation_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary_path.open("wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, generation_path)
        _fsync_directory(manifest_path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {
        "file": generation_name,
        "bytes": generation_path.stat().st_size,
        "sha256": _sha256_file(generation_path),
    }


def _state_path_from_record(
    manifest_path: Path,
    record: Mapping[str, Any],
    *,
    verify_content: bool,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "file",
        "bytes",
        "sha256",
    }:
        raise ValueError("checkpoint manifest state record is invalid")
    file_name = record["file"]
    expected_prefix = f".{manifest_path.name}.state-"
    if (
        not isinstance(file_name, str)
        or Path(file_name).name != file_name
        or not file_name.startswith(expected_prefix)
        or not file_name.endswith(".pt")
    ):
        raise ValueError("checkpoint manifest has an unsafe state filename")
    if (
        not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] < 1
    ):
        raise ValueError("checkpoint manifest has an invalid byte count")
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("checkpoint manifest has an invalid SHA-256")
    state_path = manifest_path.parent / file_name
    if verify_content:
        if not state_path.is_file():
            raise RuntimeError(f"checkpoint state file is missing: {state_path}")
        if state_path.stat().st_size != record["bytes"]:
            raise RuntimeError(
                f"checkpoint byte-count mismatch: {state_path}"
            )
        actual_digest = _sha256_file(state_path)
        if actual_digest != digest:
            raise RuntimeError(
                "checkpoint SHA-256 mismatch: "
                f"expected {digest}, got {actual_digest}"
            )
    return state_path


def _read_checkpoint_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(
            f"cannot read checkpoint manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "manifest_schema_version",
        "current",
        "previous",
    }:
        raise ValueError("checkpoint manifest fields are incomplete or unexpected")
    if manifest["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint manifest schema: "
            f"{manifest['manifest_schema_version']!r}"
        )
    current_path = _state_path_from_record(
        manifest_path, manifest["current"], verify_content=True
    )
    previous = manifest["previous"]
    if previous is not None:
        _state_path_from_record(
            manifest_path, previous, verify_content=False
        )
    return manifest, current_path


def _publish_checkpoint_manifest(
    manifest_path: Path,
    *,
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> None:
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "current": dict(current),
        "previous": dict(previous) if previous is not None else None,
    }
    encoded = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=True
    ).encode("utf-8") + b"\n"
    _atomic_bytes_save(encoded, manifest_path)

    retained = {current["file"]}
    if previous is not None:
        retained.add(previous["file"])
    generation_pattern = f".{manifest_path.name}.state-*.pt"
    for candidate in manifest_path.parent.glob(generation_pattern):
        if candidate.name not in retained and candidate.is_file():
            candidate.unlink()


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    phase: str,
    best_index: Mapping[str, float],
    identity: RunIdentity,
    model: Any,
    optimizer: Any,
    center_criterion: Any,
    optimizer_center: Any,
    scheduler: Any,
    scaler: Any,
) -> None:
    """Atomically persist one complete, resumable epoch boundary."""

    normalized_best = _validate_boundary(epoch, phase, best_index)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "phase": phase,
        "best_index": normalized_best,
        "identity": identity.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "center_criterion": center_criterion.state_dict(),
        "optimizer_center": optimizer_center.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _capture_rng_state(),
    }
    _validate_nested_training_state(
        payload, optimizer=optimizer, scheduler=scheduler
    )
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if manifest_path.exists():
        old_manifest, _ = _read_checkpoint_manifest(manifest_path)
        previous = old_manifest["current"]
    current = _write_state_generation(payload, manifest_path)
    _publish_checkpoint_manifest(
        manifest_path, current=current, previous=previous
    )
    logging.getLogger("DeMo.train").info(
        "Durable recovery boundary: epoch=%d phase=%s path=%s",
        epoch,
        phase,
        path,
    )


def restore_training_checkpoint(
    path: Path,
    *,
    expected_identity: RunIdentity,
    model: Any,
    optimizer: Any,
    center_criterion: Any,
    optimizer_center: Any,
    scheduler: Any,
    scaler: Any,
) -> ResumeCursor:
    """Validate and restore a complete checkpoint, returning its next cursor."""

    checkpoint_path = Path(path)
    try:
        _, state_path = _read_checkpoint_manifest(checkpoint_path)
        payload = torch.load(
            state_path, map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise RuntimeError(
            f"cannot load recovery checkpoint {checkpoint_path}: {error}"
        ) from error
    required = {
        "schema_version",
        "epoch",
        "phase",
        "best_index",
        "identity",
        "model",
        "optimizer",
        "center_criterion",
        "optimizer_center",
        "scheduler",
        "scaler",
        "rng",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("checkpoint fields are incomplete or unexpected")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "unsupported recovery checkpoint schema: "
            f"{payload['schema_version']!r}"
        )
    if payload["identity"] != expected_identity.to_dict():
        raise ValueError("recovery checkpoint run identity does not match")
    normalized_best = _validate_boundary(
        payload["epoch"], payload["phase"], payload["best_index"]
    )
    _validate_nested_training_state(
        payload, optimizer=optimizer, scheduler=scheduler
    )

    model.load_state_dict(payload["model"], strict=True)
    center_criterion.load_state_dict(payload["center_criterion"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_center.load_state_dict(payload["optimizer_center"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    _restore_rng_state(payload["rng"])
    return ResumeCursor(
        epoch=payload["epoch"],
        phase=payload["phase"],
        best_index=normalized_best,
    )


def run_resumable_epochs(
    *,
    checkpoint_path: Path,
    identity: RunIdentity,
    max_epochs: int,
    eval_period: int,
    initial_best_index: Mapping[str, float],
    train_epoch: Callable[[int], None],
    evaluate_epoch: Callable[
        [int, dict[str, float]], Mapping[str, float]
    ],
    model: Any,
    optimizer: Any,
    center_criterion: Any,
    optimizer_center: Any,
    scheduler: Any,
    scaler: Any,
) -> ResumeCursor:
    """Run epoch callbacks from atomic boundaries until the fixed endpoint."""

    path = Path(checkpoint_path)
    state_objects = {
        "model": model,
        "optimizer": optimizer,
        "center_criterion": center_criterion,
        "optimizer_center": optimizer_center,
        "scheduler": scheduler,
        "scaler": scaler,
    }
    if path.exists():
        cursor = restore_training_checkpoint(
            path,
            expected_identity=identity,
            **state_objects,
        )
    else:
        normalized_best = _validate_boundary(
            0, "post_eval", initial_best_index
        )
        cursor = ResumeCursor(
            epoch=0, phase="post_eval", best_index=normalized_best
        )
        save_training_checkpoint(
            path,
            epoch=cursor.epoch,
            phase=cursor.phase,
            best_index=cursor.best_index,
            identity=identity,
            **state_objects,
        )

    while True:
        action = next_resume_action(
            cursor, max_epochs=max_epochs, eval_period=eval_period
        )
        if action.kind == "complete":
            return cursor
        if action.kind == "train":
            train_epoch(action.epoch)
            cursor = ResumeCursor(
                epoch=action.epoch,
                phase="post_train",
                best_index=dict(cursor.best_index),
            )
        elif action.kind == "evaluate":
            updated_best = evaluate_epoch(
                action.epoch, dict(cursor.best_index)
            )
            cursor = ResumeCursor(
                epoch=action.epoch,
                phase="post_eval",
                best_index=_validate_boundary(
                    action.epoch, "post_eval", updated_best
                ),
            )
        elif action.kind == "finalize":
            cursor = ResumeCursor(
                epoch=action.epoch,
                phase="post_eval",
                best_index=dict(cursor.best_index),
            )
        else:
            raise AssertionError(f"unhandled resume action: {action.kind}")
        save_training_checkpoint(
            path,
            epoch=cursor.epoch,
            phase=cursor.phase,
            best_index=cursor.best_index,
            identity=identity,
            **state_objects,
        )

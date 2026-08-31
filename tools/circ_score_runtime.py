"""GPU runtime behind the public ``build_circ_targets.py`` scoring operation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np
import torch
import yaml

from modeling.trifusion.builder import build_trifusion_from_clip
from modeling.trifusion.circ_scoring import (
    apply_registered_condition_batch,
    audit_proxy_target_transfer,
    audit_query_gallery_symmetry,
    build_reference_margin_bank,
    expand_registered_conditions,
    forward_fused_embeddings,
    score_registered_condition,
    select_proxy_transfer_rows,
)
from modeling.trifusion.data import (
    build_rgbnt201_dev_loaders,
    build_rgbnt201_oof_loaders,
    build_rgbnt201_record_eval_loader,
)
from modeling.trifusion.intervention_targets import (
    CIRCTargetCache,
    compile_circ_targets,
    write_circ_target_cache,
)
from modeling.trifusion.interventions import FullNetworkIntervention
from modeling.trifusion.protocol import (
    load_trusted_circ_protocol,
    trifusion_source_hashes,
)
from modeling.trifusion.state import EXPERT_ORDER, MODALITY_ORDER


PROJECT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_mapping(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    if path.suffix.lower() in (".yaml", ".yml"):
        loaded = yaml.safe_load(payload.decode("utf-8"))
    else:
        loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return loaded, hashlib.sha256(payload).hexdigest()


def _resolve_file(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_directory(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _collect_images(loader: Any) -> tuple[dict[str, torch.Tensor], list[int], list[int], list[str]]:
    chunks = {modality: [] for modality in MODALITY_ORDER}
    identities: list[int] = []
    cameras: list[int] = []
    sample_keys: list[str] = []
    for images, pids, camids, _camids_batch, _viewids, paths in loader:
        if tuple(images) != MODALITY_ORDER:
            raise ValueError("RGBNT201 loader modality order changed")
        for modality in MODALITY_ORDER:
            chunks[modality].append(images[modality].contiguous())
        identities.extend(int(value) for value in pids)
        cameras.extend(int(value) for value in camids.tolist())
        sample_keys.extend(str(value) for value in paths)
    if not sample_keys or len(set(sample_keys)) != len(sample_keys):
        raise ValueError("CIRC scoring sample keys must be nonempty and unique")
    return (
        {modality: torch.cat(chunks[modality], dim=0) for modality in MODALITY_ORDER},
        identities,
        cameras,
        sample_keys,
    )


def _reference_digest(
    embeddings: torch.Tensor,
    *,
    metadata: dict[str, Any],
) -> str:
    tensor = embeddings.detach().cpu().contiguous().float()
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(metadata))
    digest.update(b"\0")
    digest.update(
        _canonical_json_bytes(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        )
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_numpy(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, tensor.detach().cpu().contiguous().float().numpy(), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _build_model(
    generator_config: dict[str, Any],
    *,
    num_classes: int,
    checkpoint: Path,
) -> torch.nn.Module:
    model_config = generator_config["MODEL"]
    built = build_trifusion_from_clip(
        model_config["CLIP_CHECKPOINT"],
        num_classes=num_classes,
        image_size=tuple(model_config["IMAGE_SIZE"]),
        patch_size=int(model_config["PATCH_SIZE"]),
        cnn_width=int(model_config["CNN_WIDTH"]),
        mamba_width=int(model_config["MAMBA_WIDTH"]),
        embedding_width=int(model_config["EMBEDDING_WIDTH"]),
        private_width=int(model_config["PRIVATE_WIDTH"]),
        relay_rank=int(model_config["RELAY_RANK"]),
        reliability_mode="uniform",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("OOF checkpoint does not contain a state dictionary")
    built.model.load_state_dict(state, strict=True)
    return built.model.cuda().eval()


def score_oof_interventions(
    config: dict[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    mode: str,
    output: Path,
    validate_generators: Callable[..., int],
) -> int:
    """Validate three generators, score all frozen interventions and compile cache."""

    if mode != "development":
        raise ValueError("current CIRC OOF scoring operation is development-only")
    if config.get("schema_version") != "circ-scoring-orchestration-v1":
        raise ValueError("unsupported CIRC scoring orchestration schema")
    if config.get("operation") != "score-oof-interventions":
        raise ValueError("CIRC scoring operation name is invalid")
    if os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1":
        raise ValueError("scientific CIRC scoring forbids contract-testing mode")
    protocol_path = _resolve_file(config_path, config["circ_protocol"])
    protocol, protocol_sha256 = load_trusted_circ_protocol(protocol_path)
    generator_orchestration_path = _resolve_file(
        config_path, config["generator_orchestration_config"]
    )
    generator_orchestration, generator_orchestration_sha256 = _load_mapping(
        generator_orchestration_path
    )
    generators_root = _resolve_directory(config_path, config["generators_root"])
    if validate_generators(
        generator_orchestration,
        config_path=generator_orchestration_path,
        config_sha256=generator_orchestration_sha256,
        mode="development",
        output=generators_root,
    ) != 0:
        raise RuntimeError("OOF generator validation did not complete")
    aggregate_path = generators_root / "generators_receipt.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if (
        aggregate.get("status") != "COMPLETE"
        or aggregate.get("scientific_evidence_eligible") is not True
        or aggregate.get("contract_testing") is not False
        or int(aggregate.get("official_test_access_count", -1)) != 0
        or aggregate.get("circ_protocol_sha256") != protocol_sha256
    ):
        raise ValueError("generator aggregate is not eligible scientific evidence")
    generator_config_path = _resolve_file(
        generator_orchestration_path,
        generator_orchestration["generator_config"],
    )
    generator_config, generator_config_sha256 = _load_mapping(generator_config_path)
    seed = int(generator_config["EXPERIMENT"]["SEED"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    conditions = expand_registered_conditions(protocol)
    operators = dict(protocol.get("condition_operators", {}))
    if set(operators) != {str(condition["family"]) for condition in conditions}:
        raise ValueError("frozen CIRC condition operators are incomplete")
    from tools.run_trifusion_experiment import _preflight

    preflight = _preflight(generator_config_path, "hfer_uniform_generator")
    if not preflight["launch_allowed"]:
        raise RuntimeError(f"CIRC scoring GPU preflight blocked: {preflight['blockers']}")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "circ-scoring-run-v1",
        "operation": "score-oof-interventions",
        "config_sha256": config_sha256,
        "circ_protocol_sha256": protocol_sha256,
        "generator_orchestration_sha256": generator_orchestration_sha256,
        "generator_aggregate_sha256": _sha256(aggregate_path),
        "generator_config_sha256": generator_config_sha256,
        "source_sha256": trifusion_source_hashes(),
        "conditions": list(conditions),
        "condition_operators": operators,
        "determinism": {
            "seed": seed,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "reference_recovery_requires_bitwise_equality": True,
        },
        "official_test_access_count": 0,
        "contract_testing": False,
        "scientific_evidence_eligible": True,
    }
    identity_path = output / "run_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("foreign CIRC scoring recovery identity")
    elif any(output.iterdir()):
        raise ValueError("nonempty CIRC scoring output lacks run identity")
    else:
        _atomic_json(identity_path, identity)

    data_config = generator_config["DATA"]
    batch_size = int(data_config["EVAL_BATCH_SIZE"])
    num_workers = int(data_config["NUM_WORKERS"])
    device = torch.device("cuda")
    amp = bool(generator_config["OPTIMIZATION"]["AMP"])
    all_samples: list[dict[str, Any]] = []
    all_symmetry: list[dict[str, Any]] = []
    fold_receipts: dict[str, str] = {}
    for target_fold in range(3):
        fold_generator_receipt_path = (
            generators_root / f"fold-{target_fold}" / "generator_receipt.json"
        )
        generator_receipt = json.loads(
            fold_generator_receipt_path.read_text(encoding="utf-8")
        )
        checkpoint = Path(generator_receipt["checkpoint"]).resolve()
        if _sha256(checkpoint) != generator_receipt["checkpoint_sha256"]:
            raise ValueError(f"fold-{target_fold} checkpoint changed after validation")
        data = build_rgbnt201_oof_loaders(
            dataset_root=Path(data_config["DATASET_ROOT"]),
            protocol_path=PROJECT / str(data_config["DEV_PROTOCOL"]),
            circ_protocol_path=protocol_path,
            target_fold=target_fold,
            train_batch_size=int(data_config["TRAIN_BATCH_SIZE"]),
            num_instances=int(data_config["NUM_INSTANCES"]),
            eval_batch_size=batch_size,
            num_workers=num_workers,
        )
        model = _build_model(
            generator_config,
            num_classes=data.num_classes,
            checkpoint=checkpoint,
        )
        query_loader = build_rgbnt201_record_eval_loader(
            data.query_records,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        reference_loader = build_rgbnt201_record_eval_loader(
            data.gallery_records,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        query_images, query_ids, query_cameras, query_keys = _collect_images(query_loader)
        reference_images, reference_ids, reference_cameras, reference_keys = (
            _collect_images(reference_loader)
        )
        reference_mask = torch.ones(len(reference_keys), 3, dtype=torch.bool)
        reference_indices = torch.arange(len(reference_keys), dtype=torch.long)
        reference_embeddings = forward_fused_embeddings(
            model,
            reference_images,
            reference_mask,
            reference_indices,
            batch_size=batch_size,
            device=device,
            amp=amp,
        )
        reference_metadata = {
            "schema_version": "circ-reference-bank-v1",
            "target_fold": target_fold,
            "sample_keys": reference_keys,
            "identities": reference_ids,
            "cameras": reference_cameras,
            "generator_checkpoint_sha256": generator_receipt["checkpoint_sha256"],
            "circ_protocol_sha256": protocol_sha256,
            "official_test_access_count": 0,
        }
        reference_sha256 = _reference_digest(
            reference_embeddings,
            metadata=reference_metadata,
        )
        fold_output = output / f"fold-{target_fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        reference_features_path = fold_output / "reference_features.npy"
        reference_metadata_path = fold_output / "reference_metadata.json"
        if reference_features_path.exists() or reference_metadata_path.exists():
            if not reference_features_path.is_file() or not reference_metadata_path.is_file():
                raise ValueError("partial reference-bank recovery")
            recovered_features = torch.from_numpy(
                np.load(reference_features_path, allow_pickle=False)
            )
            recovered_metadata = json.loads(
                reference_metadata_path.read_text(encoding="utf-8")
            )
            if (
                recovered_metadata != {**reference_metadata, "reference_bank_sha256": reference_sha256}
                or _reference_digest(
                    recovered_features,
                    metadata=reference_metadata,
                )
                != reference_sha256
                or not torch.equal(recovered_features, reference_embeddings)
            ):
                raise ValueError("reference-bank recovery differs from regenerated evidence")
        else:
            _atomic_numpy(reference_features_path, reference_embeddings)
            _atomic_json(
                reference_metadata_path,
                {**reference_metadata, "reference_bank_sha256": reference_sha256},
            )
        reference_bank = build_reference_margin_bank(
            reference_embeddings,
            reference_ids,
            reference_cameras,
            query_ids,
            query_cameras,
        )
        symmetric_banks = {}
        for expert in EXPERT_ORDER:
            for modality in MODALITY_ORDER:
                contribution = f"{expert}.{modality}"
                intervened_reference = forward_fused_embeddings(
                    model,
                    reference_images,
                    reference_mask,
                    reference_indices,
                    batch_size=batch_size,
                    device=device,
                    amp=amp,
                    intervention=FullNetworkIntervention(
                        kind="total",
                        expert=expert,
                        modality=modality,
                    ),
                )
                symmetric_banks[contribution] = build_reference_margin_bank(
                    intervened_reference,
                    reference_ids,
                    reference_cameras,
                    query_ids,
                    query_cameras,
                )

        fold_samples: list[dict[str, Any]] = []
        fold_symmetry: list[dict[str, Any]] = []
        condition_receipts = {}
        for condition in conditions:
            condition_key = (
                f"{condition['family']}-s{int(condition['severity'])}-"
                f"seed{int(condition['seed'])}"
            )
            rows_path = fold_output / "conditions" / f"{condition_key}.jsonl"
            symmetry_path = fold_output / "conditions" / f"{condition_key}.symmetry.jsonl"
            receipt_path = fold_output / "conditions" / f"{condition_key}.receipt.json"
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("condition") != condition
                    or receipt.get("generator_checkpoint_sha256")
                    != generator_receipt["checkpoint_sha256"]
                    or receipt.get("reference_bank_sha256") != reference_sha256
                    or receipt.get("rows_sha256") != _sha256(rows_path)
                    or receipt.get("symmetry_sha256") != _sha256(symmetry_path)
                    or int(receipt.get("official_test_access_count", -1)) != 0
                ):
                    raise ValueError("condition recovery receipt is foreign or corrupt")
                rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
                symmetry_rows = [
                    json.loads(line)
                    for line in symmetry_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            else:
                rows, symmetry_rows = score_registered_condition(
                    model,
                    query_images,
                    query_keys,
                    query_ids,
                    query_cameras,
                    reference_bank,
                    condition,
                    operators=operators,
                    protocol_hash=protocol_sha256,
                    generator_training_identities=data.provenance[
                        "generator_training_identity_values"
                    ],
                    generator_checkpoint_sha256=generator_receipt[
                        "checkpoint_sha256"
                    ],
                    reference_bank_sha256=reference_sha256,
                    batch_size=batch_size,
                    device=device,
                    amp=amp,
                    symmetric_banks=symmetric_banks,
                )
                rows_bytes = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
                symmetry_bytes = b"".join(
                    _canonical_json_bytes(row) + b"\n" for row in symmetry_rows
                )
                _atomic_bytes(rows_path, rows_bytes)
                _atomic_bytes(symmetry_path, symmetry_bytes)
                receipt = {
                    "schema_version": "circ-condition-score-v1",
                    "target_fold": target_fold,
                    "condition": condition,
                    "rows": len(rows),
                    "symmetry_effects": len(symmetry_rows),
                    "rows_sha256": _sha256(rows_path),
                    "symmetry_sha256": _sha256(symmetry_path),
                    "generator_checkpoint_sha256": generator_receipt[
                        "checkpoint_sha256"
                    ],
                    "reference_bank_sha256": reference_sha256,
                    "official_test_access_count": 0,
                    "scientific_evidence_eligible": True,
                }
                _atomic_json(receipt_path, receipt)
            fold_samples.extend(rows)
            fold_symmetry.extend(symmetry_rows)
            condition_receipts[condition_key] = _sha256(receipt_path)
        fold_receipt = {
            "schema_version": "circ-fold-scoring-v1",
            "target_fold": target_fold,
            "sample_condition_rows": len(fold_samples),
            "symmetry_effects": len(fold_symmetry),
            "query_records": len(query_keys),
            "reference_records": len(reference_keys),
            "reference_bank_sha256": reference_sha256,
            "generator_receipt_sha256": _sha256(fold_generator_receipt_path),
            "condition_receipts": condition_receipts,
            "official_test_access_count": 0,
            "scientific_evidence_eligible": True,
        }
        fold_receipt_path = fold_output / "fold_scoring_receipt.json"
        _atomic_json(fold_receipt_path, fold_receipt)
        fold_receipts[str(target_fold)] = _sha256(fold_receipt_path)
        all_samples.extend(fold_samples)
        all_symmetry.extend(fold_symmetry)
        del model
        torch.cuda.empty_cache()

    audits = dict(protocol["audits"])
    epsilon = float(protocol["target_definition"]["epsilon_cf"])
    symmetry_audit = audit_query_gallery_symmetry(
        all_symmetry,
        protocol_hash=protocol_sha256,
        epsilon=epsilon,
        audit_specification=dict(audits["query_gallery_symmetry"]),
    )
    dev_protocol = json.loads(
        (PROJECT / protocol["development"]["dev_protocol"]).read_text(
            encoding="utf-8"
        )
    )
    source_config = {
        "schema_version": "circ-source-v1",
        "protocol_hash": protocol_sha256,
        "fold_salt": protocol["folds"]["salt"],
        "fold_count": int(protocol["folds"]["count"]),
        "epsilon": epsilon,
        "configuration_frozen": True,
        "official_test_access_count": 0,
        "development_forbidden_identities": dev_protocol["dev_ids"],
        "samples": all_samples,
        "query_gallery_symmetry_audit": symmetry_audit,
        "proxy_target_transfer_audit": {
            "status": "PENDING_DEPLOYED_MODEL",
            "claim_eligible": False,
            "sample_rows": int(audits["proxy_target_transfer"]["sample_rows"]),
            "minimum_sign_agreement": float(
                audits["proxy_target_transfer"]["minimum_sign_agreement"]
            ),
            "minimum_spearman": float(
                audits["proxy_target_transfer"]["minimum_spearman"]
            ),
        },
    }
    source_path = output / "scored_source.json"
    source_bytes = json.dumps(source_config, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if source_path.is_file() and source_path.read_bytes() != source_bytes:
        raise ValueError("existing scored CIRC source differs from regenerated source")
    if not source_path.is_file():
        _atomic_bytes(source_path, source_bytes)
    rows, receipt = compile_circ_targets(
        source_config,
        mode="development",
        config_sha256=_sha256(source_path),
    )
    cache_output = output / "cache"
    if cache_output.exists():
        existing_receipt = json.loads(
            (cache_output / "receipt.json").read_text(encoding="utf-8")
        )
        expected_targets = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
        if (
            existing_receipt.get("targets_sha256")
            != hashlib.sha256(expected_targets).hexdigest()
            or (cache_output / "targets.jsonl").read_bytes() != expected_targets
        ):
            raise ValueError("existing immutable CIRC cache differs from regenerated cache")
    else:
        write_circ_target_cache(rows, receipt, output_directory=cache_output)
    final_receipt = {
        "schema_version": "circ-scoring-receipt-v1",
        "status": "COMPLETE",
        "rows": len(rows),
        "conditions": len(conditions),
        "circ_protocol_sha256": protocol_sha256,
        "fold_receipts": fold_receipts,
        "source_sha256": _sha256(source_path),
        "targets_sha256": _sha256(cache_output / "targets.jsonl"),
        "cache_receipt_sha256": _sha256(cache_output / "receipt.json"),
        "symmetry_audit": symmetry_audit,
        "proxy_target_transfer_status": "PENDING_DEPLOYED_MODEL",
        "official_test_access_count": 0,
        "contract_testing": False,
        "scientific_evidence_eligible": True,
    }
    _atomic_json(output / "scoring_receipt.json", final_receipt)
    return 0


def audit_deployed_transfer(
    config: dict[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    mode: str,
    output: Path,
) -> int:
    """Audit whether OOF proxy effects transfer to the frozen deployed dev model."""

    if mode != "development":
        raise ValueError("deployed proxy-transfer audit is development-only")
    if config.get("schema_version") != "circ-transfer-audit-v1":
        raise ValueError("unsupported CIRC transfer-audit schema")
    if config.get("operation") != "audit-deployed-transfer":
        raise ValueError("CIRC transfer-audit operation name is invalid")
    if os.environ.get("TRIFUSION_CONTRACT_TESTING") == "1":
        raise ValueError("scientific transfer audit forbids contract-testing mode")

    protocol_path = _resolve_file(config_path, config["circ_protocol"])
    protocol, protocol_sha256 = load_trusted_circ_protocol(protocol_path)
    model_config_path = _resolve_file(config_path, config["model_config"])
    model_config, model_config_sha256 = _load_mapping(model_config_path)
    if model_config.get("EXPERIMENT", {}).get("VARIANT") != "trifusion_circ_urgc":
        raise ValueError("transfer audit requires the full CIRC+URGC model config")
    run_root = _resolve_directory(config_path, config["dev_run_root"])
    best_receipt_path = run_root / "best_dev_receipt.json"
    run_summary_path = run_root / "run_summary.json"
    run_identity_path = run_root / "run_identity.json"
    for path in (best_receipt_path, run_summary_path, run_identity_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    best_receipt = json.loads(best_receipt_path.read_text(encoding="utf-8"))
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
    current_source_sha256 = trifusion_source_hashes()
    if (
        best_receipt.get("variant") != "trifusion_circ_urgc"
        or best_receipt.get("phase") != "complete"
        or best_receipt.get("scientific_evidence_eligible") is not True
        or best_receipt.get("contract_testing") is not False
        or int(best_receipt.get("official_test_access_count", -1)) != 0
        or best_receipt.get("circ_protocol_sha256") != protocol_sha256
        or best_receipt.get("config_sha256") != model_config_sha256
        or run_summary.get("status") != "PASS"
        or int(run_summary.get("official_test_access_count", -1)) != 0
        or run_summary.get("sota_claim_supported") is not False
        or run_identity.get("source_sha256") != current_source_sha256
        or run_identity.get("circ_protocol_sha256") != protocol_sha256
        or run_identity.get("config_sha256") != model_config_sha256
        or run_identity.get("scientific_evidence_eligible") is not True
        or run_identity.get("contract_testing") is not False
        or _sha256(run_identity_path) != best_receipt.get("run_identity_sha256")
    ):
        raise ValueError("full dev endpoint is not eligible transfer-audit evidence")
    checkpoint = Path(str(best_receipt.get("checkpoint", ""))).expanduser().resolve()
    if (
        checkpoint != (run_root / "best_dev_model.pth").resolve()
        or not checkpoint.is_file()
        or _sha256(checkpoint) != best_receipt.get("checkpoint_sha256")
        or run_summary.get("best_checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise ValueError("full dev checkpoint is missing or hash-mismatched")
    recovery_manifest_path = Path(
        str(best_receipt.get("recovery_manifest", ""))
    ).expanduser().resolve()
    expected_manifest_path = (run_root / ".resume/latest.json").resolve()
    if recovery_manifest_path != expected_manifest_path or not recovery_manifest_path.is_file():
        raise ValueError("full dev recovery manifest is missing or outside its run")
    recovery_manifest = json.loads(
        recovery_manifest_path.read_text(encoding="utf-8")
    )
    completion_evidence = dict(recovery_manifest.get("completion_evidence", {}))
    current_generation = dict(recovery_manifest.get("current", {}))
    current_generation_path = (
        run_root / str(current_generation.get("path", ""))
    ).resolve()
    if (
        recovery_manifest.get("phase") != "complete"
        or int(recovery_manifest.get("epoch", -1)) != 60
        or recovery_manifest.get("run_identity_sha256") != _sha256(run_identity_path)
        or _sha256(recovery_manifest_path)
        != best_receipt.get("recovery_manifest_sha256")
        or completion_evidence.get("kind") != "selector"
        or int(completion_evidence.get("epoch", -1)) != 60
        or completion_evidence.get("run_identity_sha256")
        != _sha256(run_identity_path)
        or completion_evidence.get("best_checkpoint_sha256")
        != _sha256(checkpoint)
        or completion_evidence.get("scientific_evidence_eligible") is not True
        or completion_evidence.get("contract_testing") is not False
        or not current_generation_path.is_file()
        or _sha256(current_generation_path) != current_generation.get("sha256")
    ):
        raise ValueError("full dev endpoint is not recovery-bound")

    target_cache_path = _resolve_directory(config_path, config["target_cache"])
    target_cache = CIRCTargetCache.from_directory(target_cache_path)
    if (
        target_cache.receipt.get("protocol_hash") != protocol_sha256
        or int(target_cache.receipt.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("transfer audit target cache is not protocol-bound")
    audit_specification = dict(protocol["audits"]["proxy_target_transfer"])
    if (
        audit_specification.get("comparisons")
        != [
            "proxy_effect_vs_deployed_effect",
            "router_helpfulness_vs_deployed_effect",
        ]
        or float(audit_specification.get("router_helpful_threshold", -1.0))
        != 0.5
        or audit_specification.get("identity_clustered") is not True
    ):
        raise ValueError("proxy-transfer audit semantics differ from the frozen gate")
    selected = select_proxy_transfer_rows(
        target_cache.rows,
        sample_count=int(audit_specification["sample_rows"]),
        protocol_hash=protocol_sha256,
    )

    from tools.run_trifusion_experiment import _preflight

    preflight = _preflight(model_config_path, "trifusion_circ_urgc")
    if not preflight["launch_allowed"]:
        raise RuntimeError(f"transfer-audit GPU preflight blocked: {preflight['blockers']}")
    seed = int(model_config["EXPERIMENT"]["SEED"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "circ-transfer-run-v1",
        "operation": "audit-deployed-transfer",
        "config_sha256": config_sha256,
        "model_config_sha256": model_config_sha256,
        "circ_protocol_sha256": protocol_sha256,
        "target_cache_targets_sha256": target_cache.receipt["targets_sha256"],
        "dev_run_identity_sha256": _sha256(run_identity_path),
        "dev_checkpoint_sha256": _sha256(checkpoint),
        "source_sha256": current_source_sha256,
        "seed": seed,
        "selected_rows": len(selected),
        "official_test_access_count": 0,
        "contract_testing": False,
        "scientific_evidence_eligible": True,
    }
    identity_path = output / "run_identity.json"
    receipt_path = output / "target_transfer_receipt.json"
    effects_path = output / "paired_effects.jsonl"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("foreign deployed transfer-audit recovery identity")
    elif any(output.iterdir()):
        raise ValueError("nonempty transfer-audit output lacks run identity")
    else:
        _atomic_json(identity_path, identity)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("run_identity_sha256") != _sha256(identity_path)
            or receipt.get("paired_effects_file_sha256") != _sha256(effects_path)
            or int(receipt.get("official_test_access_count", -1)) != 0
        ):
            raise ValueError("deployed transfer-audit recovery is corrupt")
        return 0 if receipt.get("status") == "PASS" else 5

    data_config = model_config["DATA"]
    data = build_rgbnt201_dev_loaders(
        dataset_root=Path(data_config["DATASET_ROOT"]),
        protocol_path=PROJECT / str(data_config["DEV_PROTOCOL"]),
        train_batch_size=int(data_config["TRAIN_BATCH_SIZE"]),
        num_instances=int(data_config["NUM_INSTANCES"]),
        eval_batch_size=int(data_config["EVAL_BATCH_SIZE"]),
        num_workers=int(data_config["NUM_WORKERS"]),
    )
    evaluation_loader = build_rgbnt201_record_eval_loader(
        data.train_records,
        batch_size=int(data_config["EVAL_BATCH_SIZE"]),
        num_workers=int(data_config["NUM_WORKERS"]),
    )
    images, identities, cameras, sample_keys = _collect_images(evaluation_loader)
    sample_to_index = {sample_key: index for index, sample_key in enumerate(sample_keys)}
    if len(sample_to_index) != len(sample_keys) or any(
        row["sample_key"] not in sample_to_index for row in selected
    ):
        raise ValueError("selected transfer rows do not map to unique fit samples")

    model_specification = model_config["MODEL"]
    built = build_trifusion_from_clip(
        model_specification["CLIP_CHECKPOINT"],
        num_classes=data.num_classes,
        image_size=tuple(model_specification["IMAGE_SIZE"]),
        patch_size=int(model_specification["PATCH_SIZE"]),
        cnn_width=int(model_specification["CNN_WIDTH"]),
        mamba_width=int(model_specification["MAMBA_WIDTH"]),
        embedding_width=int(model_specification["EMBEDDING_WIDTH"]),
        private_width=int(model_specification["PRIVATE_WIDTH"]),
        relay_rank=int(model_specification["RELAY_RANK"]),
        reliability_mode="joint_beta",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("deployed checkpoint is not a tensor state dictionary")
    built.model.load_state_dict(state, strict=True)
    model = built.model.cuda().eval()
    device = torch.device("cuda")
    batch_size = int(data_config["EVAL_BATCH_SIZE"])
    amp = bool(model_config["OPTIMIZATION"]["AMP"])
    reference_mask = torch.ones(len(sample_keys), 3, dtype=torch.bool)
    reference_embeddings = forward_fused_embeddings(
        model,
        images,
        reference_mask,
        torch.arange(len(sample_keys)),
        batch_size=batch_size,
        device=device,
        amp=amp,
    )
    selected_indices = torch.tensor(
        [sample_to_index[row["sample_key"]] for row in selected],
        dtype=torch.long,
    )
    selected_images = {
        modality: tensor.index_select(0, selected_indices)
        for modality, tensor in images.items()
    }
    selected_keys = tuple(str(row["sample_key"]) for row in selected)
    selected_conditions = tuple(dict(row["condition"]) for row in selected)
    conditioned_images, modality_mask = apply_registered_condition_batch(
        selected_images,
        selected_keys,
        selected_conditions,
        operators=dict(protocol["condition_operators"]),
    )
    selected_identities = [int(row["identity"]) for row in selected]
    selected_cameras = [int(row["camera"]) for row in selected]
    if any(
        int(sample_keys[index][:6]) != identity or cameras[index] != camera
        for index, identity, camera in zip(
            selected_indices.tolist(), selected_identities, selected_cameras
        )
    ):
        raise ValueError("selected transfer metadata differs from RGBNT201 records")
    selected_model_identities = [
        identities[index] for index in selected_indices.tolist()
    ]
    reference_bank = build_reference_margin_bank(
        reference_embeddings,
        identities,
        cameras,
        selected_model_identities,
        selected_cameras,
    )
    audit_indices = torch.arange(len(selected), dtype=torch.long)
    baseline_embedding_chunks = []
    reliability_chunks = []
    for start in range(0, len(selected), batch_size):
        batch_indices = audit_indices[start : start + batch_size]
        batch = {
            "images": {
                modality: tensor.index_select(0, batch_indices).to(device)
                for modality, tensor in conditioned_images.items()
            },
            "modality_mask": modality_mask.index_select(0, batch_indices).to(
                device
            ),
        }
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            enabled=amp,
        ):
            baseline_output = model(batch, return_aux=True)
        baseline_embedding_chunks.append(
            torch.nn.functional.normalize(
                baseline_output.fused_embedding.detach().float(), dim=1
            ).cpu()
        )
        reliability_chunks.append(baseline_output.reliability.r.detach().float().cpu())
    baseline_embeddings = torch.cat(baseline_embedding_chunks, dim=0)
    deployed_reliability = torch.cat(reliability_chunks, dim=0)
    router_scores = []
    for index, row in enumerate(selected):
        expert, modality = str(row["contribution"]).split(".", maxsplit=1)
        router_scores.append(
            float(
                deployed_reliability[
                    index,
                    EXPERT_ORDER.index(expert),
                    MODALITY_ORDER.index(modality),
                ]
            )
        )
    baseline_margins = reference_bank.margins(baseline_embeddings).cpu()
    deployed_deltas = [float("nan")] * len(selected)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(selected):
        grouped.setdefault(str(row["contribution"]), []).append(index)
    for contribution, group_indices in grouped.items():
        expert, modality = contribution.split(".", maxsplit=1)
        intervention = FullNetworkIntervention(
            kind="total",
            expert=expert,
            modality=modality,
        )
        embeddings = forward_fused_embeddings(
            model,
            conditioned_images,
            modality_mask,
            group_indices,
            batch_size=batch_size,
            device=device,
            amp=amp,
            intervention=intervention,
        )
        margins = reference_bank.margins(embeddings, group_indices).cpu()
        deltas = baseline_margins[group_indices] - margins
        for local_index, global_index in enumerate(group_indices):
            deployed_deltas[global_index] = float(deltas[local_index])
    if not all(math.isfinite(value) for value in deployed_deltas):
        raise FloatingPointError("deployed transfer audit produced nonfinite effects")

    audit = audit_proxy_target_transfer(
        selected,
        deployed_deltas,
        router_scores,
        epsilon=float(protocol["target_definition"]["epsilon_cf"]),
        minimum_sign_agreement=float(
            audit_specification["minimum_sign_agreement"]
        ),
        minimum_spearman=float(audit_specification["minimum_spearman"]),
    )
    paired_rows = [
        {
            **dict(row),
            "deployed_delta": deployed_delta,
            "router_score": router_score,
        }
        for row, deployed_delta, router_score in zip(
            selected, deployed_deltas, router_scores
        )
    ]
    _atomic_bytes(
        effects_path,
        b"".join(_canonical_json_bytes(row) + b"\n" for row in paired_rows),
    )
    receipt = {
        "schema_version": "circ-target-transfer-receipt-v1",
        **audit,
        "circ_protocol_sha256": protocol_sha256,
        "target_cache_targets_sha256": target_cache.receipt["targets_sha256"],
        "dev_checkpoint_sha256": _sha256(checkpoint),
        "run_identity_sha256": _sha256(identity_path),
        "paired_effects_file_sha256": _sha256(effects_path),
        "official_test_access_count": 0,
        "contract_testing": False,
        "scientific_evidence_eligible": audit["claim_eligible"],
    }
    _atomic_json(receipt_path, receipt)
    return 0 if audit["status"] == "PASS" else 5


__all__ = ["audit_deployed_transfer", "score_oof_interventions"]

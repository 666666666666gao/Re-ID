"""Deterministic, identity-disjoint CIRC target-cache primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from .state import EXPERT_ORDER, MODALITY_ORDER


CIRC_EDGE_SALT = "TriFusion-CIRC-edge-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class EdgeSelection:
    stage: int
    edge: str
    digest_sha256: str
    ordered_edge_count: int
    selected_index: int


def select_audit_edge(
    valid_edges: Sequence[str],
    *,
    protocol_hash: str,
    sample_key: str,
    condition: Mapping[str, object],
    stage: int,
) -> EdgeSelection:
    """Select exactly one lexicographically ordered valid edge for one stage."""

    if stage not in (1, 2):
        raise ValueError("CIRC edge stage must be 1 or 2")
    ordered_edges = tuple(sorted(set(valid_edges)))
    if not ordered_edges:
        raise ValueError("at least one valid no-self edge is required")
    required_condition_fields = {"family", "severity", "seed"}
    if set(condition) != required_condition_fields:
        raise ValueError(
            f"condition must contain exactly {sorted(required_condition_fields)}"
        )
    hash_input = {
        "salt": CIRC_EDGE_SALT,
        "protocol_hash": protocol_hash,
        "sample_key": sample_key,
        "condition": dict(condition),
        "stage": stage,
    }
    digest = hashlib.sha256(_canonical_json(hash_input)).hexdigest()
    selected_index = int(digest, 16) % len(ordered_edges)
    return EdgeSelection(
        stage=stage,
        edge=ordered_edges[selected_index],
        digest_sha256=digest,
        ordered_edge_count=len(ordered_edges),
        selected_index=selected_index,
    )


def assign_identity_fold(
    identity: int | str, *, fold_salt: str, fold_count: int
) -> int:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    identity_text = canonical_unsigned_identity(identity)
    payload = _canonical_json({"salt": fold_salt, "identity": identity_text})
    return int(hashlib.sha256(payload).hexdigest(), 16) % fold_count


def canonical_unsigned_identity(identity: int | str) -> str:
    """Return the sole CIRC identity representation: ASCII unsigned decimal."""

    identity_text = str(identity)
    if not identity_text or any(character not in "0123456789" for character in identity_text):
        raise ValueError("CIRC identity must be ASCII unsigned decimal")
    return str(int(identity_text, 10))


def valid_edges_for_mask(modality_mask: Sequence[bool]) -> tuple[str, ...]:
    if len(modality_mask) != len(MODALITY_ORDER):
        raise ValueError("modality mask must follow RGB, NI, TI")
    return tuple(
        sorted(
            f"{source}->{target}:{modality}"
            for source in EXPERT_ORDER
            for target in EXPERT_ORDER
            if source != target
            for modality, valid in zip(MODALITY_ORDER, modality_mask)
            if valid
        )
    )


def _effect_label(delta: float, epsilon: float) -> str:
    if delta > epsilon:
        return "helpful"
    if delta < -epsilon:
        return "harmful"
    return "neutral"


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal") from error
    return value.lower()


def _require_finite(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _binary_calibration_metrics(
    probabilities: Sequence[float], labels: Sequence[int], *, bins: int = 10
) -> dict[str, float | int]:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("calibration probabilities and labels must align and be nonempty")
    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    clipped = [min(max(float(value), 1e-6), 1.0 - 1e-6) for value in probabilities]
    binary = [int(value) for value in labels]
    if any(value not in (0, 1) for value in binary):
        raise ValueError("calibration labels must be binary")
    count = len(binary)
    bce = -sum(
        label * math.log(probability)
        + (1 - label) * math.log(1.0 - probability)
        for probability, label in zip(clipped, binary)
    ) / count
    brier = sum(
        (probability - label) ** 2
        for probability, label in zip(clipped, binary)
    ) / count
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, probability in enumerate(clipped)
            if probability >= lower
            and (probability < upper or (bin_index == bins - 1 and probability <= upper))
        ]
        if not indices:
            continue
        confidence = sum(clipped[index] for index in indices) / len(indices)
        accuracy = sum(binary[index] for index in indices) / len(indices)
        ece += len(indices) / count * abs(confidence - accuracy)
    return {
        "rows": count,
        "BCE": bce,
        "Brier": brier,
        "ECE": ece,
        "positive_rate": sum(binary) / count,
        "mean_probability": sum(clipped) / count,
    }


def compute_calibration_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    router_probabilities: Mapping[tuple[str, str, str], float] | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Audit calibration without counting corruption repeats as iid evidence."""

    if not rows:
        raise ValueError("calibration audit requires immutable CIRC target rows")
    observations: list[dict[str, Any]] = []
    for row in rows:
        if row.get("cross_camera_support") is not True:
            raise ValueError("calibration rows require cross-camera positive support")
        cluster = f"{int(row['identity'])}|{row['sample_key']}"
        condition_key = str(row["condition_key"])
        groups = dict(row["groups"])
        for expert in EXPERT_ORDER:
            for modality in MODALITY_ORDER:
                contribution = dict(row["contributions"][f"{expert}.{modality}"])
                if not bool(contribution["valid"]):
                    continue
                observations.append(
                    {
                        "cluster": cluster,
                        "sample_key": str(row["sample_key"]),
                        "condition": condition_key,
                        "expert": expert,
                        "modality": modality,
                        "expert_modality": f"{expert}.{modality}",
                        "camera": str(groups["camera"]),
                        "identity_frequency": str(groups["identity_frequency"]),
                        "label": int(contribution["helpful_target"]),
                    }
                )
    if not observations:
        raise ValueError("calibration audit has no valid expert-modality outcomes")

    prediction_source = (
        "deployed_router" if router_probabilities is not None else "cluster_loo_beta11"
    )
    if router_probabilities is None:
        strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for observation in observations:
            strata[
                (
                    observation["condition"],
                    observation["expert"],
                    observation["modality"],
                )
            ].append(observation)
        for observation in observations:
            peers = strata[
                (
                    observation["condition"],
                    observation["expert"],
                    observation["modality"],
                )
            ]
            peer_labels = [
                int(peer["label"])
                for peer in peers
                if peer["cluster"] != observation["cluster"]
            ]
            observation["probability"] = (1.0 + sum(peer_labels)) / (
                2.0 + len(peer_labels)
            )
    else:
        for observation in observations:
            key = (
                observation["sample_key"],
                observation["condition"],
                observation["expert_modality"],
            )
            if key not in router_probabilities:
                raise ValueError(f"missing deployed-router probability for {key}")
            probability = _require_finite(
                router_probabilities[key], f"router_probabilities[{key}]"
            )
            if probability < 0.0 or probability > 1.0:
                raise ValueError("deployed-router probabilities must lie in [0,1]")
            observation["probability"] = probability

    def metrics(selected: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return _binary_calibration_metrics(
            [float(value["probability"]) for value in selected],
            [int(value["label"]) for value in selected],
            bins=ece_bins,
        )

    axes = [
        "condition",
        "expert",
        "modality",
        "expert_modality",
        "camera",
        "identity_frequency",
    ]
    per_condition: dict[str, Any] = {}
    for condition in sorted({str(value["condition"]) for value in observations}):
        selected = [value for value in observations if value["condition"] == condition]
        per_condition[condition] = {
            "overall": metrics(selected),
            "groups": {
                axis: {
                    group: metrics(
                        [value for value in selected if str(value[axis]) == group]
                    )
                    for group in sorted({str(value[axis]) for value in selected})
                }
                for axis in axes[1:]
            },
            "identity_query_clusters": len(
                {str(value["cluster"]) for value in selected}
            ),
        }

    overdispersion = {}
    concentration = {}
    for condition in per_condition:
        selected = [value for value in observations if value["condition"] == condition]
        cluster_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in selected:
            cluster_rows[str(value["cluster"])].append(value)
        residual_terms = [
            (float(value["label"]) - float(value["probability"])) ** 2
            / max(
                float(value["probability"]) * (1.0 - float(value["probability"])),
                1e-6,
            )
            for value in selected
        ]
        coverage_flags = []
        for cluster, cluster_values in cluster_rows.items():
            other = [value for value in selected if value["cluster"] != cluster]
            successes = sum(int(value["label"]) for value in other)
            alpha = 1.0 + successes
            beta = 1.0 + len(other) - successes
            posterior_mean = alpha / (alpha + beta)
            posterior_variance = alpha * beta / (
                (alpha + beta) ** 2 * (alpha + beta + 1.0)
            )
            predictive_variance = posterior_variance + posterior_mean * (
                1.0 - posterior_mean
            ) / max(1, len(cluster_values))
            radius = 1.96 * math.sqrt(predictive_variance)
            observed = sum(int(value["label"]) for value in cluster_values) / len(
                cluster_values
            )
            coverage_flags.append(
                max(0.0, posterior_mean - radius)
                <= observed
                <= min(1.0, posterior_mean + radius)
            )
        overdispersion[condition] = {
            "pearson_dispersion": sum(residual_terms)
            / max(1, len(residual_terms) - len(cluster_rows)),
            "rows": len(selected),
            "identity_query_clusters": len(cluster_rows),
        }
        empirical_coverage = sum(coverage_flags) / len(coverage_flags)
        concentration[condition] = {
            "nominal_coverage": 0.95,
            "empirical_coverage": empirical_coverage,
            "identity_query_clusters": len(cluster_rows),
            "claim_eligible": len(cluster_rows) >= 20 and empirical_coverage >= 0.90,
        }

    audit: dict[str, Any] = {
        "schema_version": "circ-calibration-audit-v1",
        "status": "COMPLETE",
        "prediction_source": prediction_source,
        "calibration_metrics": ["BCE", "Brier", "ECE"],
        "ece_bins": ece_bins,
        "group_axes": axes,
        "overall": metrics(observations),
        "per_condition": per_condition,
        "overdispersion": overdispersion,
        "empirical_concentration_coverage": concentration,
        "effective_sample_size": {
            "unit": "identity_query_cluster",
            "identity_query_clusters": len(
                {str(value["cluster"]) for value in observations}
            ),
            "raw_condition_contribution_rows": len(observations),
            "condition_seed_rows_are_not_iid": True,
        },
        "cross_camera_rows_only": True,
    }
    audit["audit_sha256"] = hashlib.sha256(_canonical_json(audit)).hexdigest()
    return audit


def compile_circ_targets(
    config: Mapping[str, Any], *, mode: str, config_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate scored full-network interventions and compile immutable rows."""

    if mode not in ("development", "postfreeze-final"):
        raise ValueError("mode must be development or postfreeze-final")
    if int(config.get("official_test_access_count", -1)) != 0:
        raise ValueError("CIRC target construction must never access official test")
    if mode == "postfreeze-final" and not bool(config.get("configuration_frozen")):
        raise ValueError("postfreeze-final mode requires a frozen configuration")
    protocol_hash = _require_sha256(config.get("protocol_hash"), "protocol_hash")
    fold_salt = str(config.get("fold_salt", ""))
    fold_count = int(config.get("fold_count", 0))
    epsilon = _require_finite(config.get("epsilon"), "epsilon")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    forbidden_development = {
        int(identity)
        for identity in config.get("development_forbidden_identities", [])
    }
    raw_samples = list(config.get("samples", []))
    if not raw_samples:
        raise ValueError("at least one scored intervention sample is required")

    identity_samples: dict[int, set[str]] = defaultdict(set)
    for sample in raw_samples:
        identity = int(sample["identity"])
        identity_samples[identity].add(str(sample["sample_key"]))

    rows = []
    seen_keys = set()
    edge_coverage: dict[str, dict[str, int]] = {
        "stage": {},
        "source": {},
        "target": {},
        "modality": {},
        "camera": {},
        "condition": {},
    }
    outcome_counts = {"helpful": 0, "neutral": 0, "harmful": 0}
    for sample in raw_samples:
        if sample.get("dataset_split") != "train":
            raise ValueError("CIRC rows must come only from the training split")
        sample_key = str(sample["sample_key"])
        identity = int(sample["identity"])
        camera = int(sample["camera"])
        condition = dict(sample["condition"])
        condition_key = _canonical_json(condition).decode("utf-8")
        unique_key = (sample_key, condition_key)
        if unique_key in seen_keys:
            raise ValueError("duplicate sample-key/condition target row")
        seen_keys.add(unique_key)
        if mode == "development" and identity in forbidden_development:
            raise ValueError("development mode attempted to read a forbidden dev identity")

        modality_mask = tuple(bool(value) for value in sample["modality_mask"])
        if len(modality_mask) != len(MODALITY_ORDER) or not any(modality_mask):
            raise ValueError("every target row needs a nonempty RGB/NI/TI mask")
        positive_cameras = {int(value) for value in sample["cross_camera_positive_cameras"]}
        cross_camera_support = any(value != camera for value in positive_cameras)
        if not cross_camera_support:
            raise ValueError("primary CIRC row lacks a different-camera positive")
        generator_training = tuple(
            sorted(int(value) for value in sample["generator_training_identities"])
        )
        if identity in generator_training:
            raise ValueError("target identity overlaps generator training identities")
        fold = assign_identity_fold(
            identity, fold_salt=fold_salt, fold_count=fold_count
        )
        generator_hash = _require_sha256(
            sample.get("generator_checkpoint_sha256"),
            "generator_checkpoint_sha256",
        )
        reference_hash = _require_sha256(
            sample.get("reference_bank_sha256"), "reference_bank_sha256"
        )

        raw_interventions = dict(sample["interventions"])
        expected_contributions = {
            f"{expert}.{modality}"
            for expert in EXPERT_ORDER
            for modality in MODALITY_ORDER
        }
        if set(raw_interventions) != expected_contributions:
            raise ValueError("interventions must contain all nine expert-modalities")
        contributions = {}
        for expert in EXPERT_ORDER:
            for modality_index, modality in enumerate(MODALITY_ORDER):
                contribution_key = f"{expert}.{modality}"
                raw_effects = dict(raw_interventions[contribution_key])
                if set(raw_effects) != {"total", "direct", "relay"}:
                    raise ValueError("each contribution needs total/direct/relay effects")
                effects = {
                    name: _require_finite(
                        raw_effects[name], f"{contribution_key}.{name}"
                    )
                    for name in ("total", "direct", "relay")
                }
                if not modality_mask[modality_index] and any(effects.values()):
                    raise ValueError("missing modality interventions must be exact zero")
                labels = {
                    name: _effect_label(value, epsilon)
                    for name, value in effects.items()
                }
                outcome_counts[labels["total"]] += 1
                contributions[contribution_key] = {
                    "effects": {
                        **effects,
                        "interaction": effects["total"]
                        - effects["direct"]
                        - effects["relay"],
                    },
                    "labels": labels,
                    "helpful_target": int(labels["total"] == "helpful"),
                    "valid": modality_mask[modality_index],
                }

        valid_edges = valid_edges_for_mask(modality_mask)
        raw_edge_effects = dict(sample["edge_effects"])
        edge_audit = []
        for stage in (1, 2):
            selected = select_audit_edge(
                valid_edges,
                protocol_hash=protocol_hash,
                sample_key=sample_key,
                condition=condition,
                stage=stage,
            )
            stage_effects = dict(raw_edge_effects.get(str(stage), {}))
            if selected.edge not in stage_effects:
                raise ValueError("selected full-network edge effect is missing")
            delta = _require_finite(
                stage_effects[selected.edge], f"edge_effects.{stage}.{selected.edge}"
            )
            source_target, modality = selected.edge.split(":", maxsplit=1)
            source, target = source_target.split("->", maxsplit=1)
            edge_audit.append(
                {
                    "stage": stage,
                    "edge": selected.edge,
                    "digest_sha256": selected.digest_sha256,
                    "selected_index": selected.selected_index,
                    "valid_edge_count": selected.ordered_edge_count,
                    "delta": delta,
                    "label": _effect_label(delta, epsilon),
                    "audit_only": True,
                }
            )
            for group, key in (
                ("stage", str(stage)),
                ("source", source),
                ("target", target),
                ("modality", modality),
                ("camera", str(camera)),
                ("condition", condition_key),
            ):
                edge_coverage[group][key] = edge_coverage[group].get(key, 0) + 1

        frequency = len(identity_samples[identity])
        frequency_group = "singleton" if frequency == 1 else "repeated"
        rows.append(
            {
                "schema_version": "circ-target-v1",
                "mode": mode,
                "protocol_hash": protocol_hash,
                "sample_key": sample_key,
                "identity": identity,
                "camera": camera,
                "groups": {
                    "camera": str(camera),
                    "identity_frequency": frequency_group,
                },
                "fold": fold,
                "modality_mask": list(modality_mask),
                "condition": condition,
                "condition_key": condition_key,
                "cross_camera_support": True,
                "cross_camera_positive_cameras": sorted(positive_cameras),
                "generator_training_identities": list(generator_training),
                "generator_checkpoint_sha256": generator_hash,
                "reference_bank_sha256": reference_hash,
                "intervention_seeds": list(sample["intervention_seeds"]),
                "contributions": contributions,
                "edge_audit": edge_audit,
                "primary_target_source": "full_network_total_intervention",
                "edge_values_used_as_training_targets": False,
            }
        )

    rows.sort(key=lambda row: (row["sample_key"], row["condition_key"]))
    receipt = {
        "schema_version": "circ-receipt-v1",
        "mode": mode,
        "protocol_hash": protocol_hash,
        "config_sha256": config_sha256,
        "row_count": len(rows),
        "identity_query_cluster_count": sum(
            len(sample_keys) for sample_keys in identity_samples.values()
        ),
        "zero_identity_overlap": True,
        "cross_camera_primary_rows": len(rows),
        "same_camera_only_rows": 0,
        "invalid_support_rows": 0,
        "official_test_access_count": 0,
        "edge_audit_runs": len(rows) * 2,
        "edge_budget_per_row": 2,
        "edge_salt": CIRC_EDGE_SALT,
        "edge_order": "lexicographic_unique_utf8",
        "edge_digest_encoding": "canonical-json-utf8,unsigned-big-endian",
        "edge_coverage": edge_coverage,
        "total_effect_outcome_counts": outcome_counts,
        "learned_target": "helpful_vs_not_helpful",
        "neutral_and_harmful_are_negative": True,
        "calibration_audit": compute_calibration_audit(rows),
        "query_gallery_symmetry_audit": config.get(
            "query_gallery_symmetry_audit",
            {"status": "not_supplied", "claim_eligible": False},
        ),
        "proxy_target_transfer_audit": config.get(
            "proxy_target_transfer_audit",
            {"status": "not_supplied", "claim_eligible": False},
        ),
    }
    return rows, receipt


def write_circ_target_cache(
    rows: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
    *,
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=False)
    targets_bytes = b"".join(_canonical_json(dict(row)) + b"\n" for row in rows)
    final_receipt = dict(receipt)
    final_receipt["targets_sha256"] = hashlib.sha256(targets_bytes).hexdigest()
    final_receipt["targets_bytes"] = len(targets_bytes)
    receipt_bytes = _canonical_json(final_receipt) + b"\n"
    symmetry_receipt = {
        "schema_version": "circ-symmetry-receipt-v1",
        "protocol_hash": final_receipt["protocol_hash"],
        "targets_sha256": final_receipt["targets_sha256"],
        **dict(final_receipt["query_gallery_symmetry_audit"]),
    }
    transfer_receipt = {
        "schema_version": "circ-target-transfer-receipt-v1",
        "protocol_hash": final_receipt["protocol_hash"],
        "targets_sha256": final_receipt["targets_sha256"],
        **dict(final_receipt["proxy_target_transfer_audit"]),
    }
    calibration_receipt = {
        **dict(final_receipt["calibration_audit"]),
        "schema_version": "circ-calibration-receipt-v1",
        "protocol_hash": final_receipt["protocol_hash"],
        "targets_sha256": final_receipt["targets_sha256"],
    }

    for filename, payload in (
        ("targets.jsonl", targets_bytes),
        ("receipt.json", receipt_bytes),
        ("symmetry_receipt.json", _canonical_json(symmetry_receipt) + b"\n"),
        (
            "target_transfer_receipt.json",
            _canonical_json(transfer_receipt) + b"\n",
        ),
        (
            "calibration_receipt.json",
            _canonical_json(calibration_receipt) + b"\n",
        ),
    ):
        destination = output_directory / filename
        temporary = output_directory / f".{filename}.{os.getpid()}.tmp"
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    return final_receipt


@dataclass(frozen=True, eq=False)
class CIRCTargetBatch:
    helpful_targets: torch.Tensor
    valid_mask: torch.Tensor
    signed_total_effects: torch.Tensor
    provenance_keys: tuple[str, ...]


class CIRCTargetCache:
    """Hash-verified, non-gradient CIRC lookup keyed by sample and condition."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        receipt: Mapping[str, Any],
    ) -> None:
        frozen_rows = tuple(MappingProxyType(dict(row)) for row in rows)
        index = {}
        for row in frozen_rows:
            if int(row["identity"]) in {
                int(value) for value in row["generator_training_identities"]
            }:
                raise ValueError("cache row has target/generator identity overlap")
            if not bool(row["cross_camera_support"]):
                raise ValueError("cache primary row lacks cross-camera support")
            if bool(row.get("edge_values_used_as_training_targets", True)):
                raise ValueError("edge audit values cannot be learned targets")
            if len(row["contributions"]) != len(EXPERT_ORDER) * len(MODALITY_ORDER):
                raise ValueError("cache row must contain all nine contributions")
            key = (str(row["sample_key"]), str(row["condition_key"]))
            if key in index:
                raise ValueError("duplicate cache sample-key/condition")
            index[key] = row
        self.rows = frozen_rows
        self.receipt = MappingProxyType(dict(receipt))
        self._index = MappingProxyType(index)

    @classmethod
    def from_directory(cls, directory: Path | str) -> "CIRCTargetCache":
        directory = Path(directory)
        targets_bytes = (directory / "targets.jsonl").read_bytes()
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(targets_bytes).hexdigest()
        if actual_hash != receipt.get("targets_sha256"):
            raise ValueError("CIRC target-cache SHA-256 mismatch")
        rows = [
            json.loads(line)
            for line in targets_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        return cls(rows, receipt=receipt)

    def lookup(
        self,
        sample_keys: Sequence[str],
        conditions: Sequence[Mapping[str, object]],
        *,
        device: torch.device | str | None = None,
        allow_missing: bool = False,
    ) -> CIRCTargetBatch:
        if len(sample_keys) != len(conditions):
            raise ValueError("sample keys and conditions must have equal length")
        helpful_rows = []
        valid_rows = []
        effect_rows = []
        provenance_keys = []
        for sample_key, condition in zip(sample_keys, conditions):
            condition_key = _canonical_json(dict(condition)).decode("utf-8")
            key = (str(sample_key), condition_key)
            if key not in self._index:
                if not allow_missing:
                    raise KeyError(f"missing immutable CIRC target for {key}")
                helpful_rows.append(
                    [[0.0] * len(MODALITY_ORDER) for _ in EXPERT_ORDER]
                )
                valid_rows.append(
                    [[False] * len(MODALITY_ORDER) for _ in EXPERT_ORDER]
                )
                effect_rows.append(
                    [[0.0] * len(MODALITY_ORDER) for _ in EXPERT_ORDER]
                )
                provenance_keys.append(
                    f"excluded-no-cross-camera-support|{sample_key}|{condition_key}"
                )
                continue
            row = self._index[key]
            helpful = []
            valid = []
            effects = []
            for expert in EXPERT_ORDER:
                helpful_expert = []
                valid_expert = []
                effect_expert = []
                for modality in MODALITY_ORDER:
                    contribution = row["contributions"][f"{expert}.{modality}"]
                    helpful_expert.append(float(contribution["helpful_target"]))
                    valid_expert.append(bool(contribution["valid"]))
                    effect_expert.append(float(contribution["effects"]["total"]))
                helpful.append(helpful_expert)
                valid.append(valid_expert)
                effects.append(effect_expert)
            helpful_rows.append(helpful)
            valid_rows.append(valid)
            effect_rows.append(effects)
            provenance_keys.append(
                f"{sample_key}|{condition_key}|{row['protocol_hash']}"
            )
        return CIRCTargetBatch(
            helpful_targets=torch.tensor(
                helpful_rows, dtype=torch.float32, device=device
            ),
            valid_mask=torch.tensor(valid_rows, dtype=torch.bool, device=device),
            signed_total_effects=torch.tensor(
                effect_rows, dtype=torch.float32, device=device
            ),
            provenance_keys=tuple(provenance_keys),
        )

    def contains(
        self,
        sample_key: str,
        condition: Mapping[str, object],
    ) -> bool:
        condition_key = _canonical_json(dict(condition)).decode("utf-8")
        return (str(sample_key), condition_key) in self._index


__all__ = [
    "CIRC_EDGE_SALT",
    "CIRCTargetBatch",
    "CIRCTargetCache",
    "EdgeSelection",
    "assign_identity_fold",
    "canonical_unsigned_identity",
    "compile_circ_targets",
    "compute_calibration_audit",
    "select_audit_edge",
    "valid_edges_for_mask",
    "write_circ_target_cache",
]

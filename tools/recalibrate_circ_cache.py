#!/usr/bin/env python3
"""Create a hash-bound cluster-calibration overlay for an immutable CIRC cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from modeling.trifusion.intervention_targets import compute_calibration_audit
from modeling.trifusion.state import EXPERT_ORDER, MODALITY_ORDER


REQUIRED_PARENT_ARTIFACTS = (
    "scoring_receipt.json",
    "run_identity.json",
    "scored_source.json",
    "cache/receipt.json",
    "cache/calibration_receipt.json",
    "cache/targets.jsonl",
    "cache/symmetry_receipt.json",
    "cache/target_transfer_receipt.json",
)
METHOD = "condition-cluster-loo-beta-binomial-moments-exact95"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_json(dict(payload)) + b"\n")


def _beta_binomial_equal_tailed_interval(
    trials: int,
    alpha: float,
    beta: float,
    *,
    tail_probability: float,
) -> tuple[int, int]:
    if trials < 1 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta-binomial parameters must be positive")
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must lie in (0, 0.5)")
    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    masses = []
    for successes in range(trials + 1):
        log_mass = (
            math.lgamma(trials + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(trials - successes + 1)
            + math.lgamma(successes + alpha)
            + math.lgamma(trials - successes + beta)
            - math.lgamma(trials + alpha + beta)
            - log_beta
        )
        masses.append(math.exp(log_mass))
    normalizer = sum(masses)
    masses = [mass / normalizer for mass in masses]

    def quantile(probability: float) -> int:
        cumulative = 0.0
        for successes, mass in enumerate(masses):
            cumulative += mass
            if cumulative >= probability:
                return successes
        return trials

    return quantile(tail_probability), quantile(1.0 - tail_probability)


def _fit_cluster_beta_binomial_moments(
    clusters: Sequence[Sequence[int]],
) -> tuple[float, float, float]:
    if len(clusters) < 2 or any(not cluster for cluster in clusters):
        raise ValueError("cluster beta-binomial fit requires two nonempty peers")
    rates = [sum(cluster) / len(cluster) for cluster in clusters]
    total_successes = sum(sum(cluster) for cluster in clusters)
    total_trials = sum(len(cluster) for cluster in clusters)
    mean = (1.0 + total_successes) / (2.0 + total_trials)
    rate_mean = sum(rates) / len(rates)
    rate_variance = sum((rate - rate_mean) ** 2 for rate in rates) / (
        len(rates) - 1
    )
    mean_inverse_size = sum(1.0 / len(cluster) for cluster in clusters) / len(
        clusters
    )
    correlation = (
        rate_variance / max(mean * (1.0 - mean), 1e-12) - mean_inverse_size
    ) / max(1.0 - mean_inverse_size, 1e-12)
    correlation = min(max(correlation, 1e-6), 1.0 - 1e-6)
    concentration = 1.0 / correlation - 1.0
    return (
        max(mean * concentration, 1e-6),
        max((1.0 - mean) * concentration, 1e-6),
        correlation,
    )


def compute_cluster_calibration_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    nominal_coverage: float = 0.95,
    minimum_empirical_coverage: float = 0.90,
    minimum_clusters: int = 20,
) -> dict[str, Any]:
    """Upgrade only target concentration; retain point-calibration diagnostics."""

    if nominal_coverage != 0.95:
        raise ValueError("the registered overlay requires nominal coverage 0.95")
    if minimum_empirical_coverage != 0.90 or minimum_clusters != 20:
        raise ValueError("the registered overlay thresholds cannot be relaxed")
    audit = compute_calibration_audit(rows)
    by_condition: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("cross_camera_support") is not True:
            raise ValueError("overlay rows require cross-camera positive support")
        cluster = f"{int(row['identity'])}|{row['sample_key']}"
        condition = str(row["condition_key"])
        for expert in EXPERT_ORDER:
            for modality in MODALITY_ORDER:
                contribution = row["contributions"][f"{expert}.{modality}"]
                if bool(contribution["valid"]):
                    label = int(contribution["helpful_target"])
                    if label not in (0, 1):
                        raise ValueError("helpful targets must be binary")
                    by_condition[condition][cluster].append(label)

    concentration = {}
    tail_probability = (1.0 - nominal_coverage) / 2.0
    for condition, cluster_rows in sorted(by_condition.items()):
        flags = []
        widths = []
        correlations = []
        ordered = sorted(cluster_rows.items())
        for cluster, labels in ordered:
            peers = [peer_labels for peer, peer_labels in ordered if peer != cluster]
            alpha, beta, correlation = _fit_cluster_beta_binomial_moments(peers)
            lower, upper = _beta_binomial_equal_tailed_interval(
                len(labels),
                alpha,
                beta,
                tail_probability=tail_probability,
            )
            flags.append(lower <= sum(labels) <= upper)
            widths.append((upper - lower) / len(labels))
            correlations.append(correlation)
        empirical = sum(flags) / len(flags)
        concentration[condition] = {
            "method": METHOD,
            "prediction_unit": "identity_query_cluster_helpful_count",
            "nominal_coverage": nominal_coverage,
            "minimum_empirical_coverage": minimum_empirical_coverage,
            "empirical_coverage": empirical,
            "mean_interval_width": sum(widths) / len(widths),
            "mean_intra_cluster_correlation": sum(correlations) / len(correlations),
            "identity_query_clusters": len(ordered),
            "claim_eligible": (
                len(ordered) >= minimum_clusters
                and empirical >= minimum_empirical_coverage
            ),
        }

    audit.pop("audit_sha256", None)
    audit["schema_version"] = "circ-calibration-audit-v2"
    audit["empirical_concentration_coverage"] = concentration
    audit["target_concentration_semantics"] = {
        "outcome": "condition-wise identity-query-cluster helpful count",
        "router_probability_independent": True,
        "reason": "target concentration and router point calibration are separate audits",
    }
    audit["audit_sha256"] = hashlib.sha256(_canonical_json(audit)).hexdigest()
    return audit


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_overlay(
    *,
    parent_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    parent_root = parent_root.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("calibration overlay output must be new")
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != "circ-calibration-overlay-protocol-v1":
        raise ValueError("unsupported calibration overlay protocol")
    if int(protocol.get("official_test_access_count", -1)) != 0:
        raise ValueError("calibration overlay must not access official test")
    expected_artifacts = dict(protocol.get("parent_artifacts_sha256", {}))
    if set(expected_artifacts) != set(REQUIRED_PARENT_ARTIFACTS):
        raise ValueError("overlay protocol does not bind every parent artifact")
    parent_paths = {name: parent_root / name for name in REQUIRED_PARENT_ARTIFACTS}
    for name, path in parent_paths.items():
        if not path.is_file() or _sha256(path) != expected_artifacts[name]:
            raise ValueError(f"parent artifact is missing or changed: {name}")

    parent_scoring = _load_json(parent_paths["scoring_receipt.json"])
    parent_identity = _load_json(parent_paths["run_identity.json"])
    parent_cache = _load_json(parent_paths["cache/receipt.json"])
    target_bytes = parent_paths["cache/targets.jsonl"].read_bytes()
    if (
        parent_scoring.get("status") != "COMPLETE"
        or parent_scoring.get("scientific_evidence_eligible") is not False
        or parent_scoring.get("contract_testing") is not False
        or int(parent_scoring.get("official_test_access_count", -1)) != 0
        or int(parent_identity.get("official_test_access_count", -1)) != 0
        or int(parent_cache.get("official_test_access_count", -1)) != 0
        or parent_cache.get("targets_sha256") != hashlib.sha256(target_bytes).hexdigest()
        or parent_cache.get("protocol_hash")
        != protocol.get("parent_circ_protocol_sha256")
    ):
        raise ValueError("parent CIRC evidence is not the registered fail-closed run")

    rows = [json.loads(line) for line in target_bytes.decode("utf-8").splitlines() if line]
    method = dict(protocol.get("method", {}))
    audit = compute_cluster_calibration_audit(
        rows,
        nominal_coverage=float(method.get("nominal_coverage", -1.0)),
        minimum_empirical_coverage=float(
            method.get("minimum_empirical_coverage", -1.0)
        ),
        minimum_clusters=int(method.get("minimum_identity_query_clusters", -1)),
    )
    if not all(
        item.get("claim_eligible") is True
        for item in audit["empirical_concentration_coverage"].values()
    ):
        raise ValueError("cluster calibration overlay remains scientifically ineligible")

    protocol_sha256 = _sha256(protocol_path)
    implementation_sha256 = _sha256(Path(__file__).resolve())
    provenance = {
        "schema_version": "circ-calibration-overlay-v1",
        "parent_scoring_receipt_sha256": expected_artifacts["scoring_receipt.json"],
        "parent_cache_receipt_sha256": expected_artifacts["cache/receipt.json"],
        "parent_calibration_receipt_sha256": expected_artifacts[
            "cache/calibration_receipt.json"
        ],
        "parent_targets_sha256": expected_artifacts["cache/targets.jsonl"],
        "overlay_protocol_sha256": protocol_sha256,
        "overlay_implementation_sha256": implementation_sha256,
        "target_rows_changed": False,
        "official_test_access_count": 0,
    }

    cache = dict(parent_cache)
    cache["calibration_audit"] = audit
    cache["calibration_overlay"] = provenance
    calibration_receipt = {
        **audit,
        "schema_version": "circ-calibration-receipt-v2",
        "protocol_hash": cache["protocol_hash"],
        "targets_sha256": cache["targets_sha256"],
        "calibration_overlay": provenance,
    }
    _atomic_bytes(output_root / "cache/targets.jsonl", target_bytes)
    _atomic_json(output_root / "cache/receipt.json", cache)
    _atomic_json(output_root / "cache/calibration_receipt.json", calibration_receipt)
    for name in ("symmetry_receipt.json", "target_transfer_receipt.json"):
        _atomic_bytes(
            output_root / "cache" / name,
            parent_paths[f"cache/{name}"].read_bytes(),
        )
    _atomic_bytes(
        output_root / "run_identity.json",
        parent_paths["run_identity.json"].read_bytes(),
    )

    scoring = dict(parent_scoring)
    scoring.update(
        {
            "schema_version": "circ-scoring-receipt-v2",
            "cache_receipt_sha256": _sha256(output_root / "cache/receipt.json"),
            "calibration_receipt_sha256": _sha256(
                output_root / "cache/calibration_receipt.json"
            ),
            "calibration_audit": audit,
            "calibration_claim_eligible": True,
            "scientific_evidence_eligible": True,
            "calibration_overlay": provenance,
        }
    )
    _atomic_json(output_root / "scoring_receipt.json", scoring)
    overlay_receipt = {
        **provenance,
        "status": "COMPLETE",
        "output_cache_receipt_sha256": _sha256(output_root / "cache/receipt.json"),
        "output_calibration_receipt_sha256": _sha256(
            output_root / "cache/calibration_receipt.json"
        ),
        "output_scoring_receipt_sha256": _sha256(
            output_root / "scoring_receipt.json"
        ),
        "all_conditions_claim_eligible": True,
    }
    _atomic_json(output_root / "calibration_overlay_receipt.json", overlay_receipt)
    return overlay_receipt


def verify_overlay(
    *,
    parent_root: Path,
    protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Recompute the audit and verify every overlay binding before training."""

    parent_root = parent_root.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    protocol = _load_json(protocol_path)
    expected_artifacts = dict(protocol.get("parent_artifacts_sha256", {}))
    if (
        protocol.get("schema_version") != "circ-calibration-overlay-protocol-v1"
        or int(protocol.get("official_test_access_count", -1)) != 0
        or set(expected_artifacts) != set(REQUIRED_PARENT_ARTIFACTS)
    ):
        raise ValueError("invalid calibration overlay protocol")
    parent_paths = {name: parent_root / name for name in REQUIRED_PARENT_ARTIFACTS}
    for name, path in parent_paths.items():
        if not path.is_file() or _sha256(path) != expected_artifacts[name]:
            raise ValueError(f"parent artifact is missing or changed: {name}")

    required_output = (
        "run_identity.json",
        "scoring_receipt.json",
        "calibration_overlay_receipt.json",
        "cache/targets.jsonl",
        "cache/receipt.json",
        "cache/calibration_receipt.json",
        "cache/symmetry_receipt.json",
        "cache/target_transfer_receipt.json",
    )
    output_paths = {name: output_root / name for name in required_output}
    if any(not path.is_file() for path in output_paths.values()):
        raise ValueError("calibration overlay output is incomplete")
    parent_targets = parent_paths["cache/targets.jsonl"].read_bytes()
    output_targets = output_paths["cache/targets.jsonl"].read_bytes()
    if parent_targets != output_targets:
        raise ValueError("calibration overlay changed immutable target rows")
    if (
        parent_paths["run_identity.json"].read_bytes()
        != output_paths["run_identity.json"].read_bytes()
        or parent_paths["cache/symmetry_receipt.json"].read_bytes()
        != output_paths["cache/symmetry_receipt.json"].read_bytes()
        or parent_paths["cache/target_transfer_receipt.json"].read_bytes()
        != output_paths["cache/target_transfer_receipt.json"].read_bytes()
    ):
        raise ValueError("overlay changed non-calibration parent evidence")

    rows = [
        json.loads(line)
        for line in output_targets.decode("utf-8").splitlines()
        if line
    ]
    method = dict(protocol.get("method", {}))
    expected_audit = compute_cluster_calibration_audit(
        rows,
        nominal_coverage=float(method.get("nominal_coverage", -1.0)),
        minimum_empirical_coverage=float(
            method.get("minimum_empirical_coverage", -1.0)
        ),
        minimum_clusters=int(method.get("minimum_identity_query_clusters", -1)),
    )
    cache = _load_json(output_paths["cache/receipt.json"])
    calibration = _load_json(output_paths["cache/calibration_receipt.json"])
    scoring = _load_json(output_paths["scoring_receipt.json"])
    overlay = _load_json(output_paths["calibration_overlay_receipt.json"])
    provenance = dict(scoring.get("calibration_overlay", {}))
    expected_provenance = {
        "schema_version": "circ-calibration-overlay-v1",
        "parent_scoring_receipt_sha256": expected_artifacts[
            "scoring_receipt.json"
        ],
        "parent_cache_receipt_sha256": expected_artifacts["cache/receipt.json"],
        "parent_calibration_receipt_sha256": expected_artifacts[
            "cache/calibration_receipt.json"
        ],
        "parent_targets_sha256": expected_artifacts["cache/targets.jsonl"],
        "overlay_protocol_sha256": _sha256(protocol_path),
        "overlay_implementation_sha256": _sha256(Path(__file__).resolve()),
        "target_rows_changed": False,
        "official_test_access_count": 0,
    }
    if provenance != expected_provenance:
        raise ValueError("overlay provenance does not match current protocol and code")
    if (
        cache.get("calibration_audit") != expected_audit
        or cache.get("calibration_overlay") != expected_provenance
        or scoring.get("calibration_audit") != expected_audit
        or scoring.get("calibration_claim_eligible") is not True
        or scoring.get("scientific_evidence_eligible") is not True
        or int(scoring.get("official_test_access_count", -1)) != 0
        or scoring.get("cache_receipt_sha256")
        != _sha256(output_paths["cache/receipt.json"])
        or scoring.get("calibration_receipt_sha256")
        != _sha256(output_paths["cache/calibration_receipt.json"])
        or calibration.get("audit_sha256") != expected_audit["audit_sha256"]
        or calibration.get("calibration_overlay") != expected_provenance
        or overlay.get("output_cache_receipt_sha256")
        != _sha256(output_paths["cache/receipt.json"])
        or overlay.get("output_calibration_receipt_sha256")
        != _sha256(output_paths["cache/calibration_receipt.json"])
        or overlay.get("output_scoring_receipt_sha256")
        != _sha256(output_paths["scoring_receipt.json"])
        or overlay.get("all_conditions_claim_eligible") is not True
        or int(overlay.get("official_test_access_count", -1)) != 0
    ):
        raise ValueError("calibration overlay verification failed")
    return overlay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    operation = verify_overlay if args.verify_existing else build_overlay
    operation(
        parent_root=args.parent_root,
        protocol_path=args.protocol,
        output_root=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic full-network CIRC condition and margin scoring."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from .intervention_targets import valid_edges_for_mask
from .interventions import FullNetworkIntervention
from .state import EXPERT_ORDER, MODALITY_ORDER


CONDITION_FAMILIES = (
    "clean",
    "gaussian_blur",
    "occlusion",
    "exposure",
    "nir_noise",
    "thermal_noise",
    "modality_missing",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _condition_seed(sample_key: str, condition: Mapping[str, object]) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "domain": "TriFusion-CIRC-condition-v1",
                "sample_key": str(sample_key),
                "condition": dict(condition),
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def expand_registered_conditions(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, int | str], ...]:
    """Expand the frozen family/severity/seed registry without pooling seeds."""

    raw_conditions = list(protocol.get("conditions", []))
    if tuple(item.get("family") for item in raw_conditions) != CONDITION_FAMILIES:
        raise ValueError("CIRC condition family order differs from the frozen suite")
    expanded = []
    for item in raw_conditions:
        family = str(item["family"])
        severity = int(item["severity"])
        seeds = tuple(int(seed) for seed in item["seeds"])
        if severity < 0 or not seeds:
            raise ValueError("each CIRC condition needs nonnegative severity and seeds")
        for seed in seeds:
            expanded.append(
                {"family": family, "severity": severity, "seed": seed}
            )
    return tuple(expanded)


def select_training_conditions(
    sample_keys: Sequence[str],
    *,
    epoch: int,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, int | str], ...]:
    """Assign one deterministic condition per row with exact cycle coverage."""

    if epoch < 1:
        raise ValueError("CIRC training condition schedule needs epoch>=1")
    conditions = expand_registered_conditions(protocol)
    if not conditions:
        raise ValueError("CIRC training condition schedule needs conditions")
    selected = []
    for sample_key in sample_keys:
        digest = hashlib.sha256(
            _canonical_json(
                {
                    "domain": "TriFusion-CIRC-training-condition-v1",
                    "sample_key": str(sample_key),
                }
            )
        ).digest()
        offset = int.from_bytes(digest[:8], "big", signed=False) % len(conditions)
        selected.append(dict(conditions[(offset + epoch - 1) % len(conditions)]))
    return tuple(selected)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation vectors must align and contain at least two values")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def audit_query_gallery_symmetry(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_hash: str,
    epsilon: float,
    audit_specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit proxy symmetry with deterministic identity-clustered sampling."""

    sample_count = int(audit_specification["sample_rows"])
    if sample_count < 2 or len(rows) < sample_count:
        raise ValueError("symmetry audit lacks its frozen sample size")
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("symmetry audit epsilon must be finite and nonnegative")

    def row_digest(row: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "domain": "TriFusion-CIRC-symmetry-v1",
                    "protocol_hash": protocol_hash,
                    "sample_key": row["sample_key"],
                    "identity": int(row["identity"]),
                    "condition": dict(row["condition"]),
                    "contribution": row["contribution"],
                }
            )
        ).hexdigest()

    identity_clustered = bool(audit_specification["identity_clustered"])
    if identity_clustered:
        clusters: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            clusters[int(row["identity"])].append(row)
        for cluster_rows in clusters.values():
            cluster_rows.sort(key=row_digest)
        identity_order = sorted(
            clusters,
            key=lambda identity: hashlib.sha256(
                _canonical_json(
                    {
                        "domain": "TriFusion-CIRC-symmetry-identity-v1",
                        "protocol_hash": protocol_hash,
                        "identity": identity,
                    }
                )
            ).hexdigest(),
        )
        selected: list[Mapping[str, Any]] = []
        round_index = 0
        while len(selected) < sample_count:
            added = False
            for identity in identity_order:
                cluster_rows = clusters[identity]
                if round_index < len(cluster_rows):
                    selected.append(cluster_rows[round_index])
                    added = True
                    if len(selected) == sample_count:
                        break
            if not added:
                raise ValueError("identity clusters exhausted before symmetry sample size")
            round_index += 1
        selection_rule = (
            "identity-cluster round-robin by canonical SHA256 without replacement"
        )
    else:
        selected = sorted(rows, key=row_digest)[:sample_count]
        selection_rule = "lowest canonical SHA256 without replacement"

    query_only = [float(row["query_only_delta"]) for row in selected]
    symmetric = [float(row["symmetric_delta"]) for row in selected]
    if not all(math.isfinite(value) for value in (*query_only, *symmetric)):
        raise ValueError("symmetry audit effects must be finite")

    def sign(value: float) -> int:
        if value > epsilon:
            return 1
        if value < -epsilon:
            return -1
        return 0

    sign_agreement = sum(
        sign(left) == sign(right) for left, right in zip(query_only, symmetric)
    ) / sample_count
    spearman = _pearson(_average_ranks(query_only), _average_ranks(symmetric))
    minimum_sign = float(audit_specification["minimum_sign_agreement"])
    minimum_spearman = float(audit_specification["minimum_spearman"])
    passed = sign_agreement >= minimum_sign and spearman >= minimum_spearman
    selection_sha256 = hashlib.sha256(
        b"\n".join(_canonical_json(dict(row)) for row in selected)
    ).hexdigest()
    return {
        "status": "PASS" if passed else "FAIL",
        "claim_eligible": passed,
        "sample_rows": sample_count,
        "sampling_unit": "query-condition-contribution",
        "identity_clustered": identity_clustered,
        "selected_identity_count": len({int(row["identity"]) for row in selected}),
        "available_identity_clusters": len(
            {int(row["identity"]) for row in rows}
        ),
        "selection_rule": selection_rule,
        "selection_sha256": selection_sha256,
        "epsilon": epsilon,
        "sign_agreement": sign_agreement,
        "minimum_sign_agreement": minimum_sign,
        "spearman": spearman,
        "minimum_spearman": minimum_spearman,
    }


def select_proxy_transfer_rows(
    cache_rows: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    protocol_hash: str,
) -> tuple[dict[str, Any], ...]:
    """Select valid cache effects with deterministic identity-cluster coverage."""

    if sample_count < 2:
        raise ValueError("proxy-transfer audit needs at least two rows")
    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cache_row in cache_rows:
        identity = int(cache_row["identity"])
        for expert in EXPERT_ORDER:
            for modality in MODALITY_ORDER:
                contribution = f"{expert}.{modality}"
                value = dict(cache_row["contributions"][contribution])
                if not bool(value["valid"]):
                    continue
                clusters[identity].append(
                    {
                        "sample_key": str(cache_row["sample_key"]),
                        "identity": identity,
                        "camera": int(cache_row["camera"]),
                        "condition": dict(cache_row["condition"]),
                        "contribution": contribution,
                        "proxy_delta": float(value["effects"]["total"]),
                    }
                )
    available = sum(len(rows) for rows in clusters.values())
    if available < sample_count:
        raise ValueError("proxy-transfer audit lacks enough valid effects")

    def row_digest(row: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "domain": "TriFusion-CIRC-proxy-transfer-row-v1",
                    "protocol_hash": protocol_hash,
                    **dict(row),
                }
            )
        ).hexdigest()

    for cluster_rows in clusters.values():
        cluster_rows.sort(key=row_digest)
    identity_order = sorted(
        clusters,
        key=lambda identity: hashlib.sha256(
            _canonical_json(
                {
                    "domain": "TriFusion-CIRC-proxy-transfer-identity-v1",
                    "protocol_hash": protocol_hash,
                    "identity": identity,
                }
            )
        ).hexdigest(),
    )
    selected = []
    round_index = 0
    while len(selected) < sample_count:
        added = False
        for identity in identity_order:
            cluster_rows = clusters[identity]
            if round_index < len(cluster_rows):
                selected.append(dict(cluster_rows[round_index]))
                added = True
                if len(selected) == sample_count:
                    break
        if not added:
            raise ValueError("proxy-transfer clusters exhausted before sample size")
        round_index += 1
    return tuple(selected)


def audit_proxy_target_transfer(
    selected_rows: Sequence[Mapping[str, Any]],
    deployed_deltas: Sequence[float],
    router_scores: Sequence[float],
    *,
    epsilon: float,
    minimum_sign_agreement: float,
    minimum_spearman: float,
) -> dict[str, Any]:
    """Compare OOF proxy effects with effects from the frozen deployed model."""

    if not (
        len(selected_rows) == len(deployed_deltas) == len(router_scores)
    ) or len(selected_rows) < 2:
        raise ValueError("proxy, deployed and router transfer values must align")
    proxy = [float(row["proxy_delta"]) for row in selected_rows]
    deployed = [float(value) for value in deployed_deltas]
    router = [float(value) for value in router_scores]
    if not all(math.isfinite(value) for value in (*proxy, *deployed, *router)):
        raise ValueError("proxy-transfer audit effects must be finite")
    if any(value < 0.0 or value > 1.0 for value in router):
        raise ValueError("router helpfulness scores must lie in [0,1]")

    def sign(value: float) -> int:
        if value > epsilon:
            return 1
        if value < -epsilon:
            return -1
        return 0

    proxy_sign_agreement = sum(
        sign(left) == sign(right) for left, right in zip(proxy, deployed)
    ) / len(proxy)
    proxy_deployed_spearman = _pearson(
        _average_ranks(proxy), _average_ranks(deployed)
    )
    router_helpfulness_agreement = sum(
        (score > 0.5) == (delta > epsilon)
        for score, delta in zip(router, deployed)
    ) / len(router)
    router_deployed_spearman = _pearson(
        _average_ranks(router), _average_ranks(deployed)
    )
    passed = (
        proxy_sign_agreement >= float(minimum_sign_agreement)
        and proxy_deployed_spearman >= float(minimum_spearman)
        and router_helpfulness_agreement >= float(minimum_sign_agreement)
        and router_deployed_spearman >= float(minimum_spearman)
    )
    selection_sha256 = hashlib.sha256(
        b"\n".join(_canonical_json(dict(row)) for row in selected_rows)
    ).hexdigest()
    paired_sha256 = hashlib.sha256(
        b"\n".join(
            _canonical_json(
                {
                    **dict(row),
                    "deployed_delta": deployed_delta,
                    "router_score": router_score,
                }
            )
            for row, deployed_delta, router_score in zip(
                selected_rows, deployed, router
            )
        )
    ).hexdigest()
    return {
        "status": "PASS" if passed else "FAIL",
        "claim_eligible": passed,
        "sample_rows": len(selected_rows),
        "sampling_unit": "query-condition-contribution",
        "identity_clustered": True,
        "selected_identity_count": len(
            {int(row["identity"]) for row in selected_rows}
        ),
        "selection_rule": (
            "identity-cluster round-robin by canonical SHA256 without replacement"
        ),
        "selection_sha256": selection_sha256,
        "paired_effects_sha256": paired_sha256,
        "epsilon": float(epsilon),
        "proxy_sign_agreement": proxy_sign_agreement,
        "minimum_sign_agreement": float(minimum_sign_agreement),
        "proxy_deployed_spearman": proxy_deployed_spearman,
        "router_helpfulness_agreement": router_helpfulness_agreement,
        "router_deployed_spearman": router_deployed_spearman,
        "minimum_spearman": float(minimum_spearman),
    }


def _gaussian_kernel(
    *,
    kernel_size: int,
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coordinate = torch.arange(kernel_size, dtype=torch.float64, device=device)
    coordinate = coordinate - (kernel_size - 1) / 2.0
    one_dimensional = torch.exp(-(coordinate.square()) / (2.0 * sigma * sigma))
    one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = one_dimensional[:, None] * one_dimensional[None, :]
    return kernel.to(dtype=dtype).expand(3, 1, kernel_size, kernel_size)


def _blur(images: torch.Tensor, *, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        dtype=images.dtype,
        device=images.device,
    )
    padding = kernel_size // 2
    padded = functional.pad(images, (padding,) * 4, mode="reflect")
    return functional.conv2d(padded, kernel, groups=3)


def apply_registered_condition(
    images: Mapping[str, torch.Tensor],
    sample_keys: Sequence[str],
    condition: Mapping[str, object],
    *,
    operators: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply one frozen condition deterministically and return its availability mask."""

    if tuple(images) != MODALITY_ORDER:
        raise ValueError(f"condition images must be ordered as {MODALITY_ORDER}")
    if set(condition) != {"family", "severity", "seed"}:
        raise ValueError("condition must contain exactly family, severity and seed")
    family = str(condition["family"])
    if family not in CONDITION_FAMILIES or family not in operators:
        raise ValueError(f"unregistered CIRC condition family: {family}")
    batch_size = len(sample_keys)
    if any(tensor.shape[0] != batch_size for tensor in images.values()):
        raise ValueError("sample keys and condition image batches must align")
    if any(tensor.ndim != 4 or tensor.shape[1] != 3 for tensor in images.values()):
        raise ValueError("condition images must have shape B,3,H,W")
    specification = dict(operators[family])
    if int(condition["severity"]) != int(specification["severity"]):
        raise ValueError("condition severity differs from its frozen operator")
    conditioned = {name: tensor for name, tensor in images.items()}
    modality_mask = torch.ones(
        batch_size,
        3,
        dtype=torch.bool,
        device=next(iter(images.values())).device,
    )

    if family == "clean":
        if specification.get("operator") != "identity":
            raise ValueError("clean condition must be the identity operator")
        return conditioned, modality_mask

    if family == "gaussian_blur":
        kernel_size = int(specification["kernel_size"])
        sigma = float(specification["sigma"])
        if kernel_size < 3 or kernel_size % 2 == 0 or sigma <= 0:
            raise ValueError("invalid frozen Gaussian blur parameters")
        conditioned = {
            name: _blur(tensor, kernel_size=kernel_size, sigma=sigma)
            for name, tensor in images.items()
        }
        return conditioned, modality_mask

    if family == "occlusion":
        fraction = float(specification["side_fraction"])
        fill = float(specification["normalized_fill"])
        if not 0.0 < fraction < 1.0 or not math.isfinite(fill):
            raise ValueError("invalid frozen occlusion parameters")
        conditioned = {name: tensor.clone() for name, tensor in images.items()}
        height, width = next(iter(images.values())).shape[-2:]
        box_height = max(1, int(round(height * fraction)))
        box_width = max(1, int(round(width * fraction)))
        for row, sample_key in enumerate(sample_keys):
            seed = _condition_seed(sample_key, condition)
            top = seed % (height - box_height + 1)
            left = (seed // max(1, height - box_height + 1)) % (
                width - box_width + 1
            )
            for tensor in conditioned.values():
                tensor[row, :, top : top + box_height, left : left + box_width] = fill
        return conditioned, modality_mask

    if family == "exposure":
        modality = str(specification["modality"])
        raw_scale = float(specification["raw_scale"])
        if modality != "RGB" or not 0.0 < raw_scale < 1.0:
            raise ValueError("invalid frozen RGB exposure parameters")
        raw = ((images[modality] + 1.0) * 0.5).clamp(0.0, 1.0)
        conditioned[modality] = (raw * raw_scale).clamp(0.0, 1.0) * 2.0 - 1.0
        return conditioned, modality_mask

    if family in ("nir_noise", "thermal_noise"):
        expected_modality = "NI" if family == "nir_noise" else "TI"
        modality = str(specification["modality"])
        sigma = float(specification["normalized_sigma"])
        if modality != expected_modality or sigma <= 0:
            raise ValueError("invalid frozen sensor-noise parameters")
        corrupted = images[modality].clone()
        for row, sample_key in enumerate(sample_keys):
            generator = torch.Generator(device=corrupted.device)
            generator.manual_seed(_condition_seed(sample_key, condition))
            noise = torch.randn(
                corrupted[row].shape,
                dtype=corrupted.dtype,
                device=corrupted.device,
                generator=generator,
            )
            corrupted[row] = (corrupted[row] + sigma * noise).clamp(-1.0, 1.0)
        conditioned[modality] = corrupted
        return conditioned, modality_mask

    if family == "modality_missing":
        if specification.get("selection") != "sample-hash-mod-3":
            raise ValueError("invalid frozen missing-modality selection")
        conditioned = {name: tensor.clone() for name, tensor in images.items()}
        for row, sample_key in enumerate(sample_keys):
            modality_index = _condition_seed(sample_key, condition) % len(MODALITY_ORDER)
            modality_mask[row, modality_index] = False
            conditioned[MODALITY_ORDER[modality_index]][row].zero_()
        return conditioned, modality_mask

    raise AssertionError("unreachable registered condition family")


def apply_registered_condition_batch(
    images: Mapping[str, torch.Tensor],
    sample_keys: Sequence[str],
    conditions: Sequence[Mapping[str, object]],
    *,
    operators: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply potentially different registered conditions within one training batch."""

    if len(sample_keys) != len(conditions):
        raise ValueError("sample keys and per-row conditions must align")
    conditioned = {name: tensor.clone() for name, tensor in images.items()}
    device = next(iter(images.values())).device
    modality_mask = torch.ones(
        len(sample_keys),
        len(MODALITY_ORDER),
        dtype=torch.bool,
        device=device,
    )
    grouped: dict[bytes, list[int]] = defaultdict(list)
    condition_by_key: dict[bytes, Mapping[str, object]] = {}
    for row, condition in enumerate(conditions):
        key = _canonical_json(dict(condition))
        grouped[key].append(row)
        condition_by_key[key] = condition
    for key, row_indices in grouped.items():
        index = torch.tensor(row_indices, dtype=torch.long, device=device)
        selected_images = {
            name: tensor.index_select(0, index) for name, tensor in images.items()
        }
        selected_keys = [sample_keys[row] for row in row_indices]
        selected_images, selected_mask = apply_registered_condition(
            selected_images,
            selected_keys,
            condition_by_key[key],
            operators=operators,
        )
        for modality in MODALITY_ORDER:
            conditioned[modality].index_copy_(0, index, selected_images[modality])
        modality_mask.index_copy_(0, index, selected_mask)
    return conditioned, modality_mask


@dataclass(frozen=True, eq=False)
class ReferenceMarginBank:
    """Fixed reference prototypes for query-side cross-camera retrieval margins."""

    positive_prototypes: torch.Tensor
    identity_prototypes: torch.Tensor
    query_identity_indices: torch.Tensor
    query_identities: tuple[int, ...]
    query_cameras: tuple[int, ...]
    cross_camera_positive_cameras: tuple[tuple[int, ...], ...]

    def margins(
        self,
        query_embeddings: torch.Tensor,
        query_indices: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        if query_indices is None:
            query_indices = torch.arange(query_embeddings.shape[0])
        indices = torch.as_tensor(query_indices, dtype=torch.long)
        if indices.numel() != query_embeddings.shape[0]:
            raise ValueError("query embeddings and margin-bank indices must align")
        device = query_embeddings.device
        normalized = functional.normalize(query_embeddings.float(), dim=1)
        positive = self.positive_prototypes.index_select(0, indices).to(device)
        identities = self.identity_prototypes.to(device)
        owner = self.query_identity_indices.index_select(0, indices).to(device)
        positive_similarity = (normalized * positive).sum(dim=1)
        negative_similarity = normalized @ identities.transpose(0, 1)
        negative_similarity.scatter_(1, owner[:, None], float("-inf"))
        return positive_similarity - negative_similarity.max(dim=1).values


def build_reference_margin_bank(
    reference_embeddings: torch.Tensor,
    reference_identities: Sequence[int],
    reference_cameras: Sequence[int],
    query_identities: Sequence[int],
    query_cameras: Sequence[int],
) -> ReferenceMarginBank:
    """Build fixed identity and leave-camera-out positive prototypes."""

    if reference_embeddings.ndim != 2 or not reference_embeddings.shape[0]:
        raise ValueError("reference embeddings must be a nonempty N,D tensor")
    if not (
        len(reference_identities)
        == len(reference_cameras)
        == reference_embeddings.shape[0]
    ):
        raise ValueError("reference metadata does not align with embeddings")
    if len(query_identities) != len(query_cameras) or not query_identities:
        raise ValueError("query identities and cameras must be nonempty and aligned")
    reference = functional.normalize(reference_embeddings.float(), dim=1).cpu()
    identity_values = tuple(sorted({int(value) for value in reference_identities}))
    if len(identity_values) < 2:
        raise ValueError("retrieval margin bank requires at least two identities")
    identity_to_index = {identity: index for index, identity in enumerate(identity_values)}
    identity_prototypes = []
    for identity in identity_values:
        indices = [
            index
            for index, value in enumerate(reference_identities)
            if int(value) == identity
        ]
        identity_prototypes.append(
            functional.normalize(reference[indices].mean(dim=0), dim=0)
        )

    positive_prototypes = []
    query_identity_indices = []
    positive_camera_rows = []
    for identity_value, camera_value in zip(query_identities, query_cameras):
        identity = int(identity_value)
        camera = int(camera_value)
        if identity not in identity_to_index:
            raise ValueError("query identity is absent from the fixed reference bank")
        positive_indices = [
            index
            for index, (reference_identity, reference_camera) in enumerate(
                zip(reference_identities, reference_cameras)
            )
            if int(reference_identity) == identity and int(reference_camera) != camera
        ]
        if not positive_indices:
            raise ValueError("query lacks a different-camera positive reference")
        positive_prototypes.append(
            functional.normalize(reference[positive_indices].mean(dim=0), dim=0)
        )
        query_identity_indices.append(identity_to_index[identity])
        positive_camera_rows.append(
            tuple(
                sorted(
                    {
                        int(reference_cameras[index])
                        for index in positive_indices
                    }
                )
            )
        )
    return ReferenceMarginBank(
        positive_prototypes=torch.stack(positive_prototypes),
        identity_prototypes=torch.stack(identity_prototypes),
        query_identity_indices=torch.tensor(query_identity_indices, dtype=torch.long),
        query_identities=tuple(int(value) for value in query_identities),
        query_cameras=tuple(int(value) for value in query_cameras),
        cross_camera_positive_cameras=tuple(positive_camera_rows),
    )


def forward_fused_embeddings(
    model: torch.nn.Module,
    images: Mapping[str, torch.Tensor],
    modality_mask: torch.Tensor,
    indices: Sequence[int] | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
    intervention: FullNetworkIntervention | None = None,
) -> torch.Tensor:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    chunks = []
    for start in range(0, index_tensor.numel(), batch_size):
        selected = index_tensor[start : start + batch_size]
        batch = {
            "images": {
                name: tensor.index_select(0, selected).to(device, non_blocking=False)
                for name, tensor in images.items()
            },
            "modality_mask": modality_mask.index_select(0, selected).to(
                device, non_blocking=False
            ),
        }
        if intervention is not None:
            batch["intervention"] = intervention
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            output = model(batch, return_aux=True)
        if not hasattr(output, "fused_embedding"):
            raise TypeError("CIRC scorer requires a collaborative fused embedding")
        embedding = functional.normalize(output.fused_embedding.detach().float(), dim=1)
        if not torch.isfinite(embedding).all():
            raise FloatingPointError("nonfinite full-network intervention embedding")
        chunks.append(embedding.cpu())
    return torch.cat(chunks, dim=0)


def score_registered_condition(
    model: torch.nn.Module,
    images: Mapping[str, torch.Tensor],
    sample_keys: Sequence[str],
    query_identities: Sequence[int],
    query_cameras: Sequence[int],
    reference_bank: ReferenceMarginBank,
    condition: Mapping[str, object],
    *,
    operators: Mapping[str, Mapping[str, object]],
    protocol_hash: str,
    generator_training_identities: Sequence[int],
    generator_checkpoint_sha256: str,
    reference_bank_sha256: str,
    batch_size: int,
    device: torch.device,
    amp: bool,
    symmetric_banks: Mapping[str, ReferenceMarginBank] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score base, total/direct/relay and two sampled full-network edge removals."""

    if model.training:
        raise ValueError("CIRC interventions require a frozen evaluation-mode model")
    row_count = len(sample_keys)
    if not (
        row_count == len(query_identities) == len(query_cameras)
        and row_count == reference_bank.positive_prototypes.shape[0]
    ):
        raise ValueError("query images, metadata and reference bank must align")
    conditioned, modality_mask = apply_registered_condition(
        images,
        sample_keys,
        condition,
        operators=operators,
    )
    all_indices = torch.arange(row_count, dtype=torch.long)
    baseline_embeddings = forward_fused_embeddings(
        model,
        conditioned,
        modality_mask,
        all_indices,
        batch_size=batch_size,
        device=device,
        amp=amp,
    )
    baseline_margins = reference_bank.margins(baseline_embeddings).cpu()
    effects: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    symmetry_values: dict[str, torch.Tensor] = {}
    for expert in EXPERT_ORDER:
        for modality_index, modality in enumerate(MODALITY_ORDER):
            contribution = f"{expert}.{modality}"
            for kind in ("total", "direct", "relay"):
                intervention = FullNetworkIntervention(
                    kind=kind,
                    expert=expert,
                    modality=modality,
                )
                intervened_embeddings = forward_fused_embeddings(
                    model,
                    conditioned,
                    modality_mask,
                    all_indices,
                    batch_size=batch_size,
                    device=device,
                    amp=amp,
                    intervention=intervention,
                )
                intervened_margins = reference_bank.margins(
                    intervened_embeddings
                ).cpu()
                delta = baseline_margins - intervened_margins
                invalid = ~modality_mask[:, modality_index]
                if invalid.any():
                    if not torch.allclose(
                        delta[invalid],
                        torch.zeros_like(delta[invalid]),
                        atol=1e-6,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            "missing-modality contribution had a nonzero query-only effect"
                        )
                    delta[invalid] = 0.0
                effects[contribution][kind] = delta
                if kind == "total" and symmetric_banks is not None:
                    if contribution not in symmetric_banks:
                        raise KeyError(f"missing symmetric bank for {contribution}")
                    symmetric_margin = symmetric_banks[contribution].margins(
                        intervened_embeddings
                    ).cpu()
                    symmetry_values[contribution] = baseline_margins - symmetric_margin

    edge_effects: dict[int, list[dict[str, float]]] = {
        stage: [dict() for _ in range(row_count)] for stage in (1, 2)
    }
    from .intervention_targets import select_audit_edge

    for stage in (1, 2):
        grouped: dict[str, list[int]] = defaultdict(list)
        for row, (sample_key, mask_row) in enumerate(
            zip(sample_keys, modality_mask.tolist())
        ):
            selected = select_audit_edge(
                valid_edges_for_mask(mask_row),
                protocol_hash=protocol_hash,
                sample_key=str(sample_key),
                condition=condition,
                stage=stage,
            )
            grouped[selected.edge].append(row)
        for edge, group_indices in grouped.items():
            source_target, modality = edge.split(":", maxsplit=1)
            source, target = source_target.split("->", maxsplit=1)
            intervention = FullNetworkIntervention(
                kind="edge",
                stage=stage,
                source=source,
                target=target,
                modality=modality,
            )
            embeddings = forward_fused_embeddings(
                model,
                conditioned,
                modality_mask,
                group_indices,
                batch_size=batch_size,
                device=device,
                amp=amp,
                intervention=intervention,
            )
            margins = reference_bank.margins(embeddings, group_indices).cpu()
            delta = baseline_margins[group_indices] - margins
            for local_index, row in enumerate(group_indices):
                edge_effects[stage][row][edge] = float(delta[local_index])

    rows = []
    symmetry_rows = []
    generator_identities = sorted(int(value) for value in generator_training_identities)
    for row, sample_key in enumerate(sample_keys):
        contribution_effects = {
            contribution: {
                kind: float(effects[contribution][kind][row])
                for kind in ("total", "direct", "relay")
            }
            for contribution in effects
        }
        rows.append(
            {
                "sample_key": str(sample_key),
                "identity": int(query_identities[row]),
                "camera": int(query_cameras[row]),
                "dataset_split": "train",
                "modality_mask": [bool(value) for value in modality_mask[row].tolist()],
                "cross_camera_positive_cameras": list(
                    reference_bank.cross_camera_positive_cameras[row]
                ),
                "condition": dict(condition),
                "generator_training_identities": generator_identities,
                "generator_checkpoint_sha256": generator_checkpoint_sha256,
                "reference_bank_sha256": reference_bank_sha256,
                "intervention_seeds": [int(condition["seed"])],
                "interventions": contribution_effects,
                "edge_effects": {
                    str(stage): edge_effects[stage][row] for stage in (1, 2)
                },
            }
        )
        for contribution, symmetric_delta in symmetry_values.items():
            modality = contribution.split(".", maxsplit=1)[1]
            modality_index = MODALITY_ORDER.index(modality)
            if modality_mask[row, modality_index]:
                symmetry_rows.append(
                    {
                        "sample_key": str(sample_key),
                        "identity": int(query_identities[row]),
                        "condition": dict(condition),
                        "contribution": contribution,
                        "query_only_delta": float(
                            effects[contribution]["total"][row]
                        ),
                        "symmetric_delta": float(symmetric_delta[row]),
                    }
                )
    return rows, symmetry_rows


__all__ = [
    "CONDITION_FAMILIES",
    "ReferenceMarginBank",
    "apply_registered_condition",
    "apply_registered_condition_batch",
    "audit_proxy_target_transfer",
    "audit_query_gallery_symmetry",
    "build_reference_margin_bank",
    "expand_registered_conditions",
    "forward_fused_embeddings",
    "score_registered_condition",
    "select_proxy_transfer_rows",
    "select_training_conditions",
]

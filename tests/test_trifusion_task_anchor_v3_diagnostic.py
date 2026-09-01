from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch


def test_diagnostic_views_reconstruct_uniform_quality_routing() -> None:
    from tools.diagnose_trifusion_task_anchor_v3 import derive_diagnostic_views

    anchor = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    contributions = torch.tensor(
        [
            [
                [[0.25, 0.0], [0.25, 0.0], [0.25, 0.0]],
                [[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]],
                [[-0.25, 0.0], [-0.25, 0.0], [-0.25, 0.0]],
            ]
        ]
    )
    reliability = torch.full((1, 3, 3), 0.8)
    uncertainty = torch.full((1, 3, 3), 0.25)
    mask = torch.ones(1, 3, dtype=torch.bool)

    result = derive_diagnostic_views(
        anchor_embedding=anchor,
        contribution_embeddings=contributions,
        reliability=reliability,
        uncertainty=uncertainty,
        modality_mask=mask,
    )

    expected_routed = torch.tensor([[0.0, 1.0 / 6.0] * 3])
    assert tuple(result.views) == (
        "anchor",
        "fused",
        "cnn",
        "transformer",
        "mamba",
        "routed_residual",
    )
    assert torch.allclose(result.views["routed_residual"], expected_routed)
    assert torch.allclose(result.views["fused"], torch.cat((anchor, expected_routed), 1))
    assert torch.equal(
        result.views["cnn"], torch.cat((anchor, contributions[:, 0].flatten(1)), 1)
    )
    assert torch.equal(
        result.views["transformer"],
        torch.cat((anchor, contributions[:, 1].flatten(1)), 1),
    )
    assert torch.equal(
        result.views["mamba"], torch.cat((anchor, contributions[:, 2].flatten(1)), 1)
    )
    assert torch.allclose(result.routing_weights, torch.full((1, 3, 3), 1.0 / 3.0))
    assert torch.allclose(
        result.expert_to_anchor_norm_ratio,
        torch.tensor([[[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]]),
    )
    assert torch.allclose(
        result.routed_to_anchor_norm_ratio,
        torch.full((1, 3), 1.0 / 6.0),
    )
    assert math.isclose(float(result.normalized_routing_entropy.mean()), 1.0, abs_tol=1e-6)


def test_diagnostic_views_zero_missing_modalities_without_renormalizing_experts() -> None:
    from tools.diagnose_trifusion_task_anchor_v3 import derive_diagnostic_views

    anchor = torch.tensor([[3.0, 4.0, 9.0, 9.0, 0.0, 2.0]])
    contributions = torch.ones(1, 3, 3, 2)
    reliability = torch.tensor(
        [[[1.0, 100.0, 1.0], [2.0, 100.0, 2.0], [3.0, 100.0, 3.0]]]
    )
    uncertainty = torch.zeros_like(reliability)
    mask = torch.tensor([[True, False, True]])

    result = derive_diagnostic_views(
        anchor_embedding=anchor,
        contribution_embeddings=contributions,
        reliability=reliability,
        uncertainty=uncertainty,
        modality_mask=mask,
    )

    assert torch.equal(result.routing_weights[:, :, 1], torch.zeros(1, 3))
    assert torch.allclose(
        result.routing_weights[:, :, (0, 2)],
        torch.tensor([[[1.0 / 6.0, 1.0 / 6.0], [2.0 / 6.0, 2.0 / 6.0], [3.0 / 6.0, 3.0 / 6.0]]]),
    )
    assert torch.equal(result.views["routed_residual"][:, 2:4], torch.zeros(1, 2))
    assert torch.equal(result.routed_to_anchor_norm_ratio[:, 1], torch.zeros(1))


def test_frozen_dev_validation_rejects_checkpoint_outside_run_directory(
    tmp_path: Path,
) -> None:
    from tools.diagnose_trifusion_task_anchor_v3 import validate_frozen_dev_artifacts

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    checkpoint = tmp_path / "foreign.pth"
    checkpoint.write_bytes(b"foreign checkpoint")
    config = tmp_path / "config.yml"
    config.write_text("MODEL: {}\n", encoding="utf-8")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    metrics = {"fused": {"mAP": 42.0}}
    best = {
        "phase": "complete",
        "contract_testing": False,
        "scientific_evidence_eligible": True,
        "official_test_access_count": 0,
        "dev_evaluation_count": 60,
        "selection_output": "fused",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "epoch": 14,
        "metrics_percent": metrics,
    }
    worker = {
        "status": "COMPLETE",
        "phase": "complete",
        "epoch": 60,
        "dev_evaluation_count": 60,
        "official_test_access_count": 0,
        "fatal_or_nonfinite_detected": False,
        "scientific_evidence_eligible": True,
        "best_epoch": 14,
        "best_checkpoint_sha256": checkpoint_hash,
        "metrics_percent": metrics,
    }
    (output_dir / "best_dev_receipt.json").write_text(
        json.dumps(best), encoding="utf-8"
    )
    (output_dir / "dev_worker_result.json").write_text(
        json.dumps(worker), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="outside"):
        validate_frozen_dev_artifacts(output_dir=output_dir, config_path=config)

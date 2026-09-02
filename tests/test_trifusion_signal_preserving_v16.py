from __future__ import annotations

from types import SimpleNamespace

import torch


def test_v16_signal_hard_pairs_use_physical_cross_camera_positives() -> None:
    from modeling.trifusion.signal_preserving_v16 import (
        select_signal_hard_pairs_v16,
    )

    baseline = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],  # same identity/camera as query 0: must be ignored
            [0.8, 0.6],
            [0.9, 0.4],
            [0.0, 1.0],
            [-0.8, -0.6],
        ]
    )
    identities = torch.tensor([0, 0, 0, 1, 1, 2])
    physical_cameras = torch.tensor([0, 0, 1, 0, 1, 0])

    pairs = select_signal_hard_pairs_v16(
        baseline,
        identities,
        physical_cameras,
    )

    assert pairs.positive_indices[0].item() == 2
    assert pairs.negative_indices[0].item() == 3
    assert torch.equal(
        pairs.valid_query_mask,
        torch.tensor([True, True, True, True, True, False]),
    )
    assert pairs.positive_indices[5].item() == -1
    assert pairs.negative_indices[5].item() == -1


def test_v16_satr_uses_two_detached_peers_to_repair_only_the_live_receiver() -> None:
    from modeling.trifusion.signal_preserving_v16 import (
        satr_relation_objective_v16,
    )

    def directions(degrees: list[float], *, grad: bool) -> torch.Tensor:
        radians = torch.deg2rad(torch.tensor(degrees))
        value = torch.stack((torch.cos(radians), torch.sin(radians)), dim=1)
        return value.requires_grad_(grad)

    baseline = directions([0.0, 60.0, 30.0, 180.0], grad=True)
    branches = {
        "cnn": directions([0.0, 80.0, 20.0, 180.0], grad=True),
        "transformer": directions([0.0, 10.0, 100.0, 180.0], grad=True),
        "mamba": directions([0.0, 15.0, 100.0, 180.0], grad=True),
    }
    fused = directions([0.0, 60.0, 30.0, 180.0], grad=True)
    identities = torch.tensor([0, 0, 1, 1])
    physical_cameras = torch.tensor([0, 1, 0, 1])

    objective = satr_relation_objective_v16(
        baseline,
        fused,
        branches,
        identities,
        physical_cameras,
    )
    objective.repair_losses["cnn"].backward()

    assert objective.eligible_masks["cnn"][0].item() is True
    assert branches["cnn"].grad is not None
    assert torch.count_nonzero(branches["cnn"].grad) > 0
    assert branches["transformer"].grad is None
    assert branches["mamba"].grad is None
    assert baseline.grad is None
    assert fused.grad is None


def test_v16_criterion_combines_registered_reid_weights_with_satr() -> None:
    from modeling.trifusion.signal_preserving_v16 import (
        SATRV16Criterion,
        V16_ID_BRANCH_WEIGHT,
        V16_ID_FUSED_WEIGHT,
        V16_ID_RESIDUAL_WEIGHT,
        V16_TRIPLET_BRANCH_WEIGHT,
        V16_TRIPLET_FUSED_WEIGHT,
        V16_TRIPLET_RESIDUAL_WEIGHT,
    )

    torch.manual_seed(16)
    labels = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])
    output = SimpleNamespace(
        baseline_embedding=torch.randn(4, 6),
        fused_embedding=torch.randn(4, 9, requires_grad=True),
        branch_embeddings={
            expert: torch.randn(4, 9, requires_grad=True)
            for expert in ("cnn", "transformer", "mamba")
        },
        residual_embeddings={
            expert: torch.randn(4, 3, requires_grad=True)
            for expert in ("cnn", "transformer", "mamba")
        },
        fused_logits=torch.randn(4, 2, requires_grad=True),
        branch_logits={
            expert: torch.randn(4, 2, requires_grad=True)
            for expert in ("cnn", "transformer", "mamba")
        },
        residual_logits={
            expert: torch.randn(4, 2, requires_grad=True)
            for expert in ("cnn", "transformer", "mamba")
        },
    )

    losses = SATRV16Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
    )(output, labels, cameras)
    expected_supervised = (
        V16_ID_FUSED_WEIGHT * losses["id_fused"]
        + V16_TRIPLET_FUSED_WEIGHT * losses["triplet_fused"]
        + sum(
            V16_ID_BRANCH_WEIGHT * losses[f"id_{expert}"]
            + V16_TRIPLET_BRANCH_WEIGHT * losses[f"triplet_{expert}"]
            + V16_ID_RESIDUAL_WEIGHT * losses[f"id_residual_{expert}"]
            + V16_TRIPLET_RESIDUAL_WEIGHT * losses[f"triplet_residual_{expert}"]
            for expert in ("cnn", "transformer", "mamba")
        )
    )

    assert torch.equal(losses["supervised_total"], expected_supervised)
    assert torch.equal(
        losses["total"],
        losses["supervised_total"] + losses["satr_total"],
    )
    assert set(losses) >= {
        "satr_cnn",
        "satr_transformer",
        "satr_mamba",
        "satr_protection",
        "coverage_cnn",
        "coverage_transformer",
        "coverage_mamba",
    }
    losses["total"].backward()
    assert output.fused_logits.grad is not None
    assert all(value.grad is not None for value in output.branch_logits.values())
    assert all(value.grad is not None for value in output.residual_logits.values())

    no_satr = SATRV16Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
        satr_enabled=False,
    )(output, labels, cameras)
    assert torch.equal(no_satr["total"], no_satr["supervised_total"])
    assert torch.equal(no_satr["satr_total"], torch.zeros_like(no_satr["satr_total"]))

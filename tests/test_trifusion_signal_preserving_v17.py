from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _directions(degrees: list[float], *, grad: bool) -> torch.Tensor:
    radians = torch.deg2rad(torch.tensor(degrees))
    values = torch.stack((torch.cos(radians), torch.sin(radians)), dim=1)
    return values.requires_grad_(grad)


class _TinyFrozenV8(nn.Module):
    baseline_embedding_width = 3
    residual_embedding_width = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        batch: dict[str, object],
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
        retrieval_output: str = "fused",
    ) -> SimpleNamespace:
        del targets, return_aux, retrieval_output
        return SimpleNamespace(
            baseline_embedding=batch["baseline"] * self.scale,
            residual_embeddings={
                expert: batch["residuals"][expert] * self.scale
                for expert in ("cnn", "transformer", "mamba")
            },
        )


def test_v17_relation_envelope_is_one_sided_balanced_and_teacher_detached() -> None:
    from modeling.trifusion.signal_preserving_v17 import (
        relation_envelope_objective_v17,
    )

    teachers = {
        expert: _directions([0.0, 90.0, 180.0, 270.0], grad=True)
        for expert in ("cnn", "transformer", "mamba")
    }
    perfect = [0.0, 0.0, 180.0, 180.0]
    corrected = {
        "cnn": _directions([0.0, 180.0, 0.0, 180.0], grad=True),
        "transformer": _directions(perfect, grad=True),
        "mamba": _directions(perfect, grad=True),
    }
    fused = _directions(perfect, grad=True)
    identities = torch.tensor([0, 0, 1, 1])
    physical_cameras = torch.tensor([0, 1, 0, 1])

    objective = relation_envelope_objective_v17(
        teachers,
        fused,
        corrected,
        identities,
        physical_cameras,
    )

    assert objective.fused_positive.item() == pytest.approx(0.0, abs=1e-7)
    assert objective.fused_negative.item() == pytest.approx(0.0, abs=1e-7)
    assert objective.branch_positive["cnn"].item() == pytest.approx(1.0)
    assert objective.branch_negative["cnn"].item() == pytest.approx(2.0)
    assert objective.total.item() == pytest.approx(0.25)
    assert sum(objective.positive_source_counts.values()) == 4
    assert sum(objective.negative_source_counts.values()) == 8

    objective.total.backward()

    assert corrected["cnn"].grad is not None
    assert torch.count_nonzero(corrected["cnn"].grad) > 0
    assert all(teacher.grad is None for teacher in teachers.values())


def test_v17_triadic_correction_is_normalized_peer_coupled_and_trainable() -> None:
    from modeling.trifusion.signal_preserving_v17 import TriadicCorrectionV17

    torch.manual_seed(17)
    correction = TriadicCorrectionV17(residual_width=4, adapter_width=3)
    residuals = {
        expert: torch.randn(5, 4, requires_grad=True)
        for expert in ("cnn", "transformer", "mamba")
    }

    output = correction(residuals)

    assert tuple(output.corrected_residuals) == ("cnn", "transformer", "mamba")
    assert output.fused_residual.shape == (5, 12)
    assert torch.allclose(
        output.fused_residual.norm(dim=1),
        torch.ones(5),
        atol=1e-6,
    )
    for value in output.corrected_residuals.values():
        assert value.shape == (5, 4)
        assert torch.allclose(value.norm(dim=1), torch.ones(5), atol=1e-6)

    changed_peer = dict(residuals)
    changed_peer["mamba"] = residuals["mamba"] + torch.tensor(
        [1.0, 0.0, 0.0, 0.0]
    )
    changed_output = correction(changed_peer)
    assert not torch.allclose(
        output.corrected_residuals["cnn"],
        changed_output.corrected_residuals["cnn"],
    )

    weights = torch.arange(1, 13, dtype=output.fused_residual.dtype)
    loss = (output.fused_residual * weights).sum()
    loss = loss + sum(
        (value * torch.arange(1, 5, dtype=value.dtype)).sum()
        for value in output.corrected_residuals.values()
    )
    loss.backward()

    assert all(value.grad is not None for value in residuals.values())
    assert all(parameter.grad is not None for parameter in correction.parameters())


def test_v17_model_freezes_v8_and_preserves_exact_signal_retrieval_prefix() -> None:
    from modeling.trifusion.signal_preserving_v17 import (
        SignalPreservingCollaborativeV17,
    )

    torch.manual_seed(1701)
    base_v8 = _TinyFrozenV8()
    model = SignalPreservingCollaborativeV17(
        base_v8=base_v8,
        num_classes=2,
        adapter_width=3,
    ).train()
    batch = {
        "baseline": torch.randn(4, 3),
        "residuals": {
            expert: torch.randn(4, 2)
            for expert in ("cnn", "transformer", "mamba")
        },
    }

    output = model(batch, return_aux=True)

    assert base_v8.training is False
    assert all(not parameter.requires_grad for parameter in base_v8.parameters())
    assert torch.equal(output.baseline_embedding, batch["baseline"])
    assert torch.equal(output.fused_embedding[:, :3], batch["baseline"])
    for expert in ("cnn", "transformer", "mamba"):
        assert torch.equal(output.branch_embeddings[expert][:, :3], batch["baseline"])
        assert output.residual_embeddings[expert].shape == (4, 2)
        assert output.residual_logits[expert].shape == (4, 2)
    assert output.fused_embedding.shape == (4, 9)
    assert output.fused_logits.shape == (4, 2)
    assert output.diagnostics["baseline_exact_prefix"] is True
    assert output.diagnostics["v8_frozen"] is True

    assert torch.equal(model(batch, retrieval_output="baseline_only"), batch["baseline"])
    assert model(batch, retrieval_output="fused").shape == (4, 9)
    assert model(batch, retrieval_output="cnn").shape == (4, 5)

    loss = output.fused_logits.sum()
    loss = loss + sum(value.sum() for value in output.residual_logits.values())
    loss.backward()
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.endswith(".bias")
    )


def test_v17_criterion_matches_the_registered_objective_and_weight_zero_control() -> None:
    from modeling.trifusion.signal_preserving_v17 import (
        DenseTriadicV17Criterion,
        SignalPreservingCollaborativeV17,
    )

    torch.manual_seed(1702)
    model = SignalPreservingCollaborativeV17(
        base_v8=_TinyFrozenV8(),
        num_classes=2,
        adapter_width=3,
    ).train()
    batch = {
        "baseline": torch.randn(4, 3),
        "residuals": {
            expert: torch.randn(4, 2)
            for expert in ("cnn", "transformer", "mamba")
        },
    }
    labels = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])
    output = model(batch, return_aux=True)

    losses = DenseTriadicV17Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
        envelope_enabled=True,
    )(output, labels, cameras)
    expected_supervised = losses["id_fused"] + losses["triplet_fused"]
    expected_supervised = expected_supervised + sum(
        (losses[f"id_{expert}"] + losses[f"triplet_{expert}"]) / 3.0
        for expert in ("cnn", "transformer", "mamba")
    )

    assert torch.equal(losses["supervised_total"], expected_supervised)
    assert torch.equal(
        losses["total"],
        losses["supervised_total"]
        + losses["envelope_total"]
        + 0.25 * losses["signal_protection"],
    )

    control = DenseTriadicV17Criterion(
        triplet_margin=0.3,
        label_smoothing=0.1,
        envelope_enabled=False,
    )(output, labels, cameras)
    assert torch.equal(control["supervised_total"], losses["supervised_total"])
    assert torch.equal(control["signal_protection"], losses["signal_protection"])
    assert torch.equal(
        control["total"],
        control["supervised_total"] + 0.25 * control["signal_protection"],
    )


def test_v17_builder_hash_binds_and_freezes_the_v8_endpoint() -> None:
    from modeling.trifusion.signal_preserving_v17_builder import (
        build_signal_preserving_trifusion_v17,
    )

    base_v8 = _TinyFrozenV8()
    result = build_signal_preserving_trifusion_v17(
        base_v8,
        signal_checkpoint_sha256="a" * 64,
        v8_checkpoint_sha256="b" * 64,
        num_classes=2,
        adapter_width=3,
    )

    assert result.checkpoint_sha256 == "b" * 64
    assert result.provenance["architecture"] == "signal_preserving_v17_dtred"
    assert result.provenance["signal_checkpoint_sha256"] == "a" * 64
    assert result.provenance["v8_checkpoint_sha256"] == "b" * 64
    assert result.provenance["base_v8_frozen"] is True
    assert result.provenance["relation_envelope_weight"] == 1.0
    assert result.provenance["signal_protection_weight"] == 0.25
    assert result.provenance["router_enabled"] is False
    assert result.provenance["reranking_enabled"] is False
    assert all(
        not parameter.requires_grad
        for parameter in result.model.base_v8.parameters()
    )
    assert result.provenance["trainable_parameters"] == sum(
        parameter.numel()
        for parameter in result.model.parameters()
        if parameter.requires_grad
    )

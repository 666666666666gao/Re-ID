from __future__ import annotations

from types import MappingProxyType
from types import SimpleNamespace
import os
from pathlib import Path

import torch
import yaml

from modeling.trifusion.state import (
    EXPERT_ORDER,
    ExpertState,
    ExpertStateMap,
    ReliabilityResult,
)


def _state_map(globals_by_expert: dict[str, torch.Tensor]) -> ExpertStateMap:
    first = globals_by_expert[EXPERT_ORDER[0]]
    mask = torch.ones(first.shape[:2], dtype=torch.bool, device=first.device)
    states = {}
    for expert in EXPERT_ORDER:
        values = globals_by_expert[expert]
        states[expert] = ExpertState(
            tokens=values.unsqueeze(2),
            global_embedding=values,
            private_embedding=values[..., :1],
            role_payload=MappingProxyType({"summary": values[..., :1]}),
            modality_mask=mask,
            stage=3,
            expert=expert,
        )
    reliability = ReliabilityResult(
        alpha=torch.full((first.shape[0], 3, 3), 2.0, device=first.device),
        beta=torch.full((first.shape[0], 3, 3), 2.0, device=first.device),
        r=torch.full((first.shape[0], 3, 3), 0.5, device=first.device),
        u=torch.full((first.shape[0], 3, 3), 0.5, device=first.device),
        modality_mask=mask,
    )
    return ExpertStateMap(states, modality_mask=mask, reliability=reliability)


def test_v4_residual_bank_is_non_destructive_and_equal_energy() -> None:
    from modeling.trifusion.task_anchor_v4 import EnergyBalancedResidualBankFusion

    anchor_native = torch.zeros(2, 3, 4)
    anchor_projected = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 5.0], [5.0, 0.0]],
            [[0.0, 2.0], [2.0, 0.0], [0.0, -2.0]],
        ]
    )
    states = _state_map(
        {
            "cnn": torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4).expand(2, 3, 4),
            "transformer": torch.tensor([4.0, 3.0, 2.0, 1.0]).view(1, 1, 4).expand(2, 3, 4),
            "mamba": torch.tensor([1.0, -1.0, 2.0, -2.0]).view(1, 1, 4).expand(2, 3, 4),
        }
    )
    fusion = EnergyBalancedResidualBankFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        embedding_width=2,
    )
    with torch.no_grad():
        fusion.residual_projections["cnn"].weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        )
        fusion.residual_projections["transformer"].weight.copy_(
            torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        )
        fusion.residual_projections["mamba"].weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        )

    result = fusion(
        states,
        states.reliability,
        states.modality_mask,
        anchor_native=anchor_native,
        anchor_projected=anchor_projected,
    )

    anchor = result.fused_embedding[:, :6]
    bank = result.fused_embedding[:, 6:]
    assert result.fused_embedding.shape == (2, 24)
    assert result.contribution_embeddings.shape == (2, 3, 3, 2)
    assert torch.allclose(anchor.norm(dim=1), bank.norm(dim=1), atol=1e-6, rtol=1e-6)
    cnn, transformer, mamba = bank.reshape(2, 3, 6).unbind(dim=1)
    assert not torch.allclose(cnn, transformer)
    assert not torch.allclose(cnn, mamba)
    assert not torch.allclose(transformer, mamba)


def test_v4_zero_residual_mode_is_distance_exact() -> None:
    from modeling.trifusion.task_anchor_v4 import EnergyBalancedResidualBankFusion

    torch.manual_seed(17)
    anchor_native = torch.randn(4, 3, 4)
    anchor_projected = torch.randn(4, 3, 2)
    states = _state_map(
        {
            expert: anchor_native + torch.randn_like(anchor_native)
            for expert in EXPERT_ORDER
        }
    )
    fusion = EnergyBalancedResidualBankFusion(
        expert_widths={expert: 4 for expert in EXPERT_ORDER},
        embedding_width=2,
    )

    result = fusion(
        states,
        states.reliability,
        states.modality_mask,
        anchor_native=anchor_native,
        anchor_projected=anchor_projected,
        force_zero_residual=True,
    )

    anchor = anchor_projected.flatten(1)
    assert torch.equal(result.fused_embedding[:, :6], anchor)
    assert torch.count_nonzero(result.fused_embedding[:, 6:]).item() == 0
    anchor_distances = torch.cdist(
        torch.nn.functional.normalize(anchor, dim=1),
        torch.nn.functional.normalize(anchor, dim=1),
    )
    fused_distances = torch.cdist(
        torch.nn.functional.normalize(result.fused_embedding, dim=1),
        torch.nn.functional.normalize(result.fused_embedding, dim=1),
    )
    assert torch.allclose(fused_distances, anchor_distances, atol=1e-6, rtol=1e-6)


def test_identity_utility_target_prefers_the_expert_with_better_batch_hard_gap() -> None:
    from modeling.trifusion.task_anchor_v4 import identity_utility_router_loss

    labels = torch.tensor([0, 0, 1, 1])
    cnn = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    transformer = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [-0.8, 0.2], [-1.0, 0.0]]
    )
    mamba = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    )
    branches = {
        "cnn": cnn,
        "transformer": transformer,
        "mamba": mamba,
    }
    correct = torch.tensor([0.75, 0.20, 0.05]).view(1, 3, 1).expand(4, 3, 3)
    wrong = torch.tensor([0.05, 0.20, 0.75]).view(1, 3, 1).expand(4, 3, 3)
    mask = torch.ones(4, 3, dtype=torch.bool)

    correct_result = identity_utility_router_loss(
        branches,
        correct,
        labels,
        modality_mask=mask,
    )
    wrong_result = identity_utility_router_loss(
        branches,
        wrong,
        labels,
        modality_mask=mask,
    )

    assert torch.equal(
        correct_result.target_weights.argmax(dim=1),
        torch.zeros(4, dtype=torch.long),
    )
    assert torch.all(correct_result.identity_gaps[:, 0] > correct_result.identity_gaps[:, 2])
    assert correct_result.loss < wrong_result.loss
    assert torch.isfinite(correct_result.loss)


def test_v4_criterion_uses_peer_logits_slot_to_train_the_utility_router() -> None:
    from modeling.trifusion.task_anchor_v4 import TaskAnchoredV4Criterion

    labels = torch.tensor([0, 0, 1, 1])
    branches = {
        "cnn": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
        ),
        "transformer": torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [-0.8, 0.2], [-1.0, 0.0]]
        ),
        "mamba": torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
        ),
    }
    router_logits = torch.zeros(4, 3, 3, requires_grad=True)
    router_weights = router_logits.softmax(dim=1)
    mask = torch.ones(4, 3, dtype=torch.bool)
    output = SimpleNamespace(
        fused_embedding=torch.randn(4, 8, requires_grad=True),
        branch_embeddings=branches,
        fused_logits=torch.randn(4, 2, requires_grad=True),
        branch_logits={
            expert: torch.randn(4, 2, requires_grad=True) for expert in EXPERT_ORDER
        },
        reliability=ReliabilityResult(
            alpha=torch.full((4, 3, 3), 2.0),
            beta=torch.full((4, 3, 3), 2.0),
            r=torch.full((4, 3, 3), 0.5),
            u=torch.full((4, 3, 3), 0.5),
            modality_mask=mask,
        ),
        peer_teaching=None,
        modality_mask=mask,
        anchor_logits=torch.randn(4, 2, requires_grad=True),
        anchor_embedding=torch.randn(4, 6, requires_grad=True),
        anchor_modal=torch.randn(4, 3, 2, requires_grad=True),
        router_weights=router_weights,
    )
    criterion = TaskAnchoredV4Criterion(target_cache=None)

    losses = criterion(output, labels)

    assert losses["peer_logits"] > 0
    losses["peer_logits"].backward()
    assert router_logits.grad is not None
    assert torch.count_nonzero(router_logits.grad).item() > 0


def test_real_clip_v4_builder_declares_the_full_residual_bank_contract() -> None:
    checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
    if not checkpoint_value:
        import pytest

        pytest.skip("TRIFUSION_CLIP_CHECKPOINT is not configured")
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.task_anchor_v4_builder import (
        build_task_anchored_trifusion_v4_from_clip,
    )

    result = build_task_anchored_trifusion_v4_from_clip(
        Path(checkpoint_value),
        num_classes=141,
        architecture="task_anchored_collaborative_v4",
        reliability_mode="joint_beta",
        mamba_mixer_factory=TinySequenceMixer,
    )
    model = result.model

    assert result.provenance["architecture"] == "task_anchored_collaborative_v4"
    assert result.provenance["anchor_path"] == "exact_projected_clip_cls_concat_rgb_ni_ti"
    assert result.provenance["fusion"] == "energy_balanced_utility_routed_tri_expert_residual_bank"
    assert result.provenance["loss_slot_contract"]["peer_logits"] == "identity_utility_router_kl"
    assert model.anchor_embedding_width == 1536
    assert model.fusion.residual_bank_width == 4608
    assert model.fused_embedding_width == 6144
    assert model.branch_embedding_width == 3072
    assert tuple(model.encoder.experts) == EXPERT_ORDER
    assert result.provenance["parameter_budget_pass"]


def test_v4_launcher_and_config_bind_the_utility_router_objective() -> None:
    from tools.run_trifusion_task_anchor_v4 import _partition_trainable_parameters

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/RGBNT201/TriFusion-task-anchor-v4-core-rtx3090.yml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["EXPERIMENT"]["SEED"] == 42
    assert config["DATA"]["TRAIN_BATCH_SIZE"] == 32
    assert config["DATA"]["NUM_INSTANCES"] == 4
    assert config["MODEL"]["ARCHITECTURE"] == "task_anchored_collaborative_v4"
    assert "RESIDUAL_SCALE_INIT" not in config["MODEL"]
    assert config["LOSS"]["PEER_LOGITS"] == 1.0
    checkpoint_value = os.environ.get("TRIFUSION_CLIP_CHECKPOINT")
    if not checkpoint_value:
        return
    from modeling.trifusion.experts.mamba import TinySequenceMixer
    from modeling.trifusion.task_anchor_v4_builder import (
        build_task_anchored_trifusion_v4_from_clip,
    )

    model = build_task_anchored_trifusion_v4_from_clip(
        Path(checkpoint_value),
        num_classes=141,
        architecture="task_anchored_collaborative_v4",
        mamba_mixer_factory=TinySequenceMixer,
    ).model
    pretrained, new = _partition_trainable_parameters(
        model,
        family="collaborative",
        architecture="task_anchored_collaborative_v4",
    )
    pretrained_ids = {id(parameter) for parameter in pretrained}
    new_ids = {id(parameter) for parameter in new}
    assert id(model.tokenizer.output_projection) in pretrained_ids
    assert id(model.fusion.residual_projections["cnn"].weight) in pretrained_ids
    assert id(model.encoder.reliability_gate.shared_output.weight) in new_ids
    assert not pretrained_ids & new_ids

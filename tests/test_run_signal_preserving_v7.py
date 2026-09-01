from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "RGBNT201" / "TriFusion-signal-preserving-v7-rtx3090.yml"


def test_v7_contract_uses_b64k8_stages_and_architecture_bound_sources() -> None:
    from tools.run_signal_preserving_v5 import (
        architecture_source_paths,
        load_contract,
        load_raw_config,
        receipt_schema,
    )

    contract = load_contract(CONFIG)
    config = load_raw_config(CONFIG)
    sources = architecture_source_paths(contract["architecture"])

    assert contract["architecture"] == "signal_preserving_collaborative_v7"
    assert contract["seed"] == 42
    assert contract["train_batch_size"] == 64
    assert contract["num_instances"] == 8
    assert contract["max_epochs"] == 60
    assert config["OPTIMIZATION"]["ROUTER_WARMUP_EPOCHS"] == 10
    assert config["INITIALIZATION"]["V6_CHECKPOINT_SHA256"] == (
        "32bba88cc0204cec6b563ce0a8c6239c828c46eec50647d357b2e9f30031ee2e"
    )
    assert sources["model"].name == "signal_preserving_v7.py"
    assert sources["builder"].name == "signal_preserving_v7_builder.py"
    assert receipt_schema(contract["architecture"], "capacity") == (
        "trifusion-signal-preserving-v7-capacity-v1"
    )


def test_controlled_degradation_changes_only_selected_modality() -> None:
    from tools.run_signal_preserving_v5 import apply_controlled_modality_degradation

    pattern = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]] * 2
    )[None, None]
    images = {
        modality: pattern.expand(2, 3, 4, 4).clone()
        for modality in ("RGB", "NI", "TI")
    }

    degraded, quality = apply_controlled_modality_degradation(
        images,
        selected_modalities=torch.tensor([2, 2]),
        selected_samples=torch.tensor([True, True]),
        degraded_quality=0.25,
    )

    assert torch.equal(degraded["RGB"], images["RGB"])
    assert torch.equal(degraded["NI"], images["NI"])
    assert not torch.equal(degraded["TI"], images["TI"])
    assert torch.equal(quality, torch.tensor([[1.0, 1.0, 0.25]] * 2))


def test_v7_joint_loss_includes_slot_router_alpha_and_quality() -> None:
    from tools.run_signal_preserving_v5 import weighted_training_loss

    losses = {
        "id_fused": torch.tensor(1.0),
        "triplet_fused": torch.tensor(1.0),
        "peer_logits": torch.tensor(1.0),
        "alpha": torch.tensor(1.0),
        "reliability": torch.tensor(1.0),
    }
    for expert in ("cnn", "transformer", "mamba"):
        losses[f"id_{expert}"] = torch.tensor(1.0)
        losses[f"triplet_{expert}"] = torch.tensor(1.0)
        losses[f"id_residual_{expert}"] = torch.tensor(1.0)
        losses[f"triplet_residual_{expert}"] = torch.tensor(1.0)
    config = {
        "MODEL": {"ARCHITECTURE": "signal_preserving_collaborative_v7"},
        "LOSS": {
            "ID_FUSED": 0.25,
            "TRIPLET_FUSED": 1.0,
            "ID_BRANCH": 1.0 / 12.0,
            "TRIPLET_BRANCH": 0.25,
            "ID_RESIDUAL": 1.0 / 12.0,
            "TRIPLET_RESIDUAL": 0.25,
            "PEER_LOGITS": 1.0,
            "ALPHA": 1.0,
            "RELIABILITY": 1.0,
        },
    }

    total = weighted_training_loss(losses, config, phase="joint")
    warmup = weighted_training_loss(losses, config, phase="router_warmup")

    assert torch.equal(total, torch.tensor(6.25))
    assert torch.equal(warmup, torch.tensor(3.0))


class _TinyV7(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.baseline = nn.Linear(2, 2)
        self.encoder = nn.Module()
        self.encoder.expert = nn.Linear(2, 2)
        self.encoder.reliability_gate = nn.Linear(2, 2)
        self.fusion = nn.Module()
        self.fusion.alpha_predictor = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )


def test_v7_initialization_accepts_only_the_four_new_alpha_tensors(tmp_path: Path) -> None:
    from tools.run_signal_preserving_v5 import _sha256, load_v7_initialization

    model = _TinyV7()
    old_state = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("fusion.alpha_predictor.")
    }
    checkpoint = tmp_path / "v6.pth"
    torch.save({"model_state_dict": old_state}, checkpoint)

    result = load_v7_initialization(
        model,
        {
            "V6_CHECKPOINT": str(checkpoint),
            "V6_CHECKPOINT_SHA256": _sha256(checkpoint),
        },
    )

    assert result["missing_keys"] == [
        "fusion.alpha_predictor.0.bias",
        "fusion.alpha_predictor.0.weight",
        "fusion.alpha_predictor.2.bias",
        "fusion.alpha_predictor.2.weight",
    ]
    assert result["unexpected_keys"] == []


def test_v7_router_warmup_updates_only_reliability_and_alpha() -> None:
    from tools.run_signal_preserving_v5 import set_v7_training_phase

    model = _TinyV7()
    model.baseline.requires_grad_(False)
    joint_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    phase_names = set_v7_training_phase(
        model,
        phase="router_warmup",
        joint_trainable_names=joint_trainable,
    )
    assert phase_names == {
        "encoder.reliability_gate.bias",
        "encoder.reliability_gate.weight",
        "fusion.alpha_predictor.0.bias",
        "fusion.alpha_predictor.0.weight",
        "fusion.alpha_predictor.2.bias",
        "fusion.alpha_predictor.2.weight",
    }
    assert not model.encoder.expert.weight.requires_grad

    restored = set_v7_training_phase(
        model,
        phase="joint",
        joint_trainable_names=joint_trainable,
    )
    assert restored == joint_trainable
    assert model.encoder.expert.weight.requires_grad
    assert not model.baseline.weight.requires_grad


def test_v7_overfit_gate_removes_the_label_smoothing_entropy_floor() -> None:
    from tools.run_signal_preserving_v5 import (
        evaluate_overfit_gate,
        load_raw_config,
        overfit_loss_floor,
    )

    config = load_raw_config(CONFIG)
    floor = overfit_loss_floor(config, num_classes=141)
    gate = evaluate_overfit_gate(
        [1.1750260591506958, 0.6539484858512878],
        max_ratio=0.1,
        minimum_loss=floor,
    )

    assert 0.61 < floor < 0.62
    assert gate["loss_ratio"] < 0.1
    assert gate["passed"] is True


def test_v7_quality_response_gate_requires_every_corrupted_modality_to_drop() -> None:
    from tools.run_signal_preserving_v5 import quality_response_gate

    clean = torch.tensor([[0.4, 0.35, 0.25], [0.3, 0.4, 0.3]])
    corrupted = {
        "RGB": torch.tensor([[0.2, 0.45, 0.35], [0.2, 0.45, 0.35]]),
        "NI": torch.tensor([[0.5, 0.2, 0.3], [0.45, 0.25, 0.3]]),
        "TI": torch.tensor([[0.45, 0.4, 0.15], [0.4, 0.4, 0.2]]),
    }

    gate = quality_response_gate(clean, corrupted)

    assert gate["passed"] is True
    assert all(value["decreased"] for value in gate["modalities"].values())

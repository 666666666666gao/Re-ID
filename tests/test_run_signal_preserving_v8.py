from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "RGBNT201"
    / "TriFusion-signal-preserving-v8-expert-formation-rtx3090.yml"
)


def test_v8_contract_binds_pretrained_expert_formation_sources_and_b64k8() -> None:
    from tools.run_signal_preserving_v5 import (
        architecture_source_paths,
        load_contract,
        load_raw_config,
        receipt_schema,
    )

    contract = load_contract(CONFIG)
    config = load_raw_config(CONFIG)
    sources = architecture_source_paths(contract["architecture"])

    assert contract["architecture"] == (
        "signal_preserving_collaborative_v8_expert_formation"
    )
    assert contract["seed"] == 42
    assert contract["train_batch_size"] == 64
    assert contract["num_instances"] == 8
    assert config["MODEL"]["BRANCH_AFTER_BLOCK"] == 8
    assert config["MODEL"]["SEMANTIC_WIDTH"] == 768
    assert config["MODEL"]["EXPERT_MODAL_WIDTH"] == 512
    assert config["MODEL"]["GRADIENT_CHECKPOINTING"] is True
    assert sources["model"].name == "signal_preserving_v8.py"
    assert sources["builder"].name == "signal_preserving_v8_builder.py"
    assert receipt_schema(contract["architecture"], "capacity") == (
        "trifusion-signal-preserving-v8-capacity-v1"
    )


def test_v8_expert_formation_loss_has_no_router_or_hfer_term() -> None:
    from tools.run_signal_preserving_v5 import weighted_training_loss

    losses = {
        "id_fused": torch.tensor(1.0),
        "triplet_fused": torch.tensor(1.0),
    }
    for expert in ("cnn", "transformer", "mamba"):
        losses[f"id_{expert}"] = torch.tensor(1.0)
        losses[f"triplet_{expert}"] = torch.tensor(1.0)
        losses[f"id_residual_{expert}"] = torch.tensor(1.0)
        losses[f"triplet_residual_{expert}"] = torch.tensor(1.0)
    config = {
        "MODEL": {
            "ARCHITECTURE": "signal_preserving_collaborative_v8_expert_formation"
        },
        "LOSS": {
            "ID_FUSED": 0.25,
            "TRIPLET_FUSED": 1.0,
            "ID_BRANCH": 1.0 / 12.0,
            "TRIPLET_BRANCH": 0.25,
            "ID_RESIDUAL": 1.0 / 12.0,
            "TRIPLET_RESIDUAL": 0.25,
        },
    }

    total = weighted_training_loss(losses, config, phase="expert_formation")

    assert torch.equal(total, torch.tensor(3.25))


def test_v8_contract_freezes_identity_disjoint_formation_probe() -> None:
    from tools.run_signal_preserving_v5 import load_raw_config

    config = load_raw_config(CONFIG)

    assert config["OPTIMIZATION"]["FORMATION_PROBE_EPOCHS"] == 20
    assert config["GATES"]["FORMATION_MIN_ORACLE_GAIN_MAP"] == 1.0
    assert config["GATES"]["FORMATION_MIN_UNIQUE_AP_WINS"] == 1
    assert config["GATES"]["FORMATION_MIN_EXPERT_MARGINAL_MAP"] == 0.0
    assert config["PROTOCOL"]["FORMATION_MODEL_SELECTION"] == "none_final_epoch_only"


def test_v8_formation_gate_requires_both_oracles_and_every_expert() -> None:
    from tools.run_signal_preserving_v5 import evaluate_v8_expert_formation_gate

    def summary(gain: float, *, mamba_unique: int = 2) -> dict[str, object]:
        return {
            "oracle_minus_best_fixed_percent": {"mAP": gain},
            "unique_ap_wins": {
                "cnn": 3,
                "transformer": 2,
                "mamba": mamba_unique,
            },
            "leave_one_expert_out": {
                "cnn": {"marginal_mAP": 0.4},
                "transformer": {"marginal_mAP": 0.2},
                "mamba": {"marginal_mAP": 0.1},
            },
        }

    passing = evaluate_v8_expert_formation_gate(
        branch_oracle=summary(1.5),
        residual_oracle=summary(2.0),
        min_oracle_gain_map=1.0,
        min_unique_ap_wins=1,
        min_expert_marginal_map=0.0,
    )
    missing_mamba = evaluate_v8_expert_formation_gate(
        branch_oracle=summary(1.5, mamba_unique=0),
        residual_oracle=summary(2.0),
        min_oracle_gain_map=1.0,
        min_unique_ap_wins=1,
        min_expert_marginal_map=0.0,
    )
    weak_residual = evaluate_v8_expert_formation_gate(
        branch_oracle=summary(1.5),
        residual_oracle=summary(0.9),
        min_oracle_gain_map=1.0,
        min_unique_ap_wins=1,
        min_expert_marginal_map=0.0,
    )

    assert passing["passed"] is True
    assert missing_mamba["passed"] is False
    assert missing_mamba["branch"]["experts"]["mamba"]["unique_ap_wins"] == 0
    assert weak_residual["passed"] is False
    assert weak_residual["residual_only"]["oracle_gain_mAP"] == 0.9

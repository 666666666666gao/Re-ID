from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "RGBNT201" / "TriFusion-signal-preserving-v6-rtx3090.yml"


def test_v6_training_loss_includes_direct_residual_supervision() -> None:
    from tools.run_signal_preserving_v5 import weighted_training_loss

    losses = {
        "id_fused": torch.tensor(2.0),
        "triplet_fused": torch.tensor(3.0),
        "id_cnn": torch.tensor(1.0),
        "id_transformer": torch.tensor(1.0),
        "id_mamba": torch.tensor(1.0),
        "triplet_cnn": torch.tensor(2.0),
        "triplet_transformer": torch.tensor(2.0),
        "triplet_mamba": torch.tensor(2.0),
        "id_residual_cnn": torch.tensor(4.0),
        "id_residual_transformer": torch.tensor(4.0),
        "id_residual_mamba": torch.tensor(4.0),
        "triplet_residual_cnn": torch.tensor(5.0),
        "triplet_residual_transformer": torch.tensor(5.0),
        "triplet_residual_mamba": torch.tensor(5.0),
        "peer_logits": torch.tensor(0.25),
    }
    config = {
        "MODEL": {"ARCHITECTURE": "signal_preserving_collaborative_v6"},
        "LOSS": {
            "ID_FUSED": 0.25,
            "TRIPLET_FUSED": 1.0,
            "ID_BRANCH": 1.0 / 12.0,
            "TRIPLET_BRANCH": 0.25,
            "ID_RESIDUAL": 1.0 / 12.0,
            "TRIPLET_RESIDUAL": 0.25,
            "PEER_LOGITS": 1.0,
        },
    }

    total = weighted_training_loss(losses, config)

    assert torch.equal(total, torch.tensor(10.25))


def test_v6_runner_contract_has_a_distinct_frozen_experiment_identity() -> None:
    from tools.run_signal_preserving_v5 import load_contract, receipt_schema

    contract = load_contract(CONFIG)

    assert contract["architecture"] == "signal_preserving_collaborative_v6"
    assert contract["seed"] == 42
    assert contract["train_batch_size"] == 32
    assert contract["num_instances"] == 4
    assert contract["max_epochs"] == 60
    assert contract["official_test_during_development"] is False
    assert receipt_schema(contract["architecture"], "capacity") == (
        "trifusion-signal-preserving-v6-capacity-v1"
    )

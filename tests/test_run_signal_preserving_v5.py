"""Public runner contracts for the Signal-preserving V5 main experiment."""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "RGBNT201"
    / "TriFusion-signal-preserving-v5-rtx3090.yml"
)


def test_v5_config_freezes_the_same_checkpoint_main_experiment() -> None:
    from tools.run_signal_preserving_v5 import load_contract

    contract = load_contract(CONFIG)

    assert contract["seed"] == 42
    assert contract["train_batch_size"] == 32
    assert contract["num_instances"] == 4
    assert contract["max_epochs"] == 60
    assert contract["signal_checkpoint_sha256"] == (
        "1f5c200cd43fcbc00b8a0494329519eed3e6f062d9a29d43a0ecdd97ff4966c3"
    )
    assert contract["baseline_width"] == 3072
    assert contract["retrieval_outputs"] == (
        "baseline_only",
        "fused",
        "cnn",
        "transformer",
        "mamba",
    )
    assert contract["dev_min_map"] == 65.0
    assert contract["official_test_during_development"] is False


def test_v5_dev_gate_requires_fused_to_clear_every_registered_floor() -> None:
    from tools.run_signal_preserving_v5 import evaluate_dev_gate

    passing = {
        "baseline_only": {"mAP": 58.0},
        "fused": {"mAP": 65.1},
        "cnn": {"mAP": 64.0},
        "transformer": {"mAP": 63.0},
        "mamba": {"mAP": 64.9},
    }
    branch_failure = {**passing, "mamba": {"mAP": 65.2}}

    assert evaluate_dev_gate(passing, min_map=65.0) == {
        "passed": True,
        "fused_mAP": 65.1,
        "minimum_mAP": 65.0,
        "strictly_beaten": {
            "baseline_only": True,
            "cnn": True,
            "transformer": True,
            "mamba": True,
        },
    }
    assert evaluate_dev_gate(branch_failure, min_map=65.0)["passed"] is False


def test_v5_cli_uses_one_versioned_config_for_each_execution_mode(
    tmp_path: Path,
) -> None:
    from tools.run_signal_preserving_v5 import parse_args

    args = parse_args(
        [
            "--mode",
            "preflight",
            "--config",
            str(CONFIG),
            "--output-dir",
            str(tmp_path / "preflight"),
        ]
    )

    assert (args.mode, args.config, args.output_dir) == (
        "preflight",
        CONFIG,
        tmp_path / "preflight",
    )


def test_v5_training_objective_uses_the_frozen_main_loss_weights() -> None:
    from tools.run_signal_preserving_v5 import load_raw_config, weighted_training_loss

    losses = {
        "id_fused": torch.tensor(2.0),
        "triplet_fused": torch.tensor(3.0),
        "id_cnn": torch.tensor(1.0),
        "id_transformer": torch.tensor(2.0),
        "id_mamba": torch.tensor(3.0),
        "triplet_cnn": torch.tensor(4.0),
        "triplet_transformer": torch.tensor(5.0),
        "triplet_mamba": torch.tensor(6.0),
        "peer_logits": torch.tensor(0.25),
    }

    total = weighted_training_loss(losses, load_raw_config(CONFIG))

    assert total.item() == 8.0


def test_v5_overfit_gate_uses_final_to_initial_loss_ratio() -> None:
    from tools.run_signal_preserving_v5 import evaluate_overfit_gate

    assert evaluate_overfit_gate([10.0, 8.0, 0.9], max_ratio=0.1) == {
        "passed": True,
        "initial_loss": 10.0,
        "final_loss": 0.9,
        "minimum_loss": 0.0,
        "initial_excess_loss": 10.0,
        "final_excess_loss": 0.9,
        "loss_ratio": 0.09,
        "maximum_loss_ratio": 0.1,
    }


def test_v5_learning_rate_warms_for_five_epochs_then_cosine_decays() -> None:
    from tools.run_signal_preserving_v5 import learning_rate_multiplier

    assert learning_rate_multiplier(1, max_epochs=60, warmup_epochs=5) == 0.2
    assert learning_rate_multiplier(5, max_epochs=60, warmup_epochs=5) == 1.0
    assert learning_rate_multiplier(6, max_epochs=60, warmup_epochs=5) == 1.0
    assert learning_rate_multiplier(60, max_epochs=60, warmup_epochs=5) < 0.001

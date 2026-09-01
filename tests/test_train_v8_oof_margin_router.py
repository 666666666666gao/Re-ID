from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "RGBNT201"
    / "TriFusion-signal-preserving-v8-router-rtx3090.yml"
)


def test_v8_router_phase_gate_requires_oof_gain_alignment_and_quality() -> None:
    from tools.train_v8_oof_margin_router import evaluate_router_phase_gate

    passing = evaluate_router_phase_gate(
        learned_oof_margin=0.20,
        fixed_oof_margin=0.15,
        learned_top1_accuracy=0.70,
        majority_top1_accuracy=0.60,
        corrupted_mass_decreases={"RGB": True, "NI": True, "TI": True},
        missing_modality_max_mass=0.0,
    )
    collapsed_quality = evaluate_router_phase_gate(
        learned_oof_margin=0.20,
        fixed_oof_margin=0.15,
        learned_top1_accuracy=0.70,
        majority_top1_accuracy=0.60,
        corrupted_mass_decreases={"RGB": True, "NI": False, "TI": True},
        missing_modality_max_mass=0.0,
    )

    assert passing["passed"] is True
    assert collapsed_quality["passed"] is False
    assert collapsed_quality["quality_response_passed"] is False


def test_v8_router_contract_freezes_oof_sources_and_single_seed() -> None:
    from tools.run_signal_preserving_v5 import load_raw_config

    config = load_raw_config(CONFIG)

    assert config["EXPERIMENT"]["SEED"] == 42
    assert config["ROUTER"]["EPOCHS"] == 100
    assert config["ROUTER"]["HIDDEN_WIDTH"] == 128
    assert config["ROUTER"]["ALPHA_MAX"] == 0.5
    assert config["ROUTER"]["ALPHA_INIT"] == 0.2
    assert config["QUALITY"]["DEGRADED_QUALITY"] == 0.2
    assert len(config["INITIALIZATION"]["PHASE_A_CHECKPOINT_SHA256"]) == 64
    assert len(config["INITIALIZATION"]["OOF_MARGIN_CACHE_SHA256"]) == 64
    assert config["PROTOCOL"]["ROUTER_VALIDATION"] == "three_fold_identity_oof"
    assert config["PROTOCOL"]["DEV_ACCESS_DURING_ROUTER_TRAINING"] is False

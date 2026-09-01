from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/RGBNT201/TriFusion-signal-preserving-v9-rtx3090.yml"
RUNNER = ROOT / "tools/run_signal_preserving_v9.py"


def test_v9_main_config_freezes_single_seed_and_final_only_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["EXPERIMENT"]["SEED"] == 42
    assert config["DATA"]["TRAIN_BATCH_SIZE"] == 64
    assert config["DATA"]["NUM_INSTANCES"] == 8
    assert config["MODEL"]["ARCHITECTURE"] == (
        "signal_preserving_collaborative_v9_orthogonal_triadic_synthesis"
    )
    assert config["MODEL"]["BASE_ARCHITECTURE"] == (
        "signal_preserving_collaborative_v8_expert_formation"
    )
    assert config["V9"] == {
        "HIDDEN_WIDTH": 256,
        "SYNERGY_MODAL_WIDTH": 512,
        "RELAY_DEPTH": 2,
        "BETA_MAX": 0.5,
        "BETA_INIT": 0.2,
    }
    assert config["OPTIMIZATION"]["MAX_EPOCHS"] == 60
    assert config["GATES"]["CAPACITY_STEPS"] == 8
    assert config["GATES"]["OVERFIT_STEPS"] == 100
    assert config["GATES"]["DEV_MIN_MAP"] == 65.0
    assert config["PROTOCOL"]["TRAINING_DEV_ACCESS"] is False
    assert config["PROTOCOL"]["MODEL_SELECTION"] == "none_final_epoch_only"
    assert config["PROTOCOL"]["OFFICIAL_TEST_DURING_DEVELOPMENT"] is False


def test_v9_runner_is_train_only_and_loads_project_namespace_after_runtime() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert 'choices=("preflight", "capacity", "overfit", "train")' in source
    assert "runtime_config[\"MODEL\"][\"ARCHITECTURE\"] = ARCHITECTURE_V8" in source
    assert source.index("runtime = _build_runtime(runtime_config)") < source.index(
        "from trifusion.signal_preserving_v9_builder import"
    )
    assert '"dev_access_count": 0' in source
    assert '"official_test_access_count": 0' in source
    assert "eval_loader" not in source
    assert "phase_a_state_unchanged" in source
    assert "router_state_unchanged" in source

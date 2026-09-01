from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/evaluate_signal_preserving_v9_dev.py"


def _metrics(fused: float) -> dict[str, dict[str, float]]:
    return {
        "baseline_only": {"mAP": 58.0},
        "phase_b": {"mAP": 58.4},
        "fused": {"mAP": fused},
        "cnn": {"mAP": 57.6},
        "transformer": {"mAP": 56.3},
        "mamba": {"mAP": 56.6},
    }


def test_v9_dev_gate_requires_65_and_strictly_beats_every_output() -> None:
    from tools.evaluate_signal_preserving_v9_dev import evaluate_v9_dev_gate

    passed = evaluate_v9_dev_gate(_metrics(65.1), min_map=65.0)
    below = evaluate_v9_dev_gate(_metrics(64.9), min_map=65.0)
    tied = _metrics(65.1)
    tied["phase_b"]["mAP"] = 65.1
    tie_gate = evaluate_v9_dev_gate(tied, min_map=65.0)

    assert passed["passed"]
    assert all(passed["strictly_beaten"].values())
    assert not below["passed"]
    assert not tie_gate["passed"]
    assert not tie_gate["strictly_beaten"]["phase_b"]


def test_v9_dev_evaluator_is_single_frozen_read_without_optimizer() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")

    assert '"dev_access_count": 1' in source
    assert '"official_test_access_count": 0' in source
    assert '"optimizer_steps": 0' in source
    assert '"training_executed": False' in source
    assert "torch.optim" not in source
    assert "checkpoint_state_unchanged" in source
    assert '"phase_b"' in source
    assert "evaluate_v9_dev_gate" in source

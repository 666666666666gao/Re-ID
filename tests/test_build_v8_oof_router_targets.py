from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_oof_tool_uses_project_trifusion_namespace_after_signal_import() -> None:
    source = (ROOT / "tools" / "build_v8_oof_router_targets.py").read_text(
        encoding="utf-8"
    )

    assert "from trifusion.aligned_data import build_aligned_train_loader" in source
    assert "from modeling.trifusion.aligned_data" not in source


def test_identity_folds_are_disjoint_and_balance_cross_camera_ids() -> None:
    from tools.build_v8_oof_router_targets import build_identity_folds

    records = []
    for identity in range(6):
        records.append(([f"{identity}_a"], identity, 0, -1))
        if identity < 3:
            records.append(([f"{identity}_b"], identity, 1, -1))

    folds = build_identity_folds(records, num_folds=3)

    assert len(folds) == 3
    assert set().union(*folds) == set(range(6))
    assert all(not (left & right) for index, left in enumerate(folds) for right in folds[index + 1 :])
    assert [len(fold & {0, 1, 2}) for fold in folds] == [1, 1, 1]


def test_oof_target_gate_requires_all_experts_and_modalities() -> None:
    from tools.build_v8_oof_router_targets import evaluate_oof_target_gate

    passing = evaluate_oof_target_gate(
        expert_winner_counts={"cnn": 8, "transformer": 4, "mamba": 5},
        modality_winner_counts={"RGB": 6, "NI": 5, "TI": 6},
        oracle_gain_map=4.0,
        min_oracle_gain_map=1.0,
    )
    collapsed = evaluate_oof_target_gate(
        expert_winner_counts={"cnn": 8, "transformer": 0, "mamba": 5},
        modality_winner_counts={"RGB": 6, "NI": 5, "TI": 6},
        oracle_gain_map=4.0,
        min_oracle_gain_map=1.0,
    )

    assert passing["passed"] is True
    assert collapsed["passed"] is False
    assert collapsed["expert_diversity_passed"] is False

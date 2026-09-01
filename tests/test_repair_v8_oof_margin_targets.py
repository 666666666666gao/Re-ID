from __future__ import annotations

import numpy as np


def test_identity_margin_uses_hardest_positive_and_nearest_negative() -> None:
    from tools.repair_v8_oof_margin_targets import per_query_identity_margin

    distances = np.asarray(
        [
            [0.0, 0.3, 0.1, 0.2],
            [0.2, 0.4, 0.0, 0.1],
        ]
    )
    margins = per_query_identity_margin(
        distances,
        query_ids=np.asarray([0, 1]),
        gallery_ids=np.asarray([0, 0, 1, 1]),
        query_cameras=np.asarray([0, 0]),
        gallery_cameras=np.asarray([0, 1, 0, 1]),
    )

    assert np.allclose(margins, np.asarray([-0.2, 0.1]))


def test_margin_gate_requires_expert_modality_diversity_and_positive_gain() -> None:
    from tools.repair_v8_oof_margin_targets import evaluate_margin_target_gate

    passing = evaluate_margin_target_gate(
        expert_winner_counts={"cnn": 3, "transformer": 4, "mamba": 5},
        modality_winner_counts={"RGB": 4, "NI": 3, "TI": 5},
        oracle_margin_gain=0.2,
    )
    collapsed = evaluate_margin_target_gate(
        expert_winner_counts={"cnn": 3, "transformer": 4, "mamba": 5},
        modality_winner_counts={"RGB": 4, "NI": 0, "TI": 5},
        oracle_margin_gain=0.2,
    )

    assert passing["passed"] is True
    assert collapsed["passed"] is False
    assert collapsed["modality_diversity_passed"] is False

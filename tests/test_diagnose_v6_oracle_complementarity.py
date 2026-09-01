from __future__ import annotations

import numpy as np


def test_per_query_scores_follow_same_camera_exclusion() -> None:
    from tools.diagnose_v6_oracle_complementarity import per_query_reid_scores

    distances = np.asarray(
        [
            [0.0, 0.3, 0.1, 0.2],
            [0.2, 0.4, 0.0, 0.1],
        ],
        dtype=np.float64,
    )
    scores = per_query_reid_scores(
        distances,
        query_ids=np.asarray([0, 1]),
        gallery_ids=np.asarray([0, 0, 1, 1]),
        query_cameras=np.asarray([0, 0]),
        gallery_cameras=np.asarray([0, 1, 0, 1]),
    )

    assert np.allclose(scores.average_precision, np.asarray([1.0 / 3.0, 1.0]))
    assert np.array_equal(scores.rank1_correct, np.asarray([False, True]))


def test_oracle_summary_reports_unique_wins_and_leave_one_out_value() -> None:
    from tools.diagnose_v6_oracle_complementarity import (
        QueryRetrievalScores,
        summarize_oracle_complementarity,
    )

    raw = {
        "baseline_only": ([0.8, 0.2, 0.5, 0.1], [1, 0, 1, 0]),
        "cnn": ([0.7, 0.9, 0.4, 0.1], [1, 1, 0, 0]),
        "transformer": ([0.6, 0.3, 0.8, 0.1], [0, 0, 1, 0]),
        "mamba": ([0.5, 0.4, 0.3, 0.7], [0, 0, 0, 1]),
    }
    scores = {
        name: QueryRetrievalScores(
            average_precision=np.asarray(average_precision, dtype=np.float64),
            rank1_correct=np.asarray(rank1, dtype=bool),
        )
        for name, (average_precision, rank1) in raw.items()
    }

    summary = summarize_oracle_complementarity(scores)

    assert summary["best_fixed_output"] == "cnn"
    assert summary["oracle_metrics_percent"] == {"mAP": 80.0, "Rank-1": 100.0}
    assert summary["oracle_minus_best_fixed_percent"]["mAP"] == 27.5
    assert summary["unique_ap_wins"] == {
        "baseline_only": 1,
        "cnn": 1,
        "transformer": 1,
        "mamba": 1,
    }
    assert summary["unique_rank1_wins"] == {
        "baseline_only": 0,
        "cnn": 1,
        "transformer": 0,
        "mamba": 1,
    }
    assert summary["all_rank1_failures"] == 0
    assert summary["leave_one_expert_out"]["mamba"]["oracle_mAP"] == 65.0
    assert summary["leave_one_expert_out"]["mamba"]["marginal_mAP"] == 15.0

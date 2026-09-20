"""Protocol regression: scene filtering and complete distractor gallery."""

import numpy as np

from tools.train_msvr310_signal_oof import scene_scores


def test_same_scene_positive_removed_and_single_scene_negative_retained():
    # Camera labels could all be identical; scene is the actual removal field.
    # Rows: self, same-ID/same-scene, same-ID/new-scene, competing ID, another ID.
    result = scene_scores(
        np.asarray([[0.0, 0.1, 0.3, 0.2, 0.4]]),
        np.asarray([1]), np.asarray([1, 1, 1, 2, 3]),
        np.asarray([2]), np.asarray([2, 2, 3, 2, 1]),
    )
    assert result["average_precision"] == [0.5]
    assert result["first_match_rank"] == [2]
    assert result["metrics"] == {"mAP": 50.0, "Rank-1": 0.0, "Rank-5": 100.0, "Rank-10": 100.0}


def test_ap_uses_all_cross_scene_positives():
    # After same-scene removal: positive, negative, positive -> AP=(1+2/3)/2.
    result = scene_scores(
        np.asarray([[0.0, 0.1, 0.2, 0.3]]),
        np.asarray([1]), np.asarray([1, 1, 2, 1]),
        np.asarray([2]), np.asarray([2, 3, 2, 4]),
    )
    assert result["first_match_rank"] == [1]
    assert np.isclose(result["average_precision"][0], 5 / 6, atol=0, rtol=1e-15)

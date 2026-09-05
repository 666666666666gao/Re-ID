import numpy as np

from tools.audit_v17_full_gallery import full_gallery_scores


def test_single_camera_identity_remains_a_gallery_distractor():
    # Identity 9 has no positive camera pair but outranks the true match for ID 1.
    distances = np.array([[0., 2., 1.], [2., 0., 3.], [1., 3., 0.]])
    result = full_gallery_scores(distances, [1, 1, 9], [0, 1, 0])
    assert result["query_indices"] == [0, 1]
    assert result["excluded_no_cross_camera_positive"] == [2]
    assert result["average_precision"] == [0.5, 1.0]
    assert result["first_match_rank"] == [2, 1]
    assert result["metrics_percent"] == {
        "mAP": 75.0, "Rank-1": 50.0, "Rank-5": 100.0, "Rank-10": 100.0,
    }

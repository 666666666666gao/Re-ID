from __future__ import annotations

import unittest

import numpy as np


class OfficialReIDEvaluationTests(unittest.TestCase):
    def test_hand_computed_cmc_map_with_same_camera_junk(self) -> None:
        from utils.reid_evaluation import evaluate_reid

        # Query 0's closest gallery item is the same identity and camera, so it
        # is junk. Its valid positive is therefore rank 2 (AP=1/2). Query 1's
        # positive is rank 1 (AP=1), giving mean AP=3/4.
        distances = np.asarray(
            [
                [0.0, 0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.0, 0.1],
            ],
            dtype=np.float64,
        )
        query_ids = np.asarray([1, 2])
        query_cameras = np.asarray([1, 2])
        gallery_ids = np.asarray([1, 3, 1, 2, 4])
        gallery_cameras = np.asarray([1, 2, 2, 1, 3])

        cmc, mean_ap = evaluate_reid(
            distances,
            query_ids,
            gallery_ids,
            query_cameras,
            gallery_cameras,
            max_rank=4,
        )

        np.testing.assert_array_equal(cmc, np.asarray([0.5, 1.0, 1.0, 1.0]))
        self.assertAlmostEqual(mean_ap, 0.75)

    def test_query_without_valid_positive_is_excluded_from_average(self) -> None:
        from utils.reid_evaluation import evaluate_reid

        distances = np.asarray([[0.1, 0.2], [0.1, 0.2]])
        query_ids = np.asarray([99, 1])
        query_cameras = np.asarray([1, 1])
        gallery_ids = np.asarray([1, 2])
        gallery_cameras = np.asarray([2, 2])

        cmc, mean_ap = evaluate_reid(
            distances,
            query_ids,
            gallery_ids,
            query_cameras,
            gallery_cameras,
            max_rank=2,
        )

        np.testing.assert_array_equal(cmc, np.asarray([1.0, 1.0]))
        self.assertEqual(mean_ap, 1.0)

    def test_all_queries_without_valid_positives_fail_closed(self) -> None:
        from utils.reid_evaluation import evaluate_reid

        with self.assertRaisesRegex(RuntimeError, "no valid query"):
            evaluate_reid(
                np.asarray([[0.1, 0.2]]),
                np.asarray([99]),
                np.asarray([1, 2]),
                np.asarray([1]),
                np.asarray([2, 2]),
                max_rank=2,
            )

    def test_legacy_evaluator_handles_unequal_post_filter_rank_lengths(self) -> None:
        from utils.metrics import eval_func

        distances = np.asarray(
            [
                [0.0, 0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.0, 0.1],
            ]
        )
        cmc, mean_ap = eval_func(
            distances,
            np.asarray([1, 2]),
            np.asarray([1, 3, 1, 2, 4]),
            np.asarray([1, 2]),
            np.asarray([1, 2, 2, 1, 3]),
            max_rank=5,
        )

        np.testing.assert_array_equal(
            cmc, np.asarray([0.5, 1.0, 1.0, 1.0, 1.0])
        )
        self.assertAlmostEqual(mean_ap, 0.75)


if __name__ == "__main__":
    unittest.main()

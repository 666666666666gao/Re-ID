from __future__ import annotations

import math

import torch


def test_identity_disjoint_utility_teacher_recovers_a_worked_mapping() -> None:
    from tools.probe_v8_frozen_router import (
        fit_least_squares_utility_teacher,
        predict_utility_probabilities,
    )

    features = torch.tensor(
        [
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [-2.0, -2.0],
            [-3.0, -3.0],
        ]
    )
    utilities = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    teacher = fit_least_squares_utility_teacher(features, utilities)
    probabilities = predict_utility_probabilities(teacher, features)

    assert torch.equal(
        probabilities.argmax(dim=1), utilities.argmax(dim=1)
    )
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(6))


def test_equal_energy_fusion_preserves_signal_prefix_and_activates_residual() -> None:
    from tools.probe_v8_frozen_router import compose_equal_energy_fused

    baseline = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    contributions = torch.tensor(
        [
            [[[1.0]], [[2.0]], [[3.0]]],
            [[[2.0]], [[1.0]], [[4.0]]],
        ]
    )
    expert_probabilities = torch.tensor(
        [[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]
    )
    modal_probabilities = torch.ones(2, 1)

    fused = compose_equal_energy_fused(
        baseline,
        contributions,
        expert_probabilities,
        modal_probabilities,
    )

    assert torch.equal(fused[:, :2], baseline)
    assert torch.allclose(
        fused[:, 2:].norm(dim=1), baseline.norm(dim=1), rtol=1e-6
    )


def test_winner_alignment_compares_against_the_fit_majority_teacher() -> None:
    from tools.probe_v8_frozen_router import summarize_winner_alignment

    utilities = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    probabilities = torch.tensor(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.7, 0.2]]
    )

    summary = summarize_winner_alignment(
        utilities,
        probabilities,
        majority_expert_index=1,
    )

    assert summary["accuracy"] == 1.0
    assert math.isclose(summary["majority_accuracy"], 2.0 / 3.0, rel_tol=1e-6)
    assert summary["beats_majority"] is True
    assert summary["target_distribution"] == [1.0 / 3.0, 2.0 / 3.0, 0.0]
    assert summary["predicted_distribution"] == [1.0 / 3.0, 2.0 / 3.0, 0.0]


def test_retrieval_teacher_keeps_only_cross_camera_fit_identities() -> None:
    from tools.probe_v8_frozen_router import select_cross_camera_records

    records = [
        (("a", "b", "c"), 1, 0, 0),
        (("d", "e", "f"), 1, 1, 0),
        (("g", "h", "i"), 2, 0, 0),
        (("j", "k", "l"), 2, 0, 0),
    ]

    assert select_cross_camera_records(records) == records[:2]

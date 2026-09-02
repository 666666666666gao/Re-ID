from __future__ import annotations

import torch


def test_v13_q0_gate_requires_actual_path_diversity_and_action_transfer() -> None:
    from tools.build_v13_deployment_aligned_targets import evaluate_v13_q0_gate

    passing = evaluate_v13_q0_gate(
        expert_unique_positive_wins={"cnn": 1, "transformer": 2, "mamba": 3},
        modality_unique_positive_wins={"RGB": 3, "NI": 2, "TI": 1},
        oracle_mean_utility=0.3,
        best_fixed_mean_utility=0.1,
        transfer_per_fold_noninferior=(True, True, True),
        transfer_aggregate_gain=0.01,
        reference_bank_immutable=True,
        dev_access_count=0,
        official_test_access_count=0,
    )
    failed_transfer = evaluate_v13_q0_gate(
        expert_unique_positive_wins={"cnn": 1, "transformer": 2, "mamba": 3},
        modality_unique_positive_wins={"RGB": 3, "NI": 2, "TI": 1},
        oracle_mean_utility=0.3,
        best_fixed_mean_utility=0.1,
        transfer_per_fold_noninferior=(True, False, True),
        transfer_aggregate_gain=0.01,
        reference_bank_immutable=True,
        dev_access_count=0,
        official_test_access_count=0,
    )

    assert passing["passed"] is True
    assert failed_transfer["passed"] is False
    assert failed_transfer["action_transfer_passed"] is False


def test_v13_paired_cache_keeps_teacher_targets_and_deployment_inputs_aligned() -> None:
    from tools.build_v13_deployment_aligned_targets import build_paired_cache_payload

    payload = build_paired_cache_payload(
        sample_keys=("sample-a", "sample-b"),
        identities=torch.tensor([10, 20]),
        cameras=torch.tensor([0, 1]),
        folds=torch.tensor([0, 1]),
        teacher_baseline=torch.tensor([[1.0], [2.0]]),
        teacher_modal_residual=torch.zeros(2, 3, 3, 1),
        teacher_utility=torch.ones(2, 3, 3),
        student_direct_modal=torch.full((2, 3, 1), 4.0),
        student_modal_residual=torch.full((2, 3, 3, 1), 5.0),
        phase_a_checkpoint_sha256="a" * 64,
    )

    assert payload["schema_version"] == "trifusion-v13-paired-target-cache-v1"
    assert payload["sample_keys"] == ("sample-a", "sample-b")
    assert torch.equal(payload["teacher_identity_utility"], torch.ones(2, 3, 3))
    assert torch.equal(payload["student_direct_modal"], torch.full((2, 3, 1), 4.0))
    assert payload["phase_a_checkpoint_sha256"] == "a" * 64

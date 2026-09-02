from __future__ import annotations


def test_v17_m0_gate_requires_pairs_freezing_capacity_and_overfit() -> None:
    from tools.train_signal_preserving_v17 import evaluate_v17_m0_gate

    common = {
        "exact_prefix": True,
        "paired_receipts_equal": True,
        "all_batches_have_positive_and_negative_pairs": True,
        "source_only_teacher": True,
        "capacity_overflow_events": 0,
        "capacity_all_trainable_tensors_reached": True,
        "capacity_frozen_state_unchanged": True,
        "capacity_peak_reserved_mib": 6000.0,
        "overfit_overflow_events": 0,
        "overfit_frozen_state_unchanged": True,
        "overfit_excess_loss_ratio": 0.09,
        "overfit_max_loss_ratio": 0.1,
    }

    passing = evaluate_v17_m0_gate(**common)
    missing_pairs = evaluate_v17_m0_gate(
        **{
            **common,
            "all_batches_have_positive_and_negative_pairs": False,
        }
    )

    assert passing["passed"] is True
    assert missing_pairs["passed"] is False
    assert missing_pairs["preflight_passed"] is False


def test_v17_q1_gate_requires_matched_gain_branches_and_both_envelopes() -> None:
    from tools.train_signal_preserving_v17 import evaluate_v17_q1_gate

    common = {
        "fold_map_gains": (
            {"fused": 0.0, "cnn": 0.1, "transformer": 0.1, "mamba": 0.1},
            {"fused": 0.4, "cnn": 0.1, "transformer": 0.1, "mamba": 0.1},
            {"fused": 2.6, "cnn": 0.1, "transformer": 0.1, "mamba": 0.1},
        ),
        "weighted_fused_gain_map": 1.0,
        "fused_gain_bootstrap_lower_bound": 0.01,
        "aggregate_branch_gains": {
            "cnn": 0.1,
            "transformer": 0.1,
            "mamba": 0.1,
        },
        "fused_violations": {
            "weight0": {"positive": 0.3, "negative": 0.2},
            "dtred": {"positive": 0.2, "negative": 0.1},
        },
        "integrity": {
            "fold_isolation": True,
            "paired_initial_state": True,
            "paired_trainable_names": True,
            "paired_sample_order": True,
            "paired_transformed_batches": True,
            "paired_seed_contract": True,
            "paired_optimizer_steps": True,
            "frozen_state_unchanged": True,
            "zero_overflow": True,
            "exact_signal_prefix": True,
            "source_only_teacher": True,
            "final_epoch_only": True,
            "access_boundary": True,
        },
    }

    passing = evaluate_v17_q1_gate(**common)
    bad_negative_envelope = evaluate_v17_q1_gate(
        **{
            **common,
            "fused_violations": {
                "weight0": {"positive": 0.3, "negative": 0.2},
                "dtred": {"positive": 0.2, "negative": 0.2},
            },
        }
    )

    assert passing["passed"] is True
    assert bad_negative_envelope["passed"] is False
    assert bad_negative_envelope["fused_envelope_improvement_passed"] is False


def test_v17_d1_gate_requires_65_map_strict_wins_and_no_reranking() -> None:
    from tools.train_signal_preserving_v17 import evaluate_v17_d1_gate

    metrics = {
        "baseline_only": {"mAP": 58.0, "Rank-1": 60.0},
        "fused": {"mAP": 65.1, "Rank-1": 68.0},
        "cnn": {"mAP": 64.0, "Rank-1": 66.0},
        "transformer": {"mAP": 63.0, "Rank-1": 65.0},
        "mamba": {"mAP": 62.0, "Rank-1": 64.0},
    }
    common = {
        "metrics_percent": metrics,
        "minimum_fused_map": 65.0,
        "v8_phase_b_map": 58.405,
        "strict_reload": True,
        "frozen_state_unchanged": True,
        "exact_signal_prefix": True,
        "official_test_access_count": 0,
    }

    passing = evaluate_v17_d1_gate(**common, reranking_enabled=False)
    reranked = evaluate_v17_d1_gate(**common, reranking_enabled=True)

    assert passing["passed"] is True
    assert reranked["passed"] is False
    assert reranked["protocol_passed"] is False


def test_v17_prior_gate_receipt_is_bound_to_code_and_access_boundary() -> None:
    from tools.train_signal_preserving_v17 import validate_v17_prior_gate

    expected_sources = {
        "core": "c" * 64,
        "builder": "b" * 64,
        "runner": "r" * 64,
        "config": "f" * 64,
    }
    receipt = {
        "schema_version": "trifusion-v17-q1-result-v1",
        "status": "PASS",
        "passed": True,
        "scientific_gate": {"passed": True},
        "next_phase_authorized": True,
        "repository_commit": "commit-17",
        "repository_diff_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        ),
        "config_sha256": "f" * 64,
        "source_file_sha256": expected_sources,
        "dev_access_count": 0,
        "official_test_access_count": 0,
    }

    assert validate_v17_prior_gate(
        receipt,
        expected_stage="q1",
        expected_repository_commit="commit-17",
        expected_config_sha256="f" * 64,
        expected_source_file_sha256=expected_sources,
    )
    assert not validate_v17_prior_gate(
        {**receipt, "repository_commit": "forged"},
        expected_stage="q1",
        expected_repository_commit="commit-17",
        expected_config_sha256="f" * 64,
        expected_source_file_sha256=expected_sources,
    )
    assert not validate_v17_prior_gate(
        {**receipt, "official_test_access_count": 1},
        expected_stage="q1",
        expected_repository_commit="commit-17",
        expected_config_sha256="f" * 64,
        expected_source_file_sha256=expected_sources,
    )

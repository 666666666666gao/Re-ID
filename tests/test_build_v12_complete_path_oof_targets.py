def test_complete_path_fold_records_exclude_heldout_identities_and_relabel_fit() -> None:
    from tools.build_v12_complete_path_oof_targets import (
        build_complete_path_fold_records,
    )

    records = [
        (["a"], 10, 0, -1),
        (["b"], 20, 1, -1),
        (["c"], 30, 2, -1),
        (["d"], 10, 1, -1),
    ]

    split = build_complete_path_fold_records(records, heldout_ids={20})

    assert split["fit_identity_ids"] == (10, 30)
    assert split["heldout_identity_ids"] == (20,)
    assert split["identity_overlap"] == ()
    assert [record[1] for record in split["train_records"]] == [0, 1, 0]
    assert [record[1] for record in split["heldout_records"]] == [20]
    assert split["label_map"] == {10: 0, 30: 1}


def test_complete_path_gate_rejects_a_signal_teacher_that_saw_heldout_identity() -> None:
    from tools.build_v12_complete_path_oof_targets import (
        evaluate_complete_path_oof_gate,
    )

    receipts = [
        {
            "signal_fit_identity_ids": [0, 1],
            "expert_fit_identity_ids": [0, 1],
            "heldout_identity_ids": [1, 2],
            "signal_epochs": 50,
            "expert_epochs": 20,
            "signal_checkpoint_selection": "final_epoch_only",
            "overflow_events": 0,
            "dev_access_count": 0,
            "official_test_access_count": 0,
        }
        for _fold in range(3)
    ]

    gate = evaluate_complete_path_oof_gate(
        fold_receipts=receipts,
        query_count=571,
        fixed_map={"cnn": 70.0, "transformer": 72.0, "mamba": 71.0, "bank": 73.0},
        expert_winner_counts={"cnn": 1, "transformer": 1, "mamba": 1},
        modality_winner_counts={"RGB": 1, "NI": 1, "TI": 1},
        residual_oracle_gain_map=2.0,
        slot_oracle_margin_gain=0.1,
    )

    assert gate["passed"] is False
    assert gate["complete_path_identity_isolation_passed"] is False

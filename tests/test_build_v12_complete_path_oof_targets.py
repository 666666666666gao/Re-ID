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

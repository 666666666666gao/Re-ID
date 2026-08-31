from __future__ import annotations

from types import SimpleNamespace

import torch


OPERATORS = {
    "clean": {"operator": "identity", "severity": 0},
    "gaussian_blur": {
        "operator": "gaussian-blur",
        "severity": 1,
        "kernel_size": 5,
        "sigma": 1.25,
    },
    "occlusion": {
        "operator": "shared-square-occlusion",
        "severity": 1,
        "side_fraction": 0.5,
        "normalized_fill": 0.0,
    },
    "exposure": {
        "operator": "raw-intensity-scale",
        "severity": 1,
        "modality": "RGB",
        "raw_scale": 0.55,
    },
    "nir_noise": {
        "operator": "normalized-gaussian-noise",
        "severity": 1,
        "modality": "NI",
        "normalized_sigma": 0.12,
    },
    "thermal_noise": {
        "operator": "normalized-gaussian-noise",
        "severity": 1,
        "modality": "TI",
        "normalized_sigma": 0.12,
    },
    "modality_missing": {
        "operator": "hard-mask-and-zero",
        "severity": 1,
        "selection": "sample-hash-mod-3",
    },
}


def test_registered_conditions_are_deterministic_and_missingness_is_explicit() -> None:
    from modeling.trifusion.circ_scoring import apply_registered_condition

    images = {
        modality: torch.linspace(-1.0, 1.0, 2 * 3 * 16 * 8).reshape(2, 3, 16, 8)
        for modality in ("RGB", "NI", "TI")
    }
    keys = ("000001_c1.jpg", "000002_c2.jpg")
    for family, seed in (("nir_noise", 42401), ("modality_missing", 42601)):
        condition = {"family": family, "severity": 1, "seed": seed}
        first_images, first_mask = apply_registered_condition(
            images,
            keys,
            condition,
            operators=OPERATORS,
        )
        repeated_images, repeated_mask = apply_registered_condition(
            images,
            keys,
            condition,
            operators=OPERATORS,
        )
        assert torch.equal(first_mask, repeated_mask)
        for modality in images:
            assert torch.equal(first_images[modality], repeated_images[modality])
    missing_images, missing_mask = apply_registered_condition(
        images,
        keys,
        {"family": "modality_missing", "severity": 1, "seed": 42601},
        operators=OPERATORS,
    )
    assert torch.equal(missing_mask.sum(dim=1), torch.tensor([2, 2]))
    for row in range(2):
        missing_index = int((~missing_mask[row]).nonzero().item())
        assert torch.count_nonzero(
            missing_images[("RGB", "NI", "TI")[missing_index]][row]
        ).item() == 0


def test_registered_training_schedule_covers_each_condition_once_per_cycle() -> None:
    from modeling.trifusion.circ_scoring import select_training_conditions

    protocol = {
        "conditions": [
            {"family": family, "severity": 0 if family == "clean" else 1, "seeds": [42000 + index]}
            for index, family in enumerate(OPERATORS)
        ]
    }
    expected = tuple(
        (item["family"], item["severity"], item["seeds"][0])
        for item in protocol["conditions"]
    )
    observed = tuple(
        (
            selected[0]["family"],
            selected[0]["severity"],
            selected[0]["seed"],
        )
        for epoch in range(1, len(expected) + 1)
        for selected in (
            select_training_conditions(
                ("000001_c1.jpg",), epoch=epoch, protocol=protocol
            ),
        )
    )

    assert set(observed) == set(expected)
    assert len(observed) == len(set(observed))
    assert select_training_conditions(
        ("000001_c1.jpg",), epoch=1, protocol=protocol
    ) == select_training_conditions(
        ("000001_c1.jpg",), epoch=1, protocol=protocol
    )


def test_mixed_condition_batch_applies_the_registered_operator_per_row() -> None:
    from modeling.trifusion.circ_scoring import apply_registered_condition_batch

    images = {
        modality: torch.ones(2, 3, 8, 4) for modality in ("RGB", "NI", "TI")
    }
    conditioned, mask = apply_registered_condition_batch(
        images,
        ("clean-row", "missing-row"),
        (
            {"family": "clean", "severity": 0, "seed": 42000},
            {"family": "modality_missing", "severity": 1, "seed": 42601},
        ),
        operators=OPERATORS,
    )

    assert mask[0].tolist() == [True, True, True]
    assert int(mask[1].sum()) == 2
    assert all(torch.equal(conditioned[name][0], images[name][0]) for name in images)
    missing_index = int((~mask[1]).nonzero().item())
    assert torch.count_nonzero(
        conditioned[("RGB", "NI", "TI")[missing_index]][1]
    ).item() == 0


def test_symmetry_audit_samples_identity_clusters_before_reusing_an_identity() -> None:
    from modeling.trifusion.circ_scoring import audit_query_gallery_symmetry

    rows = []
    for identity, sample_key in ((1, "a"), (2, "b"), (3, "c")):
        for contribution_index in range(4):
            value = 0.1 * (identity + contribution_index)
            rows.append(
                {
                    "sample_key": sample_key,
                    "identity": identity,
                    "condition": {"family": "clean", "severity": 0, "seed": 42000},
                    "contribution": f"cnn.m{contribution_index}",
                    "query_only_delta": value,
                    "symmetric_delta": value,
                }
            )

    receipt = audit_query_gallery_symmetry(
        rows,
        protocol_hash="ab" * 32,
        epsilon=0.02,
        audit_specification={
            "sample_rows": 3,
            "identity_clustered": True,
            "minimum_sign_agreement": 0.7,
            "minimum_spearman": 0.5,
        },
    )

    assert receipt["status"] == "PASS"
    assert receipt["selected_identity_count"] == 3
    assert receipt["selection_rule"] == (
        "identity-cluster round-robin by canonical SHA256 without replacement"
    )


def test_proxy_transfer_selection_and_audit_are_identity_clustered() -> None:
    from modeling.trifusion.circ_scoring import (
        audit_proxy_target_transfer,
        select_proxy_transfer_rows,
    )

    cache_rows = []
    for identity, sample_key in ((1, "a"), (2, "b"), (3, "c")):
        cache_rows.append(
            {
                "sample_key": sample_key,
                "identity": identity,
                "camera": identity % 2,
                "condition": {"family": "clean", "severity": 0, "seed": 42000},
                "contributions": {
                    f"{expert}.{modality}": {
                        "valid": True,
                        "effects": {"total": 0.1 * identity},
                    }
                    for expert in ("cnn", "transformer", "mamba")
                    for modality in ("RGB", "NI", "TI")
                },
            }
        )

    selected = select_proxy_transfer_rows(
        cache_rows,
        sample_count=3,
        protocol_hash="ab" * 32,
    )
    receipt = audit_proxy_target_transfer(
        selected,
        [row["proxy_delta"] for row in selected],
        [0.5 + row["proxy_delta"] for row in selected],
        epsilon=0.02,
        minimum_sign_agreement=0.7,
        minimum_spearman=0.5,
    )

    assert len({row["identity"] for row in selected}) == 3
    assert receipt["status"] == "PASS"
    assert receipt["claim_eligible"] is True
    assert receipt["selected_identity_count"] == 3
    assert receipt["router_helpfulness_agreement"] == 1.0
    assert receipt["router_deployed_spearman"] > 0.99


def test_reference_margin_uses_cross_camera_positive_and_other_identity_negative() -> None:
    from modeling.trifusion.circ_scoring import build_reference_margin_bank

    reference = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )
    bank = build_reference_margin_bank(
        reference,
        reference_identities=(1, 1, 2, 2),
        reference_cameras=(0, 1, 0, 1),
        query_identities=(1, 2),
        query_cameras=(0, 1),
    )
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    margins = bank.margins(query)
    assert margins.shape == (2,)
    assert torch.all(margins > 0.5)
    assert bank.cross_camera_positive_cameras == ((1,), (0,))


class _ScoringModel(torch.nn.Module):
    def forward(self, batch, return_aux=False):
        means = torch.stack(
            [batch["images"][name].mean(dim=(1, 2, 3)) for name in ("RGB", "NI", "TI")],
            dim=1,
        )
        mask = batch["modality_mask"].to(means.dtype)
        embedding = torch.cat((means * mask, torch.ones_like(means[:, :1])), dim=1)
        intervention = batch.get("intervention")
        if intervention is not None:
            modality_index = ("RGB", "NI", "TI").index(intervention.modality)
            scale = {"direct": 0.03, "relay": 0.05, "total": 0.09, "edge": 0.02}[
                intervention.kind
            ]
            embedding[:, modality_index] = (
                embedding[:, modality_index] - scale * mask[:, modality_index]
            )
        return SimpleNamespace(fused_embedding=embedding)


def test_full_condition_scorer_emits_nine_contributions_and_two_edges_per_row() -> None:
    from modeling.trifusion.circ_scoring import (
        build_reference_margin_bank,
        score_registered_condition,
    )

    reference = torch.tensor(
        [
            [1.0, 0.1, 0.1, 1.0],
            [0.9, 0.2, 0.1, 1.0],
            [0.1, 1.0, 0.2, 1.0],
            [0.2, 0.9, 0.1, 1.0],
        ]
    )
    bank = build_reference_margin_bank(
        reference,
        reference_identities=(1, 1, 2, 2),
        reference_cameras=(0, 1, 0, 1),
        query_identities=(1, 2),
        query_cameras=(0, 1),
    )
    images = {
        "RGB": torch.stack((torch.ones(3, 8, 4), torch.zeros(3, 8, 4))),
        "NI": torch.stack((torch.zeros(3, 8, 4), torch.ones(3, 8, 4))),
        "TI": torch.full((2, 3, 8, 4), 0.1),
    }
    model = _ScoringModel().eval()
    rows, symmetry = score_registered_condition(
        model,
        images,
        ("sample-1", "sample-2"),
        (1, 2),
        (0, 1),
        bank,
        {"family": "clean", "severity": 0, "seed": 42000},
        operators=OPERATORS,
        protocol_hash="ab" * 32,
        generator_training_identities=(3, 4),
        generator_checkpoint_sha256="cd" * 32,
        reference_bank_sha256="ef" * 32,
        batch_size=2,
        device=torch.device("cpu"),
        amp=False,
    )
    assert len(rows) == 2
    assert symmetry == []
    for row in rows:
        assert len(row["interventions"]) == 9
        assert all(
            set(effect) == {"total", "direct", "relay"}
            for effect in row["interventions"].values()
        )
        assert len(row["edge_effects"]["1"]) == 1
        assert len(row["edge_effects"]["2"]) == 1
        assert row["generator_training_identities"] == [3, 4]

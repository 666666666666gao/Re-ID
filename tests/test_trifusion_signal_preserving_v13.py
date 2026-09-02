from __future__ import annotations

import torch


def test_v13_shared_fusion_preserves_prefix_and_uses_fixed_energy() -> None:
    from modeling.trifusion.signal_preserving_v13 import compose_v13_fusion

    baseline = torch.tensor([[3.0, 4.0]])
    residual = torch.ones(1, 3, 3, 1)
    weights = torch.full((1, 3, 3), 1.0 / 9.0)

    output = compose_v13_fusion(baseline, residual, weights)

    assert torch.equal(output.fused_embedding[:, :2], baseline)
    assert torch.allclose(
        output.fused_embedding[:, 2:].norm(dim=1),
        torch.tensor([1.0]),
    )
    assert torch.allclose(output.retrieval_embedding.norm(dim=1), torch.ones(1))


def test_v13_query_side_counterfactual_detects_the_only_helpful_slot() -> None:
    from modeling.trifusion.signal_preserving_v13 import (
        query_side_counterfactual_utilities,
    )

    identities = torch.tensor([0, 0, 1, 1])
    cameras = torch.tensor([0, 1, 0, 1])
    baseline = torch.ones(4, 1)
    residual = torch.zeros(4, 3, 3, 1)
    residual[:2, 0, 0, 0] = 1.0
    residual[2:, 0, 0, 0] = -1.0

    result = query_side_counterfactual_utilities(
        baseline,
        residual,
        identities,
        cameras,
    )

    assert result.utilities.shape == (4, 3, 3)
    assert torch.all(result.utilities[:, 0, 0] > 0.0)
    assert torch.equal(result.utilities[:, 0, 1:], torch.zeros(4, 2))
    assert torch.equal(result.utilities[:, 1:], torch.zeros(4, 2, 3))
    assert (
        result.reference_embedding_sha256_before
        == result.reference_embedding_sha256_after
    )


def test_v13_identity_cluster_bootstrap_preserves_whole_identities() -> None:
    from modeling.trifusion.signal_preserving_v13 import (
        identity_cluster_bootstrap_lower_bound,
    )

    differences = torch.tensor([1.0, 1.0, 2.0, 2.0])
    identities = torch.tensor([10, 10, 20, 20])

    result = identity_cluster_bootstrap_lower_bound(
        differences,
        identities,
        seed=42,
        resamples=100,
    )

    assert result.observed_mean == 1.5
    assert result.lower_bound > 0.0
    assert result.cluster_count == 2
    assert result.resamples == 100

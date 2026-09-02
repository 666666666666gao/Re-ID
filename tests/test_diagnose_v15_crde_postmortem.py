import math

import torch


def test_v15_exchange_effect_statistics_report_actual_injection_geometry() -> None:
    from tools.diagnose_v15_crde_postmortem import (
        exchange_effect_statistics_v15,
    )

    own_delta = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    incoming = torch.tensor([[1.0, 0.0], [0.0, -2.0]])

    result = exchange_effect_statistics_v15(own_delta, incoming)

    assert result["own_norm_mean"] == 3.0
    assert result["incoming_norm_mean"] == 1.5
    assert result["incoming_to_own_norm_mean"] == 0.5
    assert result["incoming_own_cosine_mean"] == 0.0


def test_v15_matched_embedding_statistics_link_change_to_query_ap() -> None:
    from tools.diagnose_v15_crde_postmortem import (
        matched_embedding_statistics_v15,
    )

    exchange_off = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    exchange_on = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ap_gain = torch.tensor([0.2, -0.1])

    result = matched_embedding_statistics_v15(
        exchange_on,
        exchange_off,
        ap_gain,
    )

    assert result["cosine_mean"] == 0.5
    assert math.isclose(
        result["l2_displacement_mean"],
        math.sqrt(2.0) / 2.0,
        rel_tol=1e-6,
    )
    assert math.isclose(result["ap_gain_mean"], 0.05, abs_tol=1e-6)
    assert math.isclose(result["ap_gain_median"], 0.05, abs_tol=1e-6)
    assert result["positive_queries"] == 1
    assert result["negative_queries"] == 1
    assert result["zero_queries"] == 0
    assert math.isclose(result["displacement_ap_gain_pearson"], -1.0, abs_tol=1e-6)


def test_v15_edge_scale_stability_exposes_cross_fold_sign_flips() -> None:
    from tools.diagnose_v15_crde_postmortem import edge_scale_stability_v15

    scales = torch.zeros(3, 1, 3, 3)
    scales[:, 0, 0, 1] = torch.tensor([0.1, -0.2, 0.3])

    result = edge_scale_stability_v15(scales)
    edge = next(item for item in result if item["edge"] == "cnn__transformer")

    assert edge["stage"] == 0
    assert edge["values"] == [0.1, -0.2, 0.3]
    assert math.isclose(edge["mean_abs"], 0.2, rel_tol=1e-6)
    assert math.isclose(edge["sign_agreement"], 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(edge["range"], 0.5, rel_tol=1e-6)

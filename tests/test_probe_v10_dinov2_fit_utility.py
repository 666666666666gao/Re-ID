from __future__ import annotations

import pytest
import torch


def test_prepare_dinov2_input_uses_fixed_geometry_and_imagenet_normalization() -> None:
    from tools.probe_v10_dinov2_fit_utility import prepare_dinov2_input

    images = torch.zeros(2, 3, 256, 128)
    prepared = prepare_dinov2_input(images)
    expected = torch.tensor(
        [
            (0.5 - 0.485) / 0.229,
            (0.5 - 0.456) / 0.224,
            (0.5 - 0.406) / 0.225,
        ]
    )

    assert prepared.shape == (2, 3, 252, 126)
    assert torch.allclose(prepared[0, :, 0, 0], expected)


def test_prepare_dinov2_state_dict_removes_only_mask_token_then_requires_exact_keys() -> None:
    from tools.probe_v10_dinov2_fit_utility import prepare_dinov2_state_dict

    state = {
        "mask_token": torch.ones(1, 1, 4),
        "patch_embed.proj.weight": torch.ones(4, 3, 2, 2),
    }
    prepared = prepare_dinov2_state_dict(
        state,
        model_keys={"patch_embed.proj.weight"},
    )

    assert set(prepared) == {"patch_embed.proj.weight"}
    assert "mask_token" in state

    with pytest.raises(ValueError, match="exactly one mask_token"):
        prepare_dinov2_state_dict(
            {"patch_embed.proj.weight": state["patch_embed.proj.weight"]},
            model_keys={"patch_embed.proj.weight"},
        )
    with pytest.raises(ValueError, match="state keys do not match"):
        prepare_dinov2_state_dict(
            state,
            model_keys={"patch_embed.proj.weight", "unexpected"},
        )


def test_dinov2_embedding_and_fixed_equal_block_composition() -> None:
    from tools.probe_v10_dinov2_fit_utility import (
        compose_equal_block_embedding,
        dinov2_global_embedding,
    )

    tokens = torch.arange(2 * 3 * 163 * 4, dtype=torch.float32).reshape(
        2, 3, 163, 4
    )
    dino = dinov2_global_embedding(tokens)
    phase_b = torch.randn(2, 7)
    fused = compose_equal_block_embedding(phase_b, dino)

    assert dino.shape == (2, 3 * 2 * 4)
    assert torch.equal(dino[:, :4], tokens[:, 0, 0])
    assert torch.allclose(dino[:, 4:8], tokens[:, 0, 1:].mean(dim=1))
    assert fused.shape == (2, phase_b.shape[1] + dino.shape[1])
    assert torch.allclose(fused[:, :7].norm(dim=1), torch.ones(2))
    assert torch.allclose(fused[:, 7:].norm(dim=1), torch.ones(2))


def test_qualification_gate_requires_fixed_gain_oracle_and_two_source_wins() -> None:
    from tools.probe_v10_dinov2_fit_utility import evaluate_qualification_gate

    passing = evaluate_qualification_gate(
        phase_b_map=58.0,
        concat_map=59.1,
        oracle_gain_map=2.1,
        unique_ap_wins={"phase_b": 3, "dinov2": 4},
        min_concat_gain_map=1.0,
        min_oracle_gain_map=2.0,
    )
    weak_concat = evaluate_qualification_gate(
        phase_b_map=58.0,
        concat_map=58.9,
        oracle_gain_map=2.1,
        unique_ap_wins={"phase_b": 3, "dinov2": 4},
        min_concat_gain_map=1.0,
        min_oracle_gain_map=2.0,
    )
    collapsed = evaluate_qualification_gate(
        phase_b_map=58.0,
        concat_map=59.1,
        oracle_gain_map=2.1,
        unique_ap_wins={"phase_b": 0, "dinov2": 4},
        min_concat_gain_map=1.0,
        min_oracle_gain_map=2.0,
    )

    assert passing["passed"] is True
    assert weak_concat["passed"] is False
    assert collapsed["passed"] is False

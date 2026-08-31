from __future__ import annotations

from pathlib import Path

import pytest
import torch


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(3, 2)
        self.reliability_gate = torch.nn.Linear(2, 1)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.head = torch.nn.Linear(2, 2)


def _uniform_state(model: _Model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("encoder.reliability_gate.")
    }


def test_full_model_accepts_only_the_new_circ_posterior_as_missing(tmp_path: Path) -> None:
    from modeling.trifusion.warm_start import load_hfer_uniform_warm_start

    model = _Model()
    checkpoint = tmp_path / "uniform.pth"
    torch.save(_uniform_state(model), checkpoint)

    evidence = load_hfer_uniform_warm_start(model, checkpoint)

    assert evidence["missing_keys"] == [
        "encoder.reliability_gate.bias",
        "encoder.reliability_gate.weight",
    ]
    assert evidence["unexpected_keys"] == []
    assert evidence["loaded_tensor_count"] == len(_uniform_state(model))


def test_full_model_rejects_any_nonposterior_warm_start_gap(tmp_path: Path) -> None:
    from modeling.trifusion.warm_start import load_hfer_uniform_warm_start

    model = _Model()
    incomplete = _uniform_state(model)
    incomplete.pop("head.weight")
    checkpoint = tmp_path / "incomplete.pth"
    torch.save(incomplete, checkpoint)

    with pytest.raises(ValueError, match="non-posterior"):
        load_hfer_uniform_warm_start(model, checkpoint)


def test_postfreeze_warm_start_reinitializes_only_changed_identity_heads(
    tmp_path: Path,
) -> None:
    from modeling.trifusion.warm_start import load_hfer_uniform_warm_start

    source = _Model()
    target = _Model()
    target.head = torch.nn.Linear(2, 3)
    checkpoint = tmp_path / "dev-uniform.pth"
    torch.save(_uniform_state(source), checkpoint)

    evidence = load_hfer_uniform_warm_start(
        target,
        checkpoint,
        allow_classifier_reinitialization=True,
        classifier_prefixes=("head.",),
    )

    assert evidence["classifier_reinitialized"] is True
    assert evidence["reinitialized_classifier_keys"] == ["head.bias", "head.weight"]
    assert set(evidence["missing_keys"]) == {
        "encoder.reliability_gate.bias",
        "encoder.reliability_gate.weight",
        "head.bias",
        "head.weight",
    }

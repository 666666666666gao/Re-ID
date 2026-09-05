from copy import deepcopy

import pytest
import torch

from modeling.trifusion.experts.mamba import TinySequenceMixer
from modeling.trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
from modeling.trifusion.signal_preserving_v8_builder import (
    build_signal_preserving_trifusion_v8_expert_formation,
)
from modeling.trifusion.signal_preserving_v23 import (
    SpectralSemanticTailEncoderV23, SpectralStageAdapterV23,
)
from test_trifusion_signal_preserving_v8 import _FakeSignal, _batch


@pytest.fixture(autouse=True)
def cuda_device():
    assert torch.cuda.is_available()
    with torch.device("cuda"):
        yield


def build():
    torch.manual_seed(42)
    return build_signal_preserving_trifusion_v8_expert_formation(
        _FakeSignal(), signal_checkpoint_sha256="8" * 64, num_classes=2,
        feature_width=4, semantic_width=6, grid_size=(2, 2), branch_after_block=0,
        adapter_width=4, expert_modal_width=8, mamba_mixer_factory=TinySequenceMixer,
    ).model


@pytest.mark.parametrize("train_adapters", [False, True])
def test_zero_adapters_preserve_all_outputs_and_strict_reload(train_adapters):
    model, batch = build().eval(), _batch()
    with torch.no_grad():
        original = model(batch, return_aux=True)
    model.encoder = SpectralSemanticTailEncoderV23(
        model.encoder, adapter_width=4, train_adapters=train_adapters,
    )
    model.eval()
    with torch.no_grad():
        adapted = model(batch, return_aux=True)
    assert torch.equal(original.baseline_embedding, adapted.baseline_embedding)
    assert torch.equal(original.fused_embedding, adapted.fused_embedding)
    for expert in ("cnn", "transformer", "mamba"):
        assert torch.equal(original.branch_embeddings[expert], adapted.branch_embeddings[expert])
        assert torch.equal(original.modal_residual_embeddings[expert], adapted.modal_residual_embeddings[expert])
    restored = build()
    restored.encoder = SpectralSemanticTailEncoderV23(
        restored.encoder, adapter_width=4, train_adapters=train_adapters,
    )
    restored.load_state_dict(deepcopy(model.state_dict()), strict=True)
    restored.eval()
    with torch.no_grad():
        assert torch.equal(adapted.fused_embedding, restored(batch))


def test_modality_dispatch_is_isolated_and_zero_up_initialization_has_expected_gradients():
    torch.manual_seed(42)
    stage = SpectralStageAdapterV23(6, 4)
    x = torch.randn(4, 3, 5, 6)
    assert torch.equal(stage(x), x)
    stage(x)[:, 1].square().sum().backward()
    for index, adapter in enumerate(stage.modal_adapters):
        assert torch.count_nonzero(adapter[0].weight.grad) == 0
        if index == 1:
            assert torch.count_nonzero(adapter[2].weight.grad) > 0
        else:
            assert torch.count_nonzero(adapter[2].weight.grad) == 0
    with torch.no_grad():
        stage.modal_adapters[1][2].weight.normal_(std=0.1)
    output = stage(x)
    assert torch.equal(output[:, 0], x[:, 0])
    assert torch.equal(output[:, 2], x[:, 2])
    assert not torch.equal(output[:, 1], x[:, 1])


@pytest.mark.parametrize("train_adapters", [False, True])
def test_actual_updates_reach_candidate_adapters_and_leave_signal_unchanged(train_adapters):
    model, batch = build(), _batch()
    model.encoder = SpectralSemanticTailEncoderV23(
        model.encoder, adapter_width=4, train_adapters=train_adapters,
    )
    before = deepcopy(model.encoder.spectral_stages.state_dict())
    model.eval()
    with torch.no_grad():
        baseline = model(batch, retrieval_output="baseline_only").clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    criterion = ExpertFormationV8Criterion(triplet_margin=0.3, label_smoothing=0.1)
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(batch, return_aux=True), torch.tensor([0, 0, 1, 1]))
        sum(losses.values()).backward()
        assert all(p.grad is None for p in model.baseline.parameters())
        optimizer.step()
    changed = [not torch.equal(before[name], value)
               for name, value in model.encoder.spectral_stages.state_dict().items()]
    assert all(changed) if train_adapters else not any(changed)
    model.eval()
    with torch.no_grad():
        assert torch.equal(baseline, model(batch, retrieval_output="baseline_only"))

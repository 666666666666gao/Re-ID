from copy import deepcopy

import pytest
import torch

from modeling.trifusion.experts.mamba import TinySequenceMixer
from modeling.trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
from modeling.trifusion.signal_preserving_v8_builder import (
    build_signal_preserving_trifusion_v8_expert_formation,
)
from modeling.trifusion.signal_preserving_v19 import (
    PrivateSemanticTailEncoderV19, optimizer_parameter_groups,
    private_tail_storage_is_disjoint,
)
from test_trifusion_signal_preserving_v8 import _FakeSignal, _batch


def build():
    torch.manual_seed(42)
    return build_signal_preserving_trifusion_v8_expert_formation(
        _FakeSignal(), signal_checkpoint_sha256="8" * 64, num_classes=2,
        feature_width=4, semantic_width=6, grid_size=(2, 2), branch_after_block=0,
        adapter_width=4, expert_modal_width=8, mamba_mixer_factory=TinySequenceMixer,
    ).model


@pytest.mark.parametrize("train_tail", [False, True])
def test_private_copy_preserves_all_initial_outputs_and_reload(train_tail):
    model = build().eval()
    batch = _batch()
    with torch.no_grad():
        before = model(batch, return_aux=True)
    model.encoder = PrivateSemanticTailEncoderV19(model.encoder, train_private_tail=train_tail)
    model.eval()
    assert private_tail_storage_is_disjoint(model.encoder)
    with torch.no_grad():
        after = model(batch, return_aux=True)
    assert torch.equal(before.baseline_embedding, after.baseline_embedding)
    assert torch.equal(before.fused_embedding, after.fused_embedding)
    for expert in ("cnn", "transformer", "mamba"):
        assert torch.equal(before.branch_embeddings[expert], after.branch_embeddings[expert])
    reloaded = build()
    reloaded.encoder = PrivateSemanticTailEncoderV19(reloaded.encoder, train_private_tail=train_tail)
    reloaded.load_state_dict(deepcopy(model.state_dict()), strict=True)
    reloaded.eval()
    with torch.no_grad():
        assert torch.equal(after.fused_embedding, reloaded(batch))


@pytest.mark.parametrize("train_tail", [False, True])
def test_optimizer_updates_private_tails_only_in_candidate_and_preserves_signal(train_tail):
    model = build()
    model.encoder = PrivateSemanticTailEncoderV19(model.encoder, train_private_tail=train_tail)
    batch = _batch()
    model.eval()
    with torch.no_grad():
        baseline = model(batch, retrieval_output="baseline_only").clone()
    before = {name: p.detach().clone() for name, p in model.encoder.private_tails.named_parameters()}
    groups = optimizer_parameter_groups(model, role_lr=0.01, tail_lr=0.001)
    actual = [id(p) for group in groups for p in group["params"]]
    assert len(actual) == len(set(actual))
    assert set(actual) == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.SGD(groups)
    model.train()
    losses = ExpertFormationV8Criterion(triplet_margin=0.3, label_smoothing=0.1)(
        model(batch, return_aux=True), torch.tensor([0, 0, 1, 1]),
    )
    sum(losses.values()).backward()
    optimizer.step()
    changed = {
        name: not torch.equal(before[name], p)
        for name, p in model.encoder.private_tails.named_parameters()
    }
    assert all(changed.values()) if train_tail else not any(changed.values())
    assert all(p.grad is None for p in model.baseline.parameters())
    model.eval()
    with torch.no_grad():
        assert torch.equal(baseline, model(batch, retrieval_output="baseline_only"))

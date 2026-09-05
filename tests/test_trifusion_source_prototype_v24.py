from copy import deepcopy
import random

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from modeling.trifusion.source_prototype_v24 import CameraIdentityPrototypeMemory
from modeling.trifusion.dual_view_data_v24 import SharedGeometryDualViewTransform


@pytest.fixture(autouse=True)
def remote_cuda_required():
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    random.seed(42)


def fixture_memory():
    features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 0., 0.]], device="cuda")
    labels = torch.tensor([0, 0, 1, 2], device="cuda")
    cameras = torch.tensor([0, 1, 0, 1], device="cuda")
    return CameraIdentityPrototypeMemory(features, labels, cameras)


def test_global_identity_prototype_balances_cameras_not_image_counts():
    features = torch.tensor([[1., 0.]] * 9 + [[0., 1.], [-1., 0.]], device="cuda")
    labels = torch.tensor([0] * 10 + [1], device="cuda")
    cameras = torch.tensor([0] * 9 + [1, 0], device="cuda")
    memory = CameraIdentityPrototypeMemory(features, labels, cameras)
    expected = F.normalize(torch.tensor([1., 1.], device="cuda"), dim=0)
    torch.testing.assert_close(memory.global_prototypes()[0], expected)
    assert len(memory.prototypes) == 3 and len(memory.class_weights) == 2
    assert list(memory.parameters()) == []


def test_losses_use_real_global_identity_and_same_camera_competitors():
    memory = fixture_memory()
    query = torch.tensor([[0.8, 0.2, 0.4]], device="cuda")
    label = torch.tensor([0], device="cuda")
    camera = torch.tensor([0], device="cuda")
    total, parts = memory.loss(query, label, camera)
    q = F.normalize(query, dim=1)
    global_targets = torch.tensor([[2**-0.5, 2**-0.5, 0.], [0., 0., 1.], [-1., 0., 0.]], device="cuda")
    environment_targets = torch.tensor([[1., 0., 0.], [0., 0., 1.]], device="cuda")
    global_expected = F.cross_entropy(q @ global_targets.T / 0.05, label)
    environment_expected = F.cross_entropy(q @ environment_targets.T / 0.05, label)
    torch.testing.assert_close(parts["prototype_global"], global_expected)
    torch.testing.assert_close(parts["prototype_environment"], environment_expected)
    torch.testing.assert_close(total, (global_expected + environment_expected) / 2)


def test_strong_loss_does_not_update_memory_and_has_live_query_gradient():
    memory = fixture_memory()
    before = deepcopy(memory.state_dict())
    query = torch.tensor([[0.2, 0.3, 0.9]], device="cuda", requires_grad=True)
    loss, _ = memory.loss(query, torch.tensor([0], device="cuda"), torch.tensor([0], device="cuda"))
    loss.backward()
    assert bool(torch.isfinite(query.grad).all()) and float(query.grad.abs().sum()) > 0
    for key, value in memory.state_dict().items():
        assert torch.equal(value, before[key]) and not value.requires_grad


def test_weak_updates_group_duplicates_once_and_preserve_unobserved_pairs():
    memory, reversed_memory = fixture_memory(), fixture_memory()
    before = memory.prototypes.clone()
    weak = torch.tensor([[0., 1., 0.], [0., 0., 1.]], device="cuda", requires_grad=True)
    labels = torch.tensor([0, 0], device="cuda")
    cameras = torch.tensor([0, 0], device="cuda")
    memory.update(weak, labels, cameras, step=7)
    reversed_memory.update(weak.flip(0), labels, cameras, step=7)
    expected = F.normalize(torch.tensor([0.2, 0.4, 0.4], device="cuda"), dim=0)
    torch.testing.assert_close(memory.prototypes[0], expected)
    torch.testing.assert_close(memory.prototypes, reversed_memory.prototypes)
    assert torch.equal(memory.prototypes[1:], before[1:])
    assert memory.update_count.tolist() == [1, 0, 0, 0]
    assert memory.last_update.tolist() == [7, 0, 0, 0]
    assert weak.grad is None


def test_label_zero_and_cross_camera_positive_survive_memory_roundtrip():
    memory = fixture_memory()
    restored = fixture_memory()
    restored.load_state_dict(deepcopy(memory.state_dict()), strict=True)
    labels, cameras = torch.tensor([0], device="cuda"), torch.tensor([0], device="cuda")
    coverage = restored.coverage(labels, cameras, step=3)
    assert coverage["global_negative_identities_per_anchor"] == 2
    assert coverage["environment_negative_relations"] == 1
    assert coverage["anchors_with_real_other_camera_positive"] == 1
    assert coverage["real_other_camera_positive_prototypes"] == 1
    assert coverage["prototype_age_max_steps"] == 3


def test_geometry_is_shared_across_modalities_and_weak_strong_views():
    pixels = np.arange(32 * 16 * 3, dtype=np.uint8).reshape(32, 16, 3)
    image = Image.fromarray(pixels)
    transform = SharedGeometryDualViewTransform(size=(16, 8), padding=3,
                                               weak_erase=0, strong_erase=0, brightness=0)
    weak, strong = transform([image.copy() for _ in range(3)])
    assert weak[0].shape == (3, 16, 8)
    for value in [*weak, *strong]:
        assert torch.equal(value, weak[0])

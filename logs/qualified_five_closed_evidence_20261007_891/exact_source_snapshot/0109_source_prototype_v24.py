"""Training-only identity prototypes with real camera membership for V24."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CameraIdentityPrototypeMemory(nn.Module):
    """Weak-view EMA per identity/camera; strong-view global and camera ID loss."""

    def __init__(self, features, labels, cameras, *, temperature=0.05, momentum=0.2):
        super().__init__()
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        features = F.normalize(features.detach().float(), dim=1)
        pairs = torch.unique(torch.stack((labels, cameras), dim=1), dim=0, sorted=True)
        class_ids = torch.unique(labels, sorted=True)
        assert torch.equal(class_ids, torch.arange(len(class_ids), device=labels.device))
        prototypes = torch.stack([
            features[(labels == identity) & (cameras == camera)].mean(dim=0)
            for identity, camera in pairs
        ])
        self.register_buffer("prototypes", F.normalize(prototypes, dim=1))
        self.register_buffer("pair_labels", pairs[:, 0])
        self.register_buffer("pair_cameras", pairs[:, 1])
        weights = class_ids[:, None].eq(pairs[:, 0][None, :]).float()
        self.register_buffer("class_weights", weights / weights.sum(dim=1, keepdim=True))
        self.register_buffer("last_update", torch.zeros(len(pairs), dtype=torch.long, device=labels.device))
        self.register_buffer("update_count", torch.zeros_like(self.last_update))

    def global_prototypes(self):
        # Each real camera contributes equally to its identity, regardless of image count.
        return F.normalize(self.class_weights @ self.prototypes, dim=1)

    def loss(self, strong_features, labels, cameras):
        query = F.normalize(strong_features.float(), dim=1)
        global_logits = query @ self.global_prototypes().T / self.temperature
        environment_logits = query @ self.prototypes.T / self.temperature
        same_camera = cameras[:, None].eq(self.pair_cameras[None, :])
        targets = same_camera & labels[:, None].eq(self.pair_labels[None, :])
        assert bool(targets.sum(dim=1).eq(1).all())
        environment_logits = environment_logits.masked_fill(~same_camera, -torch.inf)
        global_loss = F.cross_entropy(global_logits, labels)
        environment_loss = F.cross_entropy(environment_logits, targets.long().argmax(dim=1))
        return (global_loss + environment_loss) / 2, {
            "prototype_global": global_loss,
            "prototype_environment": environment_loss,
        }

    @torch.no_grad()
    def update(self, weak_features, labels, cameras, *, step):
        weak = F.normalize(weak_features.detach().float(), dim=1)
        for index, (identity, camera) in enumerate(zip(self.pair_labels, self.pair_cameras, strict=True)):
            selected = labels.eq(identity) & cameras.eq(camera)
            if bool(selected.any()):
                # One update per observed identity/camera, not one EMA per repeated instance.
                mean = weak[selected].mean(dim=0)
                mixed = self.momentum * self.prototypes[index] + (1 - self.momentum) * mean
                self.prototypes[index].copy_(F.normalize(mixed, dim=0))
                self.last_update[index] = int(step)
                self.update_count[index] += 1

    def coverage(self, labels, cameras, *, step):
        same_identity = labels[:, None].eq(self.pair_labels[None, :])
        same_camera = cameras[:, None].eq(self.pair_cameras[None, :])
        other_camera_positive = same_identity & ~same_camera
        ages = int(step) - self.last_update
        return {
            "global_negative_identities_per_anchor": len(self.class_weights) - 1,
            "environment_negative_relations": int((same_camera & ~same_identity).sum()),
            "anchors_with_real_other_camera_positive": int(other_camera_positive.any(dim=1).sum()),
            "real_other_camera_positive_prototypes": int(other_camera_positive.sum()),
            "prototype_age_max_steps": int(ages.max()),
            "prototype_age_mean_steps": float(ages.float().mean()),
            "prototype_pairs_updated": int(self.update_count.gt(0).sum()),
        }

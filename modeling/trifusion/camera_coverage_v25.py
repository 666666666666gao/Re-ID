"""Registered RGBNT201 V25 camera-positive coverage sampler."""
import random

from .aligned_data import CrossCameraIdentitySampler


class TwoCrossCameraIdentitySampler(CrossCameraIdentitySampler):
    """Use two source partitions to supply two cross-camera K8 groups per batch."""

    def __iter__(self):
        assert self.batch_size == 64 and self.num_instances == 8
        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        queues, cross_camera = {}, set()
        for identity in self.indices_by_identity:
            groups, is_cross_camera = self._identity_groups(identity, rng)
            rng.shuffle(groups)
            queues[identity] = groups
            if is_cross_camera:
                cross_camera.add(identity)
        for identity in sorted(cross_camera):
            extra, is_cross_camera = self._identity_groups(identity, rng)
            assert is_cross_camera
            queues[identity].extend(extra)
            rng.shuffle(queues[identity])

        final_indices = []
        for _batch in range(self.length // self.batch_size):
            anchors = [identity for identity in cross_camera if queues[identity]]
            rng.shuffle(anchors)
            anchors.sort(key=lambda identity: len(queues[identity]), reverse=True)
            assert len(anchors) >= 2
            selected = anchors[:2]
            available = [
                identity for identity, groups in queues.items()
                if groups and identity not in cross_camera
            ]
            rng.shuffle(available)
            available.sort(key=lambda identity: len(queues[identity]), reverse=True)
            assert len(available) >= 6
            selected.extend(available[:6])
            for identity in selected:
                final_indices.extend(queues[identity].pop())
        return iter(final_indices)


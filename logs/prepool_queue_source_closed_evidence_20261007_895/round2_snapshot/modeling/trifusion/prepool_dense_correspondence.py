"""H2a: known same-image geometry supervises existing pre-pooling role keys."""
import random
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from .global_task_role_heads import GlobalTaskRoleHeads
from .state import MODALITY_ORDER


class DenseTrainingBatch(NamedTuple):
    images: dict
    labels: torch.Tensor
    cameras: torch.Tensor
    views: torch.Tensor
    paths: tuple
    visible: torch.Tensor


class VisibilityTransform:
    def __init__(self, transform):
        self.transforms = transform.transforms
        assert len(self.transforms) == 7
        self.visible = []

    def __call__(self, image):
        for transform in self.transforms[:-1]:
            image = transform(image)
        before = image.clone()
        image = self.transforms[-1](image)
        visible = before.eq(image).all(dim=0)
        # The original Pad(10)->RandomCrop can leave artificial boundary pixels.
        # This conservative mask also removes some real border content.
        visible[:10] = visible[-10:] = False
        visible[:, :10] = visible[:, -10:] = False
        self.visible.append(visible)
        return image


class VisibilityDataset(Dataset):
    def __init__(self, original):
        self.original = original
        self.transform = VisibilityTransform(original.transform)
        original.transform = self.transform

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        self.transform.visible = []
        row = self.original[index]
        assert len(row) == 5 and len(self.transform.visible) == 3
        return (*row, torch.stack(self.transform.visible))


def visible_collate(samples):
    from data.datasets.make_dataloader import train_collate_fn

    raw = train_collate_fn([sample[:5] for sample in samples])
    return DenseTrainingBatch(*raw, torch.stack([sample[5] for sample in samples]))


def attach_visibility(loader):
    """Reuse the actual sampler and original five indexed fields for batch logging."""
    old = loader.loader
    assert old.num_workers == 4 and not old.pin_memory
    assert old.generator is None and old.worker_init_fn is None
    loader.loader = DataLoader(VisibilityDataset(old.dataset), batch_sampler=old.batch_sampler,
                              num_workers=4, collate_fn=visible_collate)
    return loader


def geometric_counts(visible, flip, dy, dx):
    """Count visible primary pixels in each pair of 16x16 patch footprints."""
    assert visible.dtype == torch.bool and visible.ndim == 3
    assert visible.device.type == 'cpu'
    batch, height, width = visible.shape
    assert height % 16 == width % 16 == 0
    columns = width // 16
    patch_count = (height // 16) * columns
    yy = torch.arange(height)[:, None].expand(height, width).flatten()
    xx = torch.arange(width)[None, :].expand(height, width).flatten()
    second_y = yy - dy
    second_x = (width - 1 - xx if flip else xx) - dx
    valid = (second_y >= 0) & (second_y < height) & (second_x >= 0) & (second_x < width)
    source_patch = (yy // 16) * columns + xx // 16
    second_patch = (second_y // 16) * columns + second_x // 16
    # Invalid translated pixels have zero mass; clipping only permits scatter.
    index = source_patch * patch_count + second_patch.clamp(0, patch_count - 1)
    counts = torch.zeros(batch, patch_count * patch_count, dtype=torch.int64)
    counts.scatter_add_(1, index[None].expand(batch, -1),
                        (visible.flatten(1) & valid[None]).to(torch.int64))
    return counts.reshape(batch, patch_count, patch_count)


def teacher_keys(roles, stages):
    keys = []
    height, width = roles.grid
    for role in range(3):
        source = roles._depth_role(stages, role)[:, :, 1:]
        if role == 0:
            grid = source.reshape(-1, height, width, roles.width).permute(0, 3, 1, 2)
            grid = grid + roles.cnn(grid)
            source = grid.permute(0, 2, 3, 1).reshape(source.shape)
        with torch.autocast(source.device.type, enabled=False):
            keys.append(roles.key_projections[role](F.layer_norm(source.float(), (roles.width,))))
    return keys


def dense_objective(student_keys, other_keys, queries, counts):
    """One fixed auxiliary loss; actual Q-K observations do not add another loss."""
    counts = counts.to(student_keys[0].device, dtype=torch.float32)
    row_mass, column_mass = counts.sum(-1), counts.sum(-2)
    valid_rows, valid_columns = row_mass > 0, column_mass > 0
    assert bool(valid_rows.any(-1).all()) and bool(valid_columns.any(-1).all())
    targets = counts.clone()
    targets[valid_rows] = targets[valid_rows] / row_mass[valid_rows].unsqueeze(-1)
    row_weight, column_weight = row_mass / 256, column_mass / 256
    losses, match, entropy, mapped_l1 = [], [], [], []
    for student, other, query in zip(student_keys, other_keys, queries):
        with torch.autocast(student.device.type, enabled=False):
            assert student.shape[-2:] == other.shape[-2:] == (128, 128)
            logits = F.normalize(student.float(), dim=-1) @ F.normalize(other.float(), dim=-1).transpose(-1, -2)
            logits = logits * 128 ** 0.5
            logits = logits.masked_fill(~valid_columns[:, :, None], torch.finfo(logits.dtype).min)
            cell_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(-1)
            for modality in range(3):
                losses.append((cell_loss[:, modality] * row_weight[:, modality]).sum() /
                              row_weight[:, modality].sum())
            with torch.no_grad():
                mass = targets.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
                match.append(float((mass * row_weight).sum() / row_weight.sum()))
                primary = (query[:, None].float() @ student.float().transpose(-1, -2) * 128 ** -0.5).softmax(-1)
                second = (query[:, None].float() @ other.float().transpose(-1, -2) * 128 ** -0.5).softmax(-1)
                entropy.append(float(torch.special.entr(primary).sum(-1).mean()))
                mapped = (primary * row_weight[:, :, None]) @ targets
                restricted_second = second * column_weight[:, :, None]
                mapped_l1.append(float((mapped - restricted_second).abs().sum(-1).mean()))
    return torch.stack(losses).mean(), {
        'dense_match_mass_by_role': match, 'actual_read_entropy_by_role': entropy,
        'actual_read_mapped_l1_by_role': mapped_l1,
        'dense_supported_rows_by_modality': valid_rows.sum((0, 2)).tolist(),
        'dense_supported_columns_by_modality': valid_columns.sum((0, 2)).tolist(),
        'dense_overlap_pixels_by_modality': counts.sum((0, 2, 3)).tolist(),
    }


class PrepoolDenseHeads(GlobalTaskRoleHeads):
    def __init__(self, evidence_model):
        super().__init__(evidence_model)
        self.geometry_rng = random.Random(42)
        self.capture = False
        self.keys, self.queries = [None] * 3, [None] * 3
        self.verify_teacher_state = False
        self.collect_global = False
        self.global_rows = []
        for role in range(3):
            def key_hook(_module, _input, output, role=role):
                if self.capture:
                    self.keys[role] = output

            def query_hook(_module, _input, output, role=role):
                if self.capture:
                    self.queries[role] = output

            evidence_model.roles.key_projections[role].register_forward_hook(key_hook)
            evidence_model.roles.query_projections[role].register_forward_hook(query_hook)

        def global_hook(_module, _input, output):
            if self.collect_global and not self.training:
                self.global_rows.append(F.normalize(output[1].detach().float(), dim=1).cpu())

        evidence_model.backbone.register_forward_hook(global_hook)

    def forward(self, batch, *, return_aux=False):
        if not (return_aux and self.training):
            return super().forward(batch, return_aux=return_aux)
        self.keys, self.queries = [None] * 3, [None] * 3
        images, counts, transforms = {}, [], []
        for index, name in enumerate(MODALITY_ORDER):
            flip = self.geometry_rng.random() < 0.5
            dy, dx = self.geometry_rng.randint(-10, 10), self.geometry_rng.randint(-10, 10)
            image = batch['images'][name]
            height, width = image.shape[-2:]
            assert (height // 16, width // 16) == self.evidence_model.roles.grid
            image = image.flip(-1) if flip else image
            image = F.pad(image, (10, 10, 10, 10), value=-1)
            images[name] = image[:, :, 10 + dy:10 + dy + height, 10 + dx:10 + dx + width]
            counts.append(geometric_counts(batch['dense_visibility'][:, index], flip, dy, dx))
            transforms.append({'modality': name, 'flip': flip, 'dy': dy, 'dx': dx})
        if self.verify_teacher_state:
            buffers = {name: value.clone() for name, value in self.named_buffers()}
            cpu_rng = torch.get_rng_state()
            gpu_rng = [torch.cuda.get_rng_state(device) for device in (0, 1)]
        with torch.random.fork_rng(devices=[0, 1]), torch.no_grad(), torch.autocast('cuda', cache_enabled=False):
            stages, _global = self.evidence_model.backbone(images, batch['camera_ids'])
            other_keys = teacher_keys(self.evidence_model.roles, stages)
        del stages, _global, images
        if self.verify_teacher_state:
            assert all(torch.equal(value, buffers[name]) for name, value in self.named_buffers())
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert all(torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in zip((0, 1), gpu_rng))
        self.capture = True
        output = super().forward(batch, return_aux=True)
        self.capture = False
        assert all(value is not None for value in self.keys + self.queries)
        auxiliary, stats = dense_objective(self.keys, other_keys, self.queries, torch.stack(counts, dim=1))
        self.keys, self.queries = [None] * 3, [None] * 3
        return {**output, 'dense_loss': auxiliary, 'dense_stats': {**stats,
            'geometry_transforms': transforms, 'teacher_state_rng_checked': self.verify_teacher_state}}

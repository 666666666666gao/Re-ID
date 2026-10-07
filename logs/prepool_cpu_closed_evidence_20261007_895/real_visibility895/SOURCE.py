import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from datetime import datetime
from pathlib import Path
from argparse import Namespace
from typing import NamedTuple
import hashlib,json,random,sys
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from yacs.config import CfgNode
torch.set_num_threads(2)
root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,str(root))
from tools import run_foundation_recipe as foundation
from tools.run_training_feature_scale import BatchOrderLoader
from tools.run_signal_baseline_dev import _configure_signal_source
_configure_signal_source(root/'comparators/Signal-cd1b0a6')
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
controls=json.loads((root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
scratch=Path('/tmp/trifusion_visibility_cpu_20261007_895')
assert not scratch.exists()
scratch.mkdir()

def sha(value): return hashlib.sha256(value).hexdigest()

def measure(dataset,wrapped):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    old=next(row for row in controls['rows'] if row['dataset']==dataset and row['variant']=='semantic')
    cfg=CfgNode.load_cfg(old['initializer']['cfg_yaml'])
    protocol=foundation.runner.read_protocol(root/'logs/training_feature_scale_protocols_20261002'/f'{dataset}.json',dataset)
    label='wrapped' if wrapped else 'original'
    loader=BatchOrderLoader(foundation.train_loader(Namespace(recipe='author'),protocol,cfg),scratch/f'{dataset}_{label}.jsonl')
    if wrapped: attach_visibility(loader)
    rows=[]
    iterator=iter(loader)
    for batch_index in range(8):
        raw=next(iterator)
        images,labels,cameras,views,paths=raw[:5]
        row={'batch':batch_index,'images':{name:sha(value.numpy().tobytes()) for name,value in images.items()},
             'labels':labels.tolist(),'cameras':cameras.tolist(),'views':views.tolist(),'paths':paths}
        if wrapped:
            assert isinstance(raw,DenseTrainingBatch)
            assert raw.visible.dtype==torch.bool and raw.visible.shape==(cfg.SOLVER.IMS_PER_BATCH,3,*cfg.INPUT.SIZE_TRAIN)
            assert not bool(raw.visible[:,:,:10].any()) and not bool(raw.visible[:,:,-10:].any())
            assert not bool(raw.visible[:,:,:,:10].any()) and not bool(raw.visible[:,:,:,-10:].any())
            assert bool(raw.visible.flatten(2).any(-1).all())
        rows.append(row)
    iterator._shutdown_workers()
    del iterator,loader
    rng={'torch':sha(torch.get_rng_state().numpy().tobytes()),'python':sha(repr(random.getstate()).encode()),
         'numpy':sha(repr(np.random.get_state()).encode())}
    return rows,rng,sha((scratch/f'{dataset}_{label}.jsonl').read_bytes())

results=[]
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
    original,first_rng,first_log=measure(dataset,False)
    wrapped,second_rng,second_log=measure(dataset,True)
    assert original==wrapped,dataset
    assert first_rng==second_rng,dataset
    assert first_log==second_log,dataset
    results.append({'dataset':dataset,'batches':8,'images_labels_cameras_views_paths_exact':True,
                    'cpu_python_numpy_rng_exact':True,'batch_order_log_exact':True,
                    'batch_order_sha256':first_log,'primary_sha256':sha(json.dumps(original,sort_keys=True).encode())})

torch.manual_seed(42)
visible=torch.zeros(2,256,128,dtype=torch.bool)
visible[:,10:-10,10:-10]=True
visible[:,50:90,25:55]=False
counts=geometric_counts(visible,True,7,-4)[:,None].expand(-1,3,-1,-1).clone()
keys=[torch.randn(2,3,128,128,requires_grad=True) for _ in range(3)]
others=[torch.randn_like(key) for key in keys]
queries=[torch.randn(2,16,128) for _ in keys]
unused_global=torch.randn(2,1536,requires_grad=True)
loss,stats=dense_objective(keys,others,queries,counts)
grads=torch.autograd.grad(loss,[unused_global]+keys,allow_unused=True)
assert grads[0] is None
assert all(grad is not None and bool(torch.isfinite(grad).all()) and float(grad.norm())>0 for grad in grads[1:])
assert bool(torch.isfinite(loss))
print(json.dumps({'status':'REAL_AUTHOR_LOADER_FIRST8_AND_CPU_DENSE_VJP_PASS',
    'at':datetime.now().astimezone().isoformat(),'datasets':results,
    'cpu_fixture_loss':float(loss.detach()),'cpu_key_gradient_norms':[float(grad.norm()) for grad in grads[1:]],
    'cpu_global_unused':True,'cpu_dense_stats':stats,'production_model_forwards':0,'optimizer_updates':0,
    'boundary':'Extracted exact seven new source definitions; real CPU author DataLoader/workers4/first8 only, not full50 transform-byte equivalence. CPU random-key auxiliary shape/VJP only; no CUDA/model/teacher/M0/retrieval/GPU temperature/power/25 query or package install.'}))

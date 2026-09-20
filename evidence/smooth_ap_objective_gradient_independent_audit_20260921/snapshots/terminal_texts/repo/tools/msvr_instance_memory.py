"""Source-instance XBM mechanics, independently implemented from the paper idea."""
from collections import OrderedDict

import torch
import torch.nn.functional as F


class InstanceMemory:
    def __init__(self, records, *, capacity=512, maximum_age=8):
        self.records = {int(r['index']): r for r in records}
        self.capacity = capacity
        self.maximum_age = maximum_age
        self.entries = OrderedDict()

    def read(self, step, current_indices, prototype):
        expired = [i for i, x in self.entries.items() if step-x['step'] > self.maximum_age]
        for i in expired:
            del self.entries[i]
        current = set(current_indices)
        selected = [(i, x) for i, x in self.entries.items() if i not in current]
        metadata = [dict(record_index=i, identity=int(self.records[i]['identity']),
                         scene=int(self.records[i]['scene']), age=step-x['step'],
                         stored_step=x['step']) for i, x in selected]
        assert all(1 <= x['age'] <= self.maximum_age for x in metadata)
        values = torch.stack([x['feature'] for _, x in selected]) if selected else prototype.new_empty((0, prototype.shape[1]))
        assert not values.requires_grad and len(set(x['record_index'] for x in metadata)) == len(metadata)
        return values, metadata

    def update(self, step, indices, features):
        assert features.ndim == 2 and len(indices) == len(features)
        for index, feature in zip(indices, features.detach(), strict=True):
            index = int(index)
            assert index in self.records
            self.entries.pop(index, None)
            self.entries[index] = dict(step=step, feature=feature.clone())
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)


def expanded_triplet(features, identities, scenes, memory, metadata, margin=.3):
    """Current peers retain gradients; unique historical candidates are detached."""
    with torch.autocast(features.device.type, enabled=False):
        unit = F.normalize(features.float(), dim=1)
        ids = torch.as_tensor(identities, device=features.device)
        sc = torch.as_tensor(scenes, device=features.device)
        current = torch.cdist(unit, unit, p=2)
        positive = ids[:, None] == ids[None, :]
        positive &= ~torch.eye(len(ids), device=features.device, dtype=torch.bool)
        negative = ids[:, None] != ids[None, :]
        assert positive.any(1).all() and negative.any(1).all()
        hp = current.masked_fill(~positive, -torch.inf).max(1).values
        hn = current.masked_fill(~negative, torch.inf).min(1).values
        basic = F.relu(hp-hn+margin).mean()
        cached = torch.cdist(unit, memory.detach().float(), p=2)
        mids = torch.tensor([r['identity'] for r in metadata], device=features.device, dtype=ids.dtype)
        msc = torch.tensor([r['scene'] for r in metadata], device=features.device, dtype=sc.dtype)
        mp = ids[:, None] == mids[None, :]
        mn = ~mp
        if len(metadata):
            mph = cached.masked_fill(~mp, -torch.inf).max(1).values
            mnh = cached.masked_fill(~mn, torch.inf).min(1).values
            union_hp = torch.maximum(hp, mph)
            union_hn = torch.minimum(hn, mnh)
            harder_positive = int((mph > hp).sum())
            harder_negative = int((mnh < hn).sum())
        else:
            union_hp, union_hn = hp, hn
            harder_positive = harder_negative = 0
        loss = F.relu(union_hp-union_hn+margin).mean()
        stats = dict(memory_records=len(metadata), memory_positive_pairs=int(mp.sum()),
                     memory_negative_pairs=int(mn.sum()),
                     memory_cross_scene_positive_pairs=int((mp & (sc[:, None] != msc[None, :])).sum()),
                     memory_negative_violations_against_batch_hard_positive=int((mn & (cached < hp[:, None]+margin)).sum()),
                     harder_positive_anchors=harder_positive, harder_negative_anchors=harder_negative,
                     current_wrong_order_anchors=int((hp >= hn).sum()),
                     expanded_wrong_order_anchors=int((union_hp >= union_hn).sum()),
                     current_triplet=float(basic.detach()), expanded_triplet=float(loss.detach()),
                     expanded_hinge_positive_anchors=int((union_hp-union_hn+margin > 0).sum()),
                     maximum_memory_age=max((x['age'] for x in metadata), default=0))
    return loss, basic, unit, current, cached, stats


def replay_drift(model, probe):
    """Replay the exact stored training batch and RNG without changing buffers/RNG."""
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.set_rng_state(probe['cpu_rng'])
        torch.cuda.set_rng_state(probe['cuda_rng'])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            output = model(probe['batch'], return_aux=True)
            current = F.normalize(output.fused_embedding.float(), dim=1)
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    distances = (current-probe['feature']).norm(dim=1).detach().cpu().tolist()
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    return dict(record_indices=probe['indices'], feature_l2_drift=distances,
                same_pixels_and_rng=True, model_buffers_restored=True,
                minimum=float(min(distances)), maximum=float(max(distances)),
                mean=float(sum(distances)/len(distances)))


def check_math():
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(8, 12, generator=generator, requires_grad=True)
    ids = [0, 0, 1, 1, 2, 2, 3, 3]
    scenes = [0, 1]*4
    records = [dict(index=i, identity=ids[i], scene=scenes[i]) for i in range(8)]
    memory = InstanceMemory(records, capacity=3, maximum_age=2)
    memory.update(0, [0, 1, 0, 2], F.normalize(features[[0, 1, 0, 2]].detach(), dim=1))
    assert list(memory.entries) == [1, 0, 2]
    values, meta = memory.read(1, [0], features)
    assert [r['record_index'] for r in meta] == [1, 2]
    assert meta[0]['identity'] == 0 and not values.requires_grad
    memory.read(3, [], features)
    assert not memory.entries
    empty = features.detach().new_empty((0, features.shape[1]))
    loss, base, _, current, _, _ = expanded_triplet(features, ids, scenes, empty, [])
    assert torch.equal(loss, base)
    refs = F.normalize(torch.randn(4, 12, generator=generator), dim=1)
    metadata = [dict(record_index=10+i, identity=i, scene=1, age=1, stored_step=0) for i in range(4)]
    expanded, _, _, _, cached, stats = expanded_triplet(features, ids, scenes, refs, metadata)
    brute = []
    for i in range(8):
        positive = [current[i, j] for j in range(8) if ids[i] == ids[j] and i != j]
        negative = [current[i, j] for j in range(8) if ids[i] != ids[j]]
        positive += [cached[i, j] for j in range(4) if ids[i] == j]
        negative += [cached[i, j] for j in range(4) if ids[i] != j]
        brute.append(F.relu(torch.stack(positive).max()-torch.stack(negative).min()+.3))
    assert torch.equal(expanded, torch.stack(brute).mean())
    gradient = torch.autograd.grad(expanded, features)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    return dict(status='PASS_INSTANCE_MEMORY_MATH',class_zero_valid=True,
                duplicate_record_latest_view_only=True,current_records_excluded=True,
                exact_age_expiry=True,empty_memory_matches_current=True,
                brute_hard_triplet_equal=True,finite_nonzero_anchor_gradient=True,
                synthetic_stats=stats)

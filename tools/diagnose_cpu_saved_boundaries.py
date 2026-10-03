"""One semantic RGBNT201 saved-tensor/boundary diagnostic; two backwards, no updates."""
import argparse
from datetime import datetime
import inspect
import json
from pathlib import Path
import sys
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_native_cpu_saved as entry
from tools import queue_native_cpu_saved as panel

SCHEMA = 'trifusion-cpu-saved-boundary-diagnostic-v1'


def metadata(tensor):
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'storage_offset': tensor.storage_offset(), 'dtype': str(tensor.dtype),
            'layout': str(tensor.layout), 'device': str(tensor.device),
            'contiguous': tensor.is_contiguous(), 'requires_grad': tensor.requires_grad,
            'grad_fn': type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None}


class RecordedCPUStorage(torch.autograd.graph.saved_tensors_hooks):
    """Delegate each pack/unpack once to the installed builtin, recording metadata only."""
    def __init__(self):
        self.builtin = torch.autograd.graph.save_on_cpu(pin_memory=False)
        self.packs = []
        self.unpacks = []
        super().__init__(self.pack, self.unpack)

    def pack(self, tensor):
        index = len(self.packs)
        before = metadata(tensor)
        payload = self.builtin.pack_hook(tensor)
        self.packs.append({'index': index, 'before': before, 'cpu': metadata(payload[1])})
        return index, payload

    def unpack(self, record):
        index, payload = record
        tensor = self.builtin.unpack_hook(payload)
        self.unpacks.append({'index': index, 'order': len(self.unpacks), 'restored': metadata(tensor)})
        return tensor


def compare(x, y, tolerance):
    assert x.shape == y.shape
    return {'max_absolute_difference': float((x-y).abs().max()),
            'reference_max_abs': float(x.abs().max()), 'candidate_max_abs': float(y.abs().max()),
            'fixed_gate_pass': bool(torch.allclose(x, y, atol=tolerance, rtol=tolerance))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    assert not output_dir.exists()
    output_dir.mkdir(parents=True)
    started_at = datetime.now().astimezone().isoformat()
    (output_dir/'RUNNING.json').write_text(json.dumps({'schema': SCHEMA, 'started_at': started_at,
        'optimizer_updates': 0, 'scope': 'semantic RGBNT201 only; two backwards; no M0/training/official scoring/weights'})+'\n')
    panel.configure()
    entry.configure()
    dataset, variant = 'RGBNT201', 'semantic'
    previous = ROOT/'logs/independent_native_evidence_20261003_v4'
    protocol_path = panel.PROTOCOLS/f'{dataset}.json'
    protocol = entry.runner.read_protocol(protocol_path, dataset)
    values = argparse.Namespace(dataset=dataset, variant=variant, recipe=variant,
        mode='prepare', seed=42, epochs=50, protocol=protocol_path,
        signal_source=panel.SOURCE, clip_weight=panel.WEIGHTS/'ViT-B-16.pt',
        initialization=previous/'initialization'/f'{dataset}_{variant}.json',
        output_dir=output_dir, baseline_sha256=entry.runner.sha256(panel.WEIGHTS/'ViT-B-16.pt'))
    reference, cfg, reference_binding = entry.original_build_core(values, protocol)
    candidate, candidate_cfg, candidate_binding = entry.build_core(values, protocol)
    assert candidate_binding == panel.expected_binding(previous, dataset, variant)
    assert cfg.dump() == candidate_cfg.dump()
    assert entry.runner._module_state_sha256(reference) == entry.runner._module_state_sha256(candidate)
    raw = next(iter(entry.original_train_loader(values, protocol, cfg)))
    full_batch, full_labels = entry.runner._training_batch(raw)
    assert len(full_labels) == cfg.SOLVER.IMS_PER_BATCH == 64
    batch = {'images': {name: image[:32] for name,image in full_batch['images'].items()},
             'camera_ids': full_batch['camera_ids'][:32]}
    labels = full_labels[:32]
    assert len(labels.unique()) >= 2
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    storage = RecordedCPUStorage()
    backbone = candidate.evidence_model.backbone

    def recorded_forward(module, *positional, **keyword):
        with storage:
            return module._cpu_saved_original_forward(*positional, **keyword)

    # Replace the uninstrumented v4 wrapper; do not nest two saved-tensor contexts.
    backbone.forward = MethodType(recorded_forward, backbone)
    observed = []
    for model in (reference, candidate):
        boundaries = {}

        def capture(_module, _inputs, result):
            assert not boundaries
            stages, shared_global = result
            stages.retain_grad()
            shared_global.retain_grad()
            boundaries.update(stages=stages, shared_global=shared_global)

        handle = model.evidence_model.backbone.register_forward_hook(capture)
        model.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        torch.cuda.reset_peak_memory_stats()
        optimizer, scheduler, loss_fn = entry.entry.optimization(values, model, cfg)
        scaler = torch.amp.GradScaler('cuda', init_scale=256.0)
        with torch.autocast('cuda', dtype=torch.float16):
            result = model(batch, return_aux=True)
            loss, _components = entry.entry.loss_values(values, result, labels, batch['camera_ids'], loss_fn)
        assert bool(torch.isfinite(loss))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
                     for name,parameter in model.named_parameters() if parameter.requires_grad}
        assert all(torch.isfinite(value).all() for value in gradients.values() if value is not None)
        assert all(value.grad is not None for value in boundaries.values())
        # Copy retained boundary values only after the backward, outside callbacks.
        observed.append({'outputs': {name: result[key].detach().cpu() for name,key in
                            (('raw','raw_fused'),('fused','fused'),('global','shared_global'))},
            'loss': loss.detach().cpu(), 'heads': [(score.detach().cpu(), feature.detach().cpu())
                                                  for score,feature in result['heads']],
            'boundaries': {name: {'forward': value.detach().cpu(),
                                 'unscaled_gradient': value.grad.detach().cpu().float()/256.0,
                                 'metadata': metadata(value)} for name,value in boundaries.items()},
            'gradients': gradients, 'buffers': {name:value.detach().cpu().clone() for name,value in model.named_buffers()},
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
            'model_state': entry.runner._module_state_sha256(model),
            'bn_counts': {neck:int(getattr(model.signal,neck).num_batches_tracked) for neck,_ in model.head_names},
            'capture_hooks_removed': all(not model.signal.clip_vision_encoder.base.transformer.resblocks[index]._forward_hooks
                                         for index in model.evidence_model.backbone.layers),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        handle.remove()
        model.zero_grad(set_to_none=True)
        del result, loss, optimizer, scheduler, loss_fn, scaler, boundaries
        torch.cuda.empty_cache()
    a,b = observed
    assert set(a['gradients']) == set(b['gradients'])
    gradient_deltas = {}
    for name,x in a['gradients'].items():
        y = b['gradients'][name]
        row = {'reference_none':x is None,'candidate_none':y is None}
        if x is not None and y is not None:row.update(compare(x,y,1e-4))
        gradient_deltas[name] = row
    outputs = {name:compare(a['outputs'][name],b['outputs'][name],1e-5) for name in a['outputs']}
    outputs['loss'] = compare(a['loss'],b['loss'],1e-5)
    heads = [[compare(x,y,1e-5) for x,y in zip(left,right)] for left,right in zip(a['heads'],b['heads'])]
    boundary_deltas = {name:{'forward':compare(a['boundaries'][name]['forward'],b['boundaries'][name]['forward'],1e-5),
                            'incoming_unscaled_gradient':compare(a['boundaries'][name]['unscaled_gradient'],
                                                                  b['boundaries'][name]['unscaled_gradient'],1e-4),
                            'reference_metadata':a['boundaries'][name]['metadata'],
                            'candidate_metadata':b['boundaries'][name]['metadata']} for name in a['boundaries']}
    post_checks = {'buffers_exact':set(a['buffers'])==set(b['buffers']) and
                                  all(torch.equal(v,b['buffers'][n]) for n,v in a['buffers'].items()),
                   'cpu_rng_exact':torch.equal(a['cpu_rng'],b['cpu_rng']),
                   'cuda_rng_exact':torch.equal(a['cuda_rng'],b['cuda_rng']),
                   'state_exact':a['model_state']==b['model_state'],
                   'bn_one_batch_each':all(count==1 for arm in observed for count in arm['bn_counts'].values()),
                   'original_capture_hooks_removed':all(arm['capture_hooks_removed'] for arm in observed)}
    assert storage.packs and storage.unpacks
    metadata_differences = []
    for row in storage.unpacks:
        before = storage.packs[row['index']]['before']
        changed = [key for key in ('shape','stride','storage_offset','dtype','layout','device','contiguous')
                   if before[key] != row['restored'][key]]
        if changed:metadata_differences.append({'index':row['index'],'unpack_order':row['order'],'changed':changed})
    measured = {'schema':SCHEMA,'status':'MEASURED_BEFORE_FIXED_GATE','started_at':started_at,
        'measured_at':datetime.now().astimezone().isoformat(),'dataset':dataset,'variant':variant,
        'comparison':'original_vs_cpu_saved_instrumented','optimizer_updates':0,'forward_backwards':2,
        'witness_samples':32,'full_author_batch':64,'labels':labels.tolist(),'cameras':batch['camera_ids'].tolist(),
        'paths':list(raw[4][:32]),'images_sha256':{n:entry.clean.tensor_digest(v) for n,v in batch['images'].items()},
        'initial_cpu_rng':cpu_rng.tolist(),'initial_cuda_rng':cuda_rng.tolist(),
        'final_rng': [{'cpu':arm['cpu_rng'].tolist(),'cuda':arm['cuda_rng'].tolist()} for arm in observed],
        'reference_binding':reference_binding,'candidate_binding':candidate_binding,
        'torch_version':torch.__version__,'actual_builtin_source':inspect.getsource(torch.autograd.graph.save_on_cpu),
        'outputs':outputs,'heads':heads,'boundaries':boundary_deltas,'gradients':gradient_deltas,
        'post_checks':post_checks,'metadata_differences':metadata_differences,
        'packs':storage.packs,'unpacks':storage.unpacks,
        'memory':[{'allocated':arm['peak_allocated_bytes'],'reserved':arm['peak_reserved_bytes']} for arm in observed],
        'boundary':'One instrumented train-side diagnostic. Boundary metadata differences are observations, not unique causes; instrumentation can affect timing. No checkpoint/M0/formal/official score; a PASS does not promote sealed v4.'}
    (output_dir/'MEASURED.json').write_text(json.dumps(measured,indent=2)+'\n')
    assert all(row['fixed_gate_pass'] for row in outputs.values())
    assert all(row['fixed_gate_pass'] for head in heads for row in head)
    assert all(row['reference_none']==row['candidate_none'] and row.get('fixed_gate_pass',True)
               for row in gradient_deltas.values())
    assert all(post_checks.values())
    (output_dir/'COMPLETE.json').write_text(json.dumps({'schema':SCHEMA,'status':'INSTRUMENTED_FIXED_GATE_PASS',
        'completed_at':datetime.now().astimezone().isoformat(),
        'boundary':'Only this observed diagnostic passed; sealed v4 unchanged, no training promotion.'},indent=2)+'\n')
    print(output_dir,flush=True)


if __name__ == '__main__':
    main()

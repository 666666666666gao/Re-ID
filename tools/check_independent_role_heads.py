"""Initial behavior, loss ownership and actual full-batch head-copy checks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'modeling'))
from trifusion.global_task_role_heads import GlobalTaskRoleHeads
from trifusion.independent_role_heads import IndependentRoleHeads
from tools.check_global_task_role_heads import Core, task


def component_witness():
    rows = []
    torch.set_num_threads(1)
    for direct in (True, False):
        torch.manual_seed(42)
        core = Core(direct)
        control = GlobalTaskRoleHeads(deepcopy(core)).train()
        model = IndependentRoleHeads(core).train()
        a, b = model.state_dict(), control.state_dict()
        assert all(torch.equal(a[name], value) for name, value in b.items())
        assert all(name.startswith('role_heads.') for name in set(a) - set(b))
        for (neck, classifier), head in zip(model.head_names, model.role_heads):
            for label, name in (('neck', neck), ('classifier', classifier)):
                assert all(torch.equal(value, getattr(control.signal, name).state_dict()[key])
                           for key, value in head[label].state_dict().items())
        batch, labels = {'x': torch.randn(8, 16)}, torch.arange(8) % 4
        result, reference = model(batch, return_aux=True), control(batch, return_aux=True)
        for name in ('raw_fused', 'fused', 'shared_global'):
            assert torch.equal(result[name], reference[name]), name
        for name in ('heads', 'global_heads'):
            assert all(torch.equal(x, y) for left, right in zip(result[name], reference[name])
                       for x, y in zip(left, right)), name
        named = {name: p for name, p in model.named_parameters() if p.requires_grad}
        global_grad = torch.autograd.grad(task(result['global_heads'], labels), tuple(named.values()),
                                          retain_graph=True, allow_unused=True)
        role_grad = torch.autograd.grad(task(result['heads'], labels), tuple(named.values()), allow_unused=True)
        for (name, p), left, right in zip(named.items(), global_grad, role_grad):
            is_role = name.startswith('role_heads.') or '.role.' in name or name.endswith('.readout_gain')
            active, absent = (right, left) if is_role else (left, right)
            assert absent is None, name
            assert active is not None and torch.isfinite(active).all() and active.abs().max() > 0, name
        optimizer = torch.optim.SGD(tuple(named.values()), lr=0.001)
        initial = {name: p.detach().clone() for name, p in named.items() if name.startswith('role_heads.')}
        active = set()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            result = model(batch, return_aux=True)
            (task(result['global_heads'], labels) + task(result['heads'], labels)).backward()
            active.update(name for name in initial if torch.isfinite(named[name].grad).all()
                          and named[name].grad.abs().max() > 0)
            optimizer.step()
        assert active == set(initial)
        assert all(not torch.equal(named[name], value) for name, value in initial.items())
        assert all(int(getattr(model.signal, neck).num_batches_tracked) == 9 for neck, _ in model.head_names)
        assert all(int(head['neck'].num_batches_tracked) == 9 for head in model.role_heads)
        restored = IndependentRoleHeads(Core(direct))
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval(); restored.eval()
        with torch.inference_mode():
            assert torch.equal(model(batch), restored(batch))
        rows.append(dict(direct=direct,initial_outputs_and_logits_exact=True,
            global_role_loss_gradients_disjoint=True,all_new_head_parameters_active_and_changed=True,
            strict_component_reload_exact=True,extra_trainable_tensors=len(initial),
            extra_trainable_parameters=sum(named[name].numel() for name in initial)))
    return dict(status='CPU_HEAD_COPY_AND_OWNERSHIP_PASS',rows=rows,
                boundary='Synthetic CE witness, not full CLIP/Mamba, production M0 or retrieval evidence.')


def real_pair(campaign, dataset):
    from tools import run_independent_role_heads as entry
    from tools import queue_independent_role_heads as panel
    entry.configure(); panel.configure()
    inner = entry.inner
    controls = panel.previous.require_controls()
    old = next(row['initializer'] for row in controls['rows']
               if (row['dataset'], row['variant']) == (dataset, 'semantic'))
    args = argparse.Namespace(dataset=dataset,variant='semantic',recipe='semantic',mode='prepare',seed=42,epochs=50,
        protocol=panel.previous.previous.PROTOCOLS / f'{dataset}.json',
        signal_source=panel.previous.previous.SOURCE,clip_weight=panel.previous.previous.WEIGHTS / 'ViT-B-16.pt',
        initialization=campaign / 'initialization' / f'{dataset}_semantic.json',
        output_dir=ROOT / 'trained-model/independent_role_head_pair_unused',baseline_sha256=old['public_clip_sha256'])
    protocol = inner.runner.read_protocol(args.protocol, dataset)
    states, predictions, inputs = {}, {}, {}
    for label in ('independent_heads', 'raw_semantic_reference'):
        if label == 'independent_heads':
            model, cfg, binding = entry.build_core(args, protocol)
            assert binding == panel.base.expected_binding(campaign, dataset, 'semantic')
        else:
            inner.AuthorHeadEvidence = GlobalTaskRoleHeads
            model, cfg, binding = entry.original_build_core(args, protocol)
            assert binding['initial_model_state_sha256'] == old['initial_model_state_sha256']
        states[label] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        raw = next(iter(inner.original_train_loader(args, protocol, cfg)))
        batch, labels = inner.runner._training_batch(raw)
        assert len(labels) == cfg.SOLVER.IMS_PER_BATCH
        inputs[label] = dict(images={name: inner.clean.tensor_digest(t) for name, t in batch['images'].items()},
                             labels=labels.tolist(),cameras=batch['camera_ids'].tolist(),paths=list(raw[4]))
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            result = model(batch, return_aux=True)
        predictions[label] = {name: result[name].cpu() for name in ('raw_fused', 'fused', 'shared_global')}
        for name in ('heads', 'global_heads'):
            predictions[label][name] = [(score.cpu(), feature.cpu()) for score, feature in result[name]]
        del model, result, batch, raw
        torch.cuda.empty_cache()
    assert inputs['independent_heads'] == inputs['raw_semantic_reference']
    a, b = states['independent_heads'], states['raw_semantic_reference']
    assert set(b).issubset(a)
    assert all(name.startswith('role_heads.') for name in set(a) - set(b))
    assert all(torch.equal(a[name], value) for name, value in b.items())
    for name in ('raw_fused', 'fused', 'shared_global'):
        assert torch.equal(predictions['independent_heads'][name], predictions['raw_semantic_reference'][name]), name
    for name in ('heads', 'global_heads'):
        assert all(torch.equal(x, y) for left, right in zip(predictions['independent_heads'][name],
                    predictions['raw_semantic_reference'][name]) for x, y in zip(left, right)), name
    return dict(status='REAL_FULL_BATCH_INITIAL_HEAD_COPY_PASS',dataset=dataset,
        common_state_exact=True,reference_initial_state_matches_sealed_control=True,
        raw_deployment_global_and_all_head_predictions_exact=True,extra_state_keys=sorted(set(a)-set(b)),
        batch=inputs['independent_heads'],boundary='Fresh initialization only; no baseline retraining, scoring or backward-parity claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--dataset', choices=('RGBNT201', 'MSVR310', 'RGBNT100'))
    args = parser.parse_args()
    assert not args.output.exists()
    if args.campaign is None:
        assert args.dataset is None
        value = component_witness()
    else:
        assert args.dataset is not None
        value = real_pair(args.campaign.resolve(), args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'],output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()

"""Component activity and full author-batch zero-exit witnesses."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_region_reconstruction as entry
from trifusion.region_evidence_reconstruction import RegionCandidateReconstruction
from trifusion.role_input_detach import DetachedSemanticTriFusion


def component_witness():
    torch.set_num_threads(1)
    rows = []
    for mode in ('patch', 'mean'):
        torch.manual_seed(42)
        module = RegionCandidateReconstruction(mode)
        patches = torch.randn(2, 3, 128, 128)
        target = torch.randn_like(patches)
        assert len(list(module.parameters())) == 15
        assert sum(p.numel() for p in module.parameters()) == 105232
        assert torch.equal(module(patches), patches)
        initial = {n: p.detach().clone() for n, p in module.named_parameters()}
        active = set()
        optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(module(patches), target).backward()
            for name, parameter in module.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                if parameter.grad.abs().max() > 0:
                    active.add(name)
            optimizer.step()
        assert active == set(initial)
        assert all(not torch.equal(p, initial[n]) for n, p in module.named_parameters())
        restored = RegionCandidateReconstruction(mode)
        restored.load_state_dict(module.state_dict(), strict=True)
        assert torch.equal(restored(patches), module(patches))
        # Same active values, differing only in query routing. Mean messages
        # broadcast; patch messages vary. Compare messages before residual add.
        paired = RegionCandidateReconstruction('mean' if mode == 'patch' else 'patch')
        paired.load_state_dict(module.state_dict(), strict=True)
        q = module.query_norm(patches if mode == 'patch' else patches.mean(-2, keepdim=True).expand_as(patches))
        assert (q[:, :, 1:] - q[:, :, :1]).abs().max() > 0 if mode == 'patch' else torch.equal(q[:, :, 1:], q[:, :, :1].expand_as(q[:, :, 1:]))
        assert not torch.equal(paired(patches), module(patches))
        rows.append(dict(query_mode=mode, parameters=105232, tensors=15,
                         zero_initial_exit=True, all_parameters_active_and_changed=True,
                         strict_component_reload_exact=True, routing_control_distinct=True))
    return dict(status='CPU_SYNTHETIC_COMPONENT_PASS', rows=rows,
                boundary='Synthetic activity only; each endpoint still requires production-data M0.')


def real_pair(campaign, dataset):
    from tools import queue_region_reconstruction as panel
    panel.configure()
    controls = panel.previous.require_controls()
    old = next(r['initializer'] for r in controls['rows']
               if (r['dataset'], r['variant']) == (dataset, 'semantic'))
    args = argparse.Namespace(dataset=dataset, variant='semantic', recipe='semantic',
        mode='prepare', seed=42, epochs=50, protocol=panel.research.PROTOCOLS/f'{dataset}.json',
        signal_source=panel.research.SOURCE, clip_weight=panel.research.WEIGHTS/'ViT-B-16.pt',
        initialization=campaign/'initialization'/f'{dataset}_semantic.json',
        output_dir=ROOT/'trained-model/region_reconstruction_pair_unused',
        baseline_sha256=old['public_clip_sha256'])
    protocol = entry.inner.runner.read_protocol(args.protocol, dataset)
    states, predictions, inputs = {}, {}, {}
    for label in ('patch', 'mean', 'raw_semantic_reference'):
        if label == 'raw_semantic_reference':
            entry.inner.RawFeatureSemanticTriFusion = DetachedSemanticTriFusion
            entry.previous.configure()
            args.variant = args.recipe = 'semantic'
            model, cfg, binding = entry.original_build_core(args, protocol)
            assert binding['initial_model_state_sha256'] == old['initial_model_state_sha256']
        else:
            entry.configure()
            args.variant = args.recipe = 'semantic' if label == 'patch' else 'native'
            model, cfg, binding = entry.build_core(args, protocol)
            assert binding == panel.base.expected_binding(campaign, dataset, args.variant)
        states[label] = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        raw = next(iter(entry.inner.original_train_loader(args, protocol, cfg)))
        batch, labels = entry.inner.runner._training_batch(raw)
        assert len(labels) == cfg.SOLVER.IMS_PER_BATCH
        inputs[label] = dict(images={n: entry.inner.clean.tensor_digest(t) for n, t in batch['images'].items()},
                            labels=labels.tolist(), cameras=batch['camera_ids'].tolist(), paths=list(raw[4]))
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            result = model(batch, return_aux=True)
        predictions[label] = {n: result[n].cpu() for n in ('raw_fused', 'fused', 'shared_global')}
        predictions[label]['heads'] = [(score.cpu(), feature.cpu()) for score, feature in result['heads']]
        del model, result, batch, raw
        torch.cuda.empty_cache()
    assert inputs['patch'] == inputs['mean'] == inputs['raw_semantic_reference']
    reference = states['raw_semantic_reference']
    assert all(torch.equal(states[label][n], v) for label in ('patch', 'mean') for n, v in reference.items())
    assert set(states['patch']) - set(reference) == {n for n in states['patch'] if '.reconstruction.' in n}
    assert states['patch'].keys() == states['mean'].keys()
    assert all(torch.equal(v, states['mean'][n]) for n, v in states['patch'].items())
    for label in ('patch', 'mean'):
        for name in ('raw_fused', 'fused', 'shared_global'):
            assert torch.equal(predictions[label][name], predictions['raw_semantic_reference'][name]), name
        assert all(torch.equal(x, y) for left, right in zip(predictions[label]['heads'], predictions['raw_semantic_reference']['heads'])
                   for x, y in zip(left, right))
    return dict(status='REAL_FULL_BATCH_INITIAL_PAIR_PASS', dataset=dataset,
                original_semantic_state_and_batch_exact=True, new_arm_state_exact=True,
                zero_exit_raw_fused_global_and_head_predictions_exact=True, batch=inputs['patch'],
                boundary='Initial preservation only; no guarantee after training, no historical parity repair.')


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
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(status=value['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()

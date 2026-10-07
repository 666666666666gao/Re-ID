"""Assignment math/activity plus matched full author-batch initialization."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_selective_role_transport as entry
from trifusion.selective_role_transport import SlotMessageTransport, partial_log_assignment, message_weights
from trifusion.role_input_detach import DetachedSemanticTriFusion


def component_witness():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    strong = torch.full((2, 16, 16), -8.0)
    strong.diagonal(dim1=-2, dim2=-1).fill_(8.0)
    weak = torch.full_like(strong, -8.0)
    mixed = torch.randn_like(strong)
    marginals = torch.tensor([1.0] * 16 + [16.0])
    assignments, rows = {}, []
    for name, scores in (('diagonal', strong), ('unmatched', weak), ('mixed', mixed)):
        log_p = partial_log_assignment(scores)
        p = log_p.exp()
        assert torch.isfinite(log_p).all() and torch.isfinite(p).all()
        residual = max(float((p.sum(-1) - marginals).abs().max()),
                       float((p.sum(-2) - marginals).abs().max()))
        assert residual <= 1e-3
        assignments[name] = p
        for real in (log_p[..., :16, :16], log_p[..., :16, :16].transpose(-1, -2)):
            candidate, mass, conditional = message_weights(real, 'slot_mass')
            control, other_mass, other_conditional = message_weights(real, 'uniform_mass')
            assert mass.amax() <= 1 + 1e-3
            assert torch.equal(mass, other_mass) and torch.equal(conditional, other_conditional)
            assert torch.allclose(candidate.sum((-2, -1)), control.sum((-2, -1)), atol=1e-5, rtol=1e-5)
            assert torch.allclose(control.sum(-1), mass.mean(-1, keepdim=True).expand_as(mass), atol=1e-5, rtol=1e-5)
            if name == 'mixed':
                assert not torch.allclose(candidate, control, atol=1e-5, rtol=1e-5)
        rows.append(dict(condition=name, maximum_all_marginal_residual=residual,
                         total_mass=float(p[..., :16, :16].sum((-2,-1)).mean())))
    assert assignments['unmatched'][..., :16, :16].sum() < assignments['diagonal'][..., :16, :16].sum()
    activity = []
    for mode in ('slot_mass', 'uniform_mass'):
        torch.manual_seed(42)
        module = SlotMessageTransport(mode)
        semantic, private, target = [torch.randn(2, 3, 16, 128) for _ in range(3)]
        assert len(list(module.parameters())) == 2 and sum(p.numel() for p in module.parameters()) == 32768
        assert torch.equal(module(semantic, private), private)
        initial = {n:p.detach().clone() for n,p in module.named_parameters()}
        optimizer = torch.optim.Adam(module.parameters(), lr=0.001)
        active = set()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(module(semantic, private), target).backward()
            for name, parameter in module.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                if parameter.grad.abs().max() > 0:
                    active.add(name)
            optimizer.step()
        assert active == set(initial)
        assert all(not torch.equal(p, initial[n]) for n,p in module.named_parameters())
        restored = SlotMessageTransport(mode)
        restored.load_state_dict(module.state_dict(), strict=True)
        assert torch.equal(restored(semantic, private), module(semantic, private))
        activity.append(dict(mass_mode=mode, all_two_tensors_active_and_changed=True,
                             zero_initial_exit=True, strict_component_reload_exact=True))
    return dict(status='CPU_SYNTHETIC_COMPONENT_PASS', assignment_rows=rows, activity_rows=activity,
                boundary='Matrix-mass control at same P, not exact vector-energy matching or geometric correspondence. Synthetic only; actual full-batch M0 remains required.')


def real_pair(campaign, dataset):
    from tools import queue_selective_role_transport as panel
    panel.configure()
    controls = panel.previous.require_controls()
    old = next(r['initializer'] for r in controls['rows']
               if (r['dataset'], r['variant']) == (dataset, 'semantic'))
    args = argparse.Namespace(dataset=dataset, variant='semantic', recipe='semantic',
        mode='prepare', seed=42, epochs=50, protocol=panel.research.PROTOCOLS/f'{dataset}.json',
        signal_source=panel.research.SOURCE, clip_weight=panel.research.WEIGHTS/'ViT-B-16.pt',
        initialization=campaign/'initialization'/f'{dataset}_semantic.json',
        output_dir=ROOT/'trained-model/selective_transport_pair_unused',
        baseline_sha256=old['public_clip_sha256'])
    protocol = entry.inner.runner.read_protocol(args.protocol, dataset)
    states, predictions, inputs = {}, {}, {}
    for label in ('slot_mass', 'uniform_mass', 'raw_semantic_reference'):
        if label == 'raw_semantic_reference':
            entry.previous.configure()
            args.variant = args.recipe = 'semantic'
            model, cfg, binding = entry.original_build_core(args, protocol)
            assert binding['initial_model_state_sha256'] == old['initial_model_state_sha256']
        else:
            entry.configure()
            args.variant = args.recipe = 'semantic' if label == 'slot_mass' else 'native'
            model, cfg, binding = entry.build_core(args, protocol)
            assert binding == panel.base.expected_binding(campaign, dataset, args.variant)
        states[label] = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        raw = next(iter(entry.inner.original_train_loader(args, protocol, cfg)))
        batch, labels = entry.inner.runner._training_batch(raw)
        assert len(labels) == cfg.SOLVER.IMS_PER_BATCH
        inputs[label] = dict(images={n:entry.inner.clean.tensor_digest(t) for n,t in batch['images'].items()},
                            labels=labels.tolist(), cameras=batch['camera_ids'].tolist(), paths=list(raw[4]))
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            result = model(batch, return_aux=True)
        predictions[label] = {n:result[n].cpu() for n in ('raw_fused','fused','shared_global')}
        predictions[label]['heads'] = [(score.cpu(),feature.cpu()) for score,feature in result['heads']]
        del model,result,batch,raw
        torch.cuda.empty_cache()
    assert inputs['slot_mass'] == inputs['uniform_mass'] == inputs['raw_semantic_reference']
    reference = states['raw_semantic_reference']
    assert all(torch.equal(states[label][n],v) for label in ('slot_mass','uniform_mass') for n,v in reference.items())
    assert set(states['slot_mass']) - set(reference) == {n for n in states['slot_mass'] if '.transport.' in n}
    assert states['slot_mass'].keys() == states['uniform_mass'].keys()
    assert all(torch.equal(v,states['uniform_mass'][n]) for n,v in states['slot_mass'].items())
    for name in ('raw_fused','fused','shared_global'):
        assert torch.equal(predictions['slot_mass'][name],predictions['uniform_mass'][name]),name
    assert torch.equal(predictions['slot_mass']['shared_global'],predictions['raw_semantic_reference']['shared_global'])
    assert all(torch.equal(x,y) for left,right in zip(predictions['slot_mass']['heads'],predictions['uniform_mass']['heads'])
               for x,y in zip(left,right))
    return dict(status='REAL_FULL_BATCH_INITIAL_PAIR_PASS',dataset=dataset,
                original_semantic_parameter_state_and_batch_exact=True,new_arm_state_exact=True,
                initial_new_arm_raw_l2_global_heads_exact=True,original_global_exact=True,batch=inputs['slot_mass'],
                boundary='Private 3x16 Mamba differs from old48, so old fused equality is not required. Initial pair match does not guarantee training benefit; no historical parity repair.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--dataset',choices=('RGBNT201','MSVR310','RGBNT100'))
    args=parser.parse_args()
    assert not args.output.exists()
    if args.campaign is None:
        assert args.dataset is None
        value=component_witness()
    else:
        assert args.dataset is not None
        value=real_pair(args.campaign.resolve(),args.dataset)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],output=str(args.output))),flush=True)


if __name__=='__main__':
    main()

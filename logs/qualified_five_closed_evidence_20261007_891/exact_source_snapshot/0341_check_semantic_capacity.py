"""Real full-batch initialization and active reader CPU witnesses."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_semantic_capacity as entry
from trifusion.role_input_detach import DetachedNativeTriFusion
from trifusion.semantic_capacity_evidence import SemanticCapacityReader


def component_witness():
    rows = []
    torch.set_num_threads(1)
    for grid in ((16, 8), (8, 16)):
        torch.manual_seed(42)
        reader = SemanticCapacityReader(grid)
        assert sum(p.numel() for p in reader.parameters()) == 159096
        assert len(list(reader.parameters())) == 14
        patches, queries = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 16, 128)
        assert torch.count_nonzero(reader(patches, queries)) == 0
        target = torch.randn(2, 3, 16, 128)
        optimizer = torch.optim.Adam(reader.parameters(), lr=0.001)
        active = set()
        initial = {name: p.detach().clone() for name, p in reader.named_parameters()}
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(reader(patches, queries), target).backward()
            for name, p in reader.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all()
                if p.grad.abs().max() > 0:
                    active.add(name)
            optimizer.step()
        assert active == set(initial)
        assert all(not torch.equal(p, initial[name]) for name, p in reader.named_parameters())
        restored = SemanticCapacityReader(grid)
        restored.load_state_dict(reader.state_dict(), strict=True)
        assert torch.equal(restored(patches, queries), reader(patches, queries))
        rows.append(dict(grid=grid,parameters=159096,tensors=14,zero_initial_exit=True,
                         all_gradients_finite_and_cumulatively_active=True,all_parameters_changed=True,
                         component_reload_exact=True))
    return dict(status='CPU_SYNTHETIC_COMPONENT_PASS', rows=rows,
                boundary='No retrieval evidence or substitute for actual author-batch M0.')


def real_pair(campaign, dataset):
    entry.configure()
    inner = entry.inner
    from tools import queue_semantic_capacity as panel
    panel.configure()
    controls = json.loads(panel.CONTROLS.read_text())
    old = next(r['initializer'] for r in controls['rows']
               if (r['dataset'], r['variant']) == (dataset, 'native'))
    args = argparse.Namespace(dataset=dataset, variant='native', recipe='native',mode='prepare',
        seed=42,epochs=50,protocol=panel.previous.previous.PROTOCOLS/f'{dataset}.json',
        signal_source=panel.previous.previous.SOURCE,clip_weight=panel.previous.previous.WEIGHTS/'ViT-B-16.pt',
        initialization=campaign/'initialization'/f'{dataset}_native.json',
        output_dir=ROOT/'trained-model/semantic_capacity_pair_unused',
        baseline_sha256=old['public_clip_sha256'])
    protocol = inner.runner.read_protocol(args.protocol, dataset)
    states, predictions, inputs = {}, {}, {}
    for label in ('capacity', 'raw_native_reference'):
        if label == 'capacity':
            model, cfg, binding = entry.build_core(args, protocol)
            assert binding == panel.base.expected_binding(campaign, dataset, 'native')
            assert binding['trainable_parameters'] == old['trainable_parameters'] - 200
            assert binding['trainable_parameter_tensors'] == old['trainable_parameter_tensors']
        else:
            inner.IndependentNativeTriFusion = DetachedNativeTriFusion
            model, cfg, binding = entry.original_build_core(args, protocol)
            assert binding['initial_model_state_sha256'] == old['initial_model_state_sha256']
        states[label] = {name: value.detach().cpu().clone() for name,value in model.state_dict().items()}
        raw = next(iter(inner.original_train_loader(args, protocol, cfg)))
        batch, labels = inner.runner._training_batch(raw)
        assert len(labels) == cfg.SOLVER.IMS_PER_BATCH
        inputs[label] = dict(images={name:inner.clean.tensor_digest(t) for name,t in batch['images'].items()},
                            labels=labels.tolist(),cameras=batch['camera_ids'].tolist(),paths=list(raw[4]))
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            result=model(batch,return_aux=True)
        predictions[label] = {name:result[name].cpu() for name in ('raw_fused','fused','shared_global')}
        predictions[label]['heads']=[(score.cpu(),feature.cpu()) for score,feature in result['heads']]
        del model,result,batch,raw
        torch.cuda.empty_cache()
    assert inputs['capacity'] == inputs['raw_native_reference']
    a,b=states['capacity'],states['raw_native_reference']
    assert set(a)==set(b)
    changed=[name for name in a if not torch.equal(a[name],b[name])]
    assert len(changed)==6 and all('.detail_reader.stem.' in name for name in changed)
    for name in ('raw_fused','fused','shared_global'):
        assert torch.equal(predictions['capacity'][name],predictions['raw_native_reference'][name]),name
    assert all(torch.equal(x,y) for left,right in zip(predictions['capacity']['heads'],predictions['raw_native_reference']['heads'])
               for x,y in zip(left,right))
    return dict(status='REAL_FULL_BATCH_INITIAL_PAIR_PASS',dataset=dataset,
        capacity_source='semantic_capacity',reference='sealed_raw_native_initialization_only',
        only_changed_initial_tensors=changed,reference_initial_state_matches_sealed_control=True,
        shared_and_nonstem_state_exact=True,zero_exit_raw_fused_global_and_head_predictions_exact=True,
        batch=inputs['capacity'],
        boundary='One new full-batch initialization witness, no baseline retraining or official scoring. Historical parity failures are unchanged.')


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

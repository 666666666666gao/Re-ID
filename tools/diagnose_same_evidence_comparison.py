"""Fixed raw-role bests: compare existing residual and final region evidence, no fitting."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnose_global_task_role_best as existing

base = existing.diagnosis
SCHEMA = 'trifusion-same-evidence-comparison-diagnosis-v1'
DATASETS = ('RGBNT201', 'MSVR310', 'RGBNT100')


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def require_inputs(seal):
    assert seal['schema'] == SCHEMA
    for name, pin in seal['source_sha256'].items():
        assert base.panel.base.sha(ROOT/name) == pin, name
    for name, pin in seal['artifact_sha256'].items():
        assert base.panel.base.sha(Path(name)) == pin, name


def extract(model, protocol, split):
    cache = {key: [] for key in ('shared_global', 'correction', 'fused', 'regions')}
    captured = []

    def hook(_module, _inputs, evidence):
        assert not captured
        # These are the exact three final region tensors consumed by read_evidence.
        captured.append(torch.stack([getattr(evidence, name).detach().float().cpu()
                                    for name in ('cnn', 'transformer', 'mamba')], dim=1))

    handle = model.evidence_model.roles.register_forward_hook(hook)
    with torch.inference_mode():
        records = base.entry.runner.records_for(protocol, split)
        for raw in base.entry.runner.loader_for(protocol, records, training=False, method='PLAIN_V8'):
            batch = base.entry.runner._eval_batch(raw, protocol['dataset'])
            output = model.evidence_model.forward_features(batch)
            assert len(captured) == 1
            regions = captured.pop()
            assert regions.shape[1:] == (3, 3, 16, 128)
            cache['regions'].append(regions)
            for key in ('shared_global', 'correction', 'fused'):
                cache[key].append(output[key].detach().float().cpu())
    handle.remove()
    result = {key: torch.cat(rows) for key, rows in cache.items()}
    assert not captured
    assert all(value.shape[0] == protocol['counts'][split] and torch.isfinite(value).all()
               for value in result.values())
    assert all(result[key].shape[1:] == (1536,) for key in ('shared_global', 'correction', 'fused'))
    return result


def region_similarity_block(a, b):
    pair = torch.einsum('akpc,bkrc->abkpr', a, b)
    same_slot = pair.diagonal(dim1=-2, dim2=-1).mean(dim=(-1, -2))
    match = 0.5*(pair.max(dim=-1).values.mean(dim=(-1, -2)) +
                 pair.max(dim=-2).values.mean(dim=(-1, -2)))
    return same_slot, match


def local_scores(query, gallery):
    # No physical visibility annotations are inferred. All16 existing output slots
    # participate, separately for each role and modality; no cross-space matching.
    q = query.reshape(len(query), 9, 16, 128)
    g = gallery.reshape(len(gallery), 9, 16, 128)
    assert bool((q.norm(dim=-1) > 0).all()) and bool((g.norm(dim=-1) > 0).all())
    q, g = F.normalize(q, dim=-1), F.normalize(g, dim=-1)
    aligned = torch.empty(len(q), len(g))
    symmetric = torch.empty_like(aligned)
    started = time.perf_counter()
    # Chunk sizes bound workspace only; they do not select candidates or tokens.
    for first in range(0, len(q), 16):
        a = q[first:first+16].to('cuda:0')
        for second in range(0, len(g), 128):
            b = g[second:second+128].to('cuda:0')
            same_slot, match = region_similarity_block(a, b)
            aligned[first:first+len(a), second:second+len(b)] = same_slot.cpu()
            symmetric[first:first+len(a), second:second+len(b)] = match.cpu()
    assert torch.isfinite(aligned).all() and torch.isfinite(symmetric).all()
    assert float(symmetric.min()) >= -1.00001 and float(symmetric.max()) <= 1.00001
    return aligned, symmetric, time.perf_counter()-started


def damage(control, candidate, ids):
    result = base.paired(control, candidate, ids)
    a, b = (np.asarray(x['first_match_rank']) == 1 for x in (control, candidate))
    delta = (np.asarray(candidate['average_precision']) - np.asarray(control['average_precision'])) * 100
    result['rank1_groups'] = {name: dict(queries=int(mask.sum()),
        contribution_to_total_mAP_points=float(delta[mask].sum()/len(delta)))
        for name, mask in (('repair', ~a & b), ('new_error', a & ~b),
                           ('kept_correct', a & b), ('kept_wrong', ~a & ~b))}
    result['query_delta_ap_percentiles'] = np.percentile(delta, [0, 5, 25, 50, 75, 95, 100]).tolist()
    result['worst_5_percent_mean_delta_ap_points'] = float(np.sort(delta)[:int(np.ceil(len(delta)*0.05))].mean())
    return result


def diagnose(args, seal):
    started = time.perf_counter()
    require_inputs(seal)
    row = next(r for r in seal['rows'] if r['dataset'] == args.dataset)
    original = Path(row['run_dir'])
    output = args.output_dir/args.dataset
    output.mkdir()
    values = argparse.Namespace(dataset=args.dataset, variant='semantic', recipe='semantic', mode='evaluate',
        protocol=base.panel.PROTOCOLS/f'{args.dataset}.json', signal_source=base.panel.SOURCE,
        clip_weight=base.panel.WEIGHTS/'ViT-B-16.pt', initialization=args.campaign/'initialization'/f'{args.dataset}_semantic.json',
        output_dir=original, seed=42, epochs=50)
    values.baseline_sha256 = base.entry.runner.sha256(values.clip_weight)
    protocol = base.entry.runner.read_protocol(values.protocol, args.dataset)
    existing.intervention.configure()
    model, _cfg, binding = base.foundation.build(values, protocol)
    assert binding == row['initializer']
    payload = base.foundation.load(original/'best_map.pth', model, values)
    assert payload['epoch'] == row['best_epoch']
    model.eval()
    before = base.entry.runner._module_state_sha256(model)
    assert all(p.grad is None for p in model.parameters())
    gain = float(model.evidence_model.readout_gain.detach().cpu())
    query, gallery = [extract(model, protocol, split) for split in ('query', 'gallery')]
    after = base.entry.runner._module_state_sha256(model)
    assert before == after and all(p.grad is None for p in model.parameters())
    for features in (query, gallery):
        assert torch.allclose(features['fused'], F.normalize(features['shared_global']+gain*features['correction'],dim=1),atol=1e-6,rtol=1e-6)
        assert bool((features['shared_global'].norm(dim=1)>0).all()) and bool((features['correction'].norm(dim=1)>0).all())
    for split,features in (('query',query),('gallery',gallery)):
        torch.save(features,output/f'{split}_evidence.pt')
    data = base.metadata(protocol)
    old = torch.load(original/'official_distances.pt',map_location='cpu',weights_only=False)
    assert all(np.array_equal(data[key],old[key]) for key in data)
    # The pinned MSVR author scorer writes re.txt relative to its working directory.
    os.chdir(output.resolve())
    scores = {}
    for name, key in (('global','shared_global'),('additive','fused'),('residual','correction')):
        distance = base.entry.runner.distance_matrix(F.normalize(query[key],dim=1),F.normalize(gallery[key],dim=1))
        scores[name] = base.score(distance,data,args.dataset,values.signal_source)
        if name == 'additive':
            assert all(abs(scores[name]['metrics'][key]-value)<1e-5 for key,value in row['metrics'].items())
            original_distance_difference = float((distance-old['fused']).abs().max())
    qg,qc,gg,gc = query['shared_global'],query['correction'],gallery['shared_global'],gallery['correction']
    qnorm = (qg.square().sum(1)+gain**2*qc.square().sum(1)).sqrt()
    gnorm = (gg.square().sum(1)+gain**2*gc.square().sum(1)).sqrt()
    global_dot = qg @ gg.T
    local_dot = gain**2*(qc @ gc.T)
    denominator = qnorm[:,None]*gnorm[None,:]
    block = (global_dot+local_dot)/denominator
    scores['same_residual_block'] = base.score(2-2*block,data,args.dataset,values.signal_source)
    aligned,symmetric,comparison_seconds = local_scores(query['regions'],gallery['regions'])
    scores['aligned_regions_only'] = base.score(2-2*aligned,data,args.dataset,values.signal_source)
    scores['symmetric_regions_only'] = base.score(2-2*symmetric,data,args.dataset,values.signal_source)
    weight = gain**2*qc.norm(dim=1)[:,None]*gc.norm(dim=1)[None,:]
    regional_block = (global_dot+weight*symmetric)/denominator
    scores['fixed_energy_region_block'] = base.score(2-2*regional_block,data,args.dataset,values.signal_source)
    dispersion = {}
    for split, features in (('query',query),('gallery',gallery)):
        region = F.normalize(features['regions'].reshape(len(features['regions']),9,16,128),dim=-1)
        similarity = region @ region.transpose(-1,-2)
        off_diagonal = similarity[:,:,~torch.eye(16,dtype=torch.bool)].flatten()
        dispersion[split] = dict(mean_off_diagonal_cosine=float(off_diagonal.mean()),
                                min_off_diagonal_cosine=float(off_diagonal.min()),
                                max_off_diagonal_cosine=float(off_diagonal.max()))
    comparisons = {name:damage(scores['global'],score,data['query_ids']) for name,score in scores.items() if name!='global'}
    comparisons['additive_to_same_residual_block'] = damage(scores['additive'],scores['same_residual_block'],data['query_ids'])
    report = dict(schema=SCHEMA,status='COMPLETE',completed_at=stamp(),dataset=args.dataset,
        selected_epoch=row['best_epoch'],input_checkpoint_sha256=row['checkpoint_sha256'],gain=gain,
        optimizer_updates=0,model_state_before_sha256=before,model_state_after_sha256=after,
        original_fused_distance_max_abs_difference=original_distance_difference,scores=scores,comparisons=comparisons,
        region_dispersion=dispersion,local_comparison_seconds=comparison_seconds,elapsed_seconds=time.perf_counter()-started,
        cached_artifacts={name:dict(bytes=(output/name).stat().st_size,sha256=base.panel.base.sha(output/name))
                          for name in ('query_evidence.pt','gallery_evidence.pt')},
        boundary='Fixed originally selected raw-semantic bests, no training/fitting/selection or new coefficients. Exact same-residual block changes cross terms and norm. Final regions are actual C/T/M readout inputs, not keys/part GT. Symmetric within-role/modality MaxSim is many-to-one, no inferred visibility/rejection. Fixed-energy regional block replaces residual cosine using existing gain/norm factors. Full legal gallery; pairwise diagnostics/resource changes, not1536-vector main results or proof that compression causes loss. Official sets are consumed development benchmarks; identity aggregates are not training seeds.')
    require_inputs(seal)
    write(output/'DIAGNOSIS.json',report)
    print(json.dumps(dict(dataset=args.dataset,status='COMPLETE',metrics={k:v['metrics'] for k,v in scores.items()},
                         seconds=report['elapsed_seconds'])),flush=True)


def coordinate(args, seal):
    assert not args.output_dir.exists()
    require_inputs(seal)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    assert shutil.disk_usage(ROOT).free >= seal['storage_required_bytes']
    used = subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(line.split(',')[1])<500 for line in used.splitlines())
    args.output_dir.mkdir()
    state=dict(schema=SCHEMA,status='RUNNING',pid=os.getpid(),started_at=stamp(),jobs=[],optimizer_updates=0,physical_gpus=[0,1])
    write(args.output_dir/'campaign.json',state)
    for dataset in DATASETS:
        row=dict(dataset=dataset,status='RUNNING',started_at=stamp())
        state['jobs'].append(row)
        with (args.output_dir/f'{dataset}.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)), '--campaign',str(args.campaign),
                '--seal',str(args.seal),'--output-dir',str(args.output_dir),'--dataset',dataset],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=process.pid
            write(args.output_dir/'campaign.json',state)
            code=process.wait()
        row.update(status='FAILED' if code else 'COMPLETE',exit_code=code,completed_at=stamp())
        write(args.output_dir/'campaign.json',state)
        if code:
            state.update(status='FAILED',completed_at=stamp())
            write(args.output_dir/'campaign.json',state)
            return code
    state.update(status='COMPLETE',completed_at=stamp())
    write(args.output_dir/'campaign.json',state)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','seal','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dataset',choices=DATASETS)
    args=parser.parse_args()
    seal=json.loads(args.seal.read_text())
    if args.dataset is None:
        return coordinate(args,seal)
    diagnose(args,seal)
    return 0


if __name__=='__main__':
    raise SystemExit(main())

"""One fixed-best g/c/f pass per dataset; retain distances and sample statistics."""
import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnose_native_research_best as common
from tools import run_independent_role_heads as entry

SCHEMA='trifusion-independent-role-heads-fixed-best-diagnosis-v1'
STORAGE_BYTES=2*1024**3+256*1024**2
entry.runner=common.entry.runner
common.entry=entry
common.SCHEMA=SCHEMA
common.VARIANTS=('semantic',)
common.__file__=str(Path(__file__).resolve())


def diagnose(args,seal):
    started=time.perf_counter()
    common.require_inputs(seal)
    row=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==(args.dataset,'semantic'))
    original=Path(row['run_dir'])
    output=args.output_dir/f'{args.dataset}_semantic'
    output.mkdir()
    entry.configure()
    values=argparse.Namespace(dataset=args.dataset,variant='semantic',recipe='semantic',
        protocol=common.panel.PROTOCOLS/f'{args.dataset}.json',signal_source=common.panel.SOURCE,
        clip_weight=common.panel.WEIGHTS/'ViT-B-16.pt',
        initialization=args.campaign/'initialization'/f'{args.dataset}_semantic.json',
        output_dir=original,seed=42,epochs=50)
    values.baseline_sha256=entry.runner.sha256(values.clip_weight)
    protocol=entry.runner.read_protocol(values.protocol,args.dataset)
    model,_cfg,binding=common.foundation.build(values,protocol)
    assert binding==row['initializer']
    payload=common.foundation.load(original/'best_map.pth',model,values)
    assert payload['epoch']==row['best_epoch']
    assert all(abs(payload['metrics'][k]-v)<1e-5 for k,v in row['metrics'].items())
    model.eval()
    before=entry.runner._module_state_sha256(model)
    assert all(p.grad is None for p in model.parameters())
    gain=float(model.evidence_model.readout_gain.detach().cpu())
    query,_=common.extract(model,protocol,'query','semantic')
    gallery,_=common.extract(model,protocol,'gallery','semantic')
    after=entry.runner._module_state_sha256(model)
    assert before==after and all(p.grad is None for p in model.parameters())
    data=common.metadata(protocol)
    old=torch.load(original/'official_distances.pt',map_location='cpu',weights_only=False)
    global_row=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==(args.dataset,'global_only'))
    independent=torch.load(Path(global_row['run_dir'])/'official_distances.pt',map_location='cpu',weights_only=False)
    assert all(np.array_equal(data[k],old[k]) and np.array_equal(data[k],independent[k]) for k in data)
    distances,scores={},{}
    for name,key in (('fused','fused'),('global','shared_global'),('correction','correction')):
        q=query[key] if name=='fused' else F.normalize(query[key],dim=1)
        g=gallery[key] if name=='fused' else F.normalize(gallery[key],dim=1)
        distances[name]=entry.runner.distance_matrix(q,g)
        scores[name]=common.score(distances[name],data,args.dataset,values.signal_source)
    scores['independent_global_only']=common.score(independent['fused'],data,args.dataset,values.signal_source)
    assert all(abs(scores['fused']['metrics'][k]-v)<1e-5 for k,v in row['metrics'].items())
    assert all(abs(scores['independent_global_only']['metrics'][k]-v)<1e-5 for k,v in global_row['metrics'].items())
    diagnostic={}
    for split,features in (('query',query),('gallery',gallery)):
        g,c,h,f=(features[k] for k in common.KEYS)
        assert torch.allclose(h,g+gain*c,atol=1e-6,rtol=1e-6)
        assert torch.allclose(f,F.normalize(h,dim=1),atol=1e-6,rtol=1e-6)
        norm=g.norm(dim=1);assert bool((norm>0).all())
        samples=dict(global_norm=norm,correction_norm=c.norm(dim=1),fused_raw_norm=h.norm(dim=1),
            actual_scaled_correction_to_global_norm_ratio=(gain*c).norm(dim=1)/norm,
            global_to_fused_angle_degrees=torch.acos(F.cosine_similarity(g,f,dim=1).clamp(-1,1))*(180/np.pi))
        diagnostic[split]={k:common.describe(v) for k,v in samples.items()}
        torch.save(dict(per_sample_diagnostics=samples),output/f'{split}_statistics.pt')
    torch.save(dict(data,**distances),output/'diagnostic_distances.pt')
    artifact_names=('query_statistics.pt','gallery_statistics.pt','diagnostic_distances.pt')
    report=dict(schema=SCHEMA,status='COMPLETE',dataset=args.dataset,variant='semantic',
        completed_at=common.stamp(),elapsed_seconds=time.perf_counter()-started,
        input_checkpoint_sha256=row['checkpoint_sha256'],selected_epoch=row['best_epoch'],
        original_receipt_sha256=row['receipt_sha256'],binding=binding,readout_gain=gain,
        model_state_before_sha256=before,model_state_after_sha256=after,scores=scores,diagnostic=diagnostic,
        fused_distance_max_absolute_difference_from_original=float((distances['fused']-old['fused']).abs().max()),
        comparisons=dict(same_model_global_to_fused=common.paired(scores['global'],scores['fused'],data['query_ids']),
            independent_global_only_to_same_model_global=common.paired(scores['independent_global_only'],scores['global'],data['query_ids']),
            independent_global_only_to_fused=common.paired(scores['independent_global_only'],scores['fused'],data['query_ids'])),
        artifacts={n:dict(bytes=(output/n).stat().st_size,sha256=common.panel.base.sha(output/n)) for n in artifact_names},
        boundary='Fixed sealed mAP-best, one FP32 eval forward per original query/gallery record, zero updates or selection. g/c/f scoring, model/buffer equality and formula checks retained. Save complete distances and scalar sample statistics; full feature-vector caches intentionally not produced. Same-model global is descriptive, not an independent trained ablation or causal attribution. Original complete six-pair report remains unchanged.')
    common.require_inputs(seal)
    common.write(output/'DIAGNOSIS.json',report)
    print(common.json.dumps(dict(status='COMPLETE',dataset=args.dataset,
        metrics={k:v['metrics'] for k,v in scores.items()},seconds=report['elapsed_seconds'])),flush=True)


common.diagnose=diagnose

if __name__=='__main__':
    assert shutil.disk_usage(ROOT).free>=STORAGE_BYTES
    raise SystemExit(common.main())

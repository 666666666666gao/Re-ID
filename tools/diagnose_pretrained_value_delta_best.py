"""One frozen full-gallery diagnosis of actual value changes after completion."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_pretrained_value_delta as producer
from tools import diagnose_native_research_best as base

runner = producer.inner.runner
SCHEMA = 'trifusion-pretrained-value-fixed-best-diagnosis-v1'
KEYS = ('shared_global','correction','raw_fused','fused')


def require_inputs(seal):
    assert seal['schema'] == SCHEMA
    for name,digest in seal['source_sha256'].items():
        assert runner.sha256(ROOT/name) == digest,name
    for name,digest in seal['artifact_sha256'].items():
        assert runner.sha256(Path(name)) == digest,name
    state=json.loads(Path(seal['campaign']).read_text())
    assert state['status']=='COMPLETE' and state['accepted']==state['expected']==2
    assert json.loads(Path(seal['controller_exit']).read_text())['exit_code']==0
    assert json.loads(Path(seal['report']).read_text())['status']=='COMPLETE_SAVED_DISTANCE_DIAGNOSIS'


def moments(value):
    value=value.double();mean=value.mean(0)
    return dict(sample_mean_energy=float(value.square().sum(-1).mean()),
        common_mean_energy=float(mean.square().sum(-1).mean()),
        centered_energy=float((value-mean).square().sum(-1).mean()))


def extract(model,protocol,split):
    features={key:[] for key in KEYS};role_vectors=[]
    change_l1,change_max,row_sums,region_norm=[],[],[],[]
    with torch.inference_mode():
        for raw in runner.loader_for(protocol,runner.records_for(protocol,split),training=False,method='PLAIN_V8'):
            batch=runner._eval_batch(raw,'RGBNT201')
            output=model.evidence_model.forward_features(batch)
            for key in KEYS:features[key].append(output[key].float().cpu())
            detail=output['value_diagnostics'];delta=detail['attention_change'];regions=detail['regions']
            assert delta.shape[1:]==(3,3,16,128) and regions.shape[1:]==(3,3,16,512)
            vectors=regions.float().mean(3).flatten(2)
            assert vectors.shape[1:]==(3,1536)
            assert torch.allclose(vectors.mean(1),output['correction'].float(),atol=1e-6,rtol=1e-6)
            role_vectors.append(vectors.cpu())
            change_l1.append(delta.float().abs().sum(-1).cpu())
            change_max.append(delta.float().abs().amax(-1).cpu())
            row_sums.append(delta.float().sum(-1).abs().cpu())
            region_norm.append(regions.float().norm(dim=-1).cpu())
    vectors=torch.cat(role_vectors)
    features={key:torch.cat(value) for key,value in features.items()}
    assert all(value.shape==(protocol['counts'][split],1536) and bool(torch.isfinite(value).all()) for value in features.values())
    assert bool(torch.isfinite(vectors).all())
    raw_stats={name:torch.cat(value) for name,value in
        (('attention_change_l1',change_l1),('attention_change_max_abs',change_max),
         ('attention_change_row_sum_abs',row_sums),('actual_region_norm',region_norm))}
    assert all(value.shape==(len(vectors),3,3,16) and bool(torch.isfinite(value).all()) for value in raw_stats.values())
    return features,vectors,raw_stats


def diagnose(args,seal):
    started=time.perf_counter();require_inputs(seal)
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert not args.output_dir.exists()
    torch.set_num_threads(2)
    row=next(row for row in seal['rows'] if row['value_reference']==args.reference)
    original=Path(row['run_dir'])
    values=argparse.Namespace(dataset='RGBNT201',variant='semantic',recipe='semantic',mode='evaluate',seed=42,epochs=50,
        protocol=Path(seal['protocol']),signal_source=Path(seal['signal_source']),clip_weight=Path(seal['clip_weight']),
        initialization=Path(row['initialization']),output_dir=original)
    values.baseline_sha256=runner.sha256(values.clip_weight)
    producer.MODE=args.reference;producer.configure()
    protocol=runner.read_protocol(values.protocol,'RGBNT201')
    model,_cfg,binding=producer.inner.foundation.build(values,protocol)
    training=json.loads((original/'training.json').read_text())
    receipt=json.loads((original/'official_metrics.json').read_text())
    assert binding==training['initializer']==row['initializer']
    assert receipt['schema']==producer.SCHEMA and receipt['status']=='COMPLETE'
    assert receipt['condition']==producer.condition(values)
    assert [item['epoch'] for item in training['history']]==list(range(1,51))
    selected=max(training['history'],key=lambda item:(item['official_fused']['mAP'],item['epoch']))
    payload=producer.inner.foundation.load(original/'best_map.pth',model,values)
    assert payload['epoch']==receipt['selected_epoch']==selected['epoch']==row['best_epoch']
    assert payload['metrics']==selected['official_fused'] and receipt['metrics']==row['metrics']
    assert receipt['checkpoint_sha256']==row['checkpoint_sha256'] and receipt['distance_sha256']==row['distance_sha256']
    model.eval();before=runner._module_state_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    gain=float(model.evidence_model.readout_gain.detach().cpu())
    start=time.perf_counter()
    query,qroles,qdetail=extract(model,protocol,'query')
    gallery,groles,gdetail=extract(model,protocol,'gallery')
    extraction_seconds=time.perf_counter()-start
    after=runner._module_state_sha256(model)
    assert before==after and all(parameter.grad is None for parameter in model.parameters())
    metadata=base.metadata(protocol)
    old=torch.load(original/'official_distances.pt',map_location='cpu',weights_only=False)
    independent=torch.load(Path(seal['independent_global_distance']),map_location='cpu',weights_only=False)
    assert all(np.array_equal(metadata[key],old[key]) and np.array_equal(metadata[key],independent[key]) for key in metadata)
    distances={name:runner.distance_matrix(query[key],gallery[key])
        for name,key in (('global','shared_global'),('correction','correction'),('fused','fused'))}
    for index,name in enumerate(('cnn','transformer','mamba')):
        distances[name+'_contribution']=runner.distance_matrix(qroles[:,index],groles[:,index])
    scores={name:base.score(value,metadata,'RGBNT201',values.signal_source) for name,value in distances.items()}
    scores['independent_global_only']=base.score(independent['fused'],metadata,'RGBNT201',values.signal_source)
    difference=float((distances['fused']-old['fused']).abs().max());assert difference<=1e-5
    assert all(abs(scores['fused']['metrics'][key]-receipt['metrics'][key])<1e-5 for key in receipt['metrics'])
    assert all(abs(scores['independent_global_only']['metrics'][key]-seal['independent_global_metrics'][key])<1e-5 for key in seal['independent_global_metrics'])
    geometry={}
    for split,features,roles,detail in (('query',query,qroles,qdetail),('gallery',gallery,groles,gdetail)):
        g,c,h,f=[features[key] for key in KEYS]
        assert torch.allclose(h,g+gain*c,atol=1e-6,rtol=1e-6)
        assert torch.allclose(f,F.normalize(h,dim=1),atol=1e-6,rtol=1e-6) and bool((g.norm(dim=1)>0).all())
        geometry[split]=dict(scaled_correction_global_ratio=base.describe((gain*c).norm(dim=1)/g.norm(dim=1)),
            global_fused_angle_degrees=base.describe(torch.rad2deg(torch.acos((F.normalize(g,dim=1)*f).sum(1).clamp(-1,1)))),
            correction_energy=moments(c),role_contribution_energy=[moments(roles[:,role]) for role in range(3)],
            reading_change={name:base.describe(value.flatten()) for name,value in detail.items()},
            reading_change_mean_by_role_modality={name:value.mean((0,3)).tolist() for name,value in detail.items()})
    comparisons={name:base.paired(scores[a],scores[b],metadata['query_ids']) for name,a,b in
        (('same_model_global_to_fused','global','fused'),
         ('independent_global_to_same_model_global','independent_global_only','global'),
         ('independent_global_to_fused','independent_global_only','fused'))}
    args.output_dir.mkdir()
    for split,features,roles,detail in (('query',query,qroles,qdetail),('gallery',gallery,groles,gdetail)):
        torch.save(dict(features=features,role_contributions=roles,reading_change=detail),args.output_dir/(split+'_features.pt'))
    torch.save(dict(metadata,**distances),args.output_dir/'diagnostic_distances.pt')
    result=dict(schema=SCHEMA,status='COMPLETE',reference=args.reference,selected_epoch=row['best_epoch'],
        input_checkpoint_sha256=row['checkpoint_sha256'],seal_sha256=runner.sha256(args.seal),
        model_state_before_sha256=before,model_state_after_sha256=after,optimizer_updates=0,
        scores=scores,comparisons=comparisons,geometry=geometry,readout_gain=gain,
        fused_distance_max_abs_difference=difference,extraction_seconds=extraction_seconds,
        elapsed_seconds=time.perf_counter()-started,completed_at=datetime.now().astimezone().isoformat(),
        artifacts={name:dict(bytes=(args.output_dir/name).stat().st_size,sha256=runner.sha256(args.output_dir/name))
            for name in ('query_features.pt','gallery_features.pt','diagnostic_distances.pt')},
        boundary='One frozen selected-best Q/G encoding per accepted arm, zero optimizer/new weights/reselection. Actual attention changes and region/role contributions describe use, not causal identity evidence. Per-role standalone scores do not prove fusion necessity or replace retrained ablations. Full legal camera/gallery protocol and unchanged official fused scores; no test centering, fusion coefficients or teacher assumptions.')
    require_inputs(seal);base.write(args.output_dir/'DIAGNOSIS.json',result)
    print(json.dumps(dict(status=result['status'],reference=args.reference,scores={name:value['metrics'] for name,value in scores.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',choices=('semantic_reference','uniform_reference'),required=True)
    for name in ('seal','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.seal=args.seal.resolve();args.output_dir=args.output_dir.resolve()
    diagnose(args,json.loads(args.seal.read_text()))

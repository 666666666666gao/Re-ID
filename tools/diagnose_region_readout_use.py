"""Frozen E8 readout interventions; one encoding per batch, no training."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnose_global_conditioned_evidence_best as fixed
from tools import diagnose_same_evidence_comparison as comparisons

runner,producer,base=fixed.runner,fixed.producer,fixed.base
SCHEMA='trifusion-fixed-region-readout-use-v1'
CASES=('natural','within_group','cross_group_same_norm','slot_mean_same_norm')
KEYS=('shared_global','correction','raw_fused','fused')


def require_inputs(seal):
    assert seal['schema']==SCHEMA
    for name,digest in seal['source_sha256'].items():
        assert runner.sha256(ROOT/name)==digest,name
    assert runner.sha256(Path(seal['prior_seal']))==seal['prior_seal_sha256']
    prior=json.loads(Path(seal['prior_seal']).read_text())
    fixed.require_inputs(prior)
    for name,digest in seal['artifact_sha256'].items():
        assert runner.sha256(Path(name))==digest,name
    return prior


def extract(model,protocol,split):
    caches={key:[] for key in (*KEYS,'regions')}
    case_vectors={name:[] for name in CASES}
    captured=[]
    calls=0
    errors={'within_c_max_abs':0.0,'same_norm_max_relative':0.0}
    correction_changes={name:[] for name in CASES[1:]}
    def hook(_module,_inputs,evidence):
        assert not captured
        captured.append(evidence)
    handle=model.evidence_model.roles.register_forward_hook(hook)
    gain=model.evidence_model.readout_gain
    within=torch.tensor([1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14],device=gain.device)
    cross=torch.tensor([2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13],device=gain.device)
    with torch.inference_mode():
        records=runner.records_for(protocol,split)
        for raw in runner.loader_for(protocol,records,training=False,method='PLAIN_V8'):
            batch=runner._eval_batch(raw,'RGBNT201')
            output=model.evidence_model.forward_features(batch);calls+=1
            assert output['evidence_exposure'] is None and model.evidence_model.roles.excluded is None
            assert len(captured)==1
            evidence=captured.pop()
            natural=output['correction'].float()
            natural_norm=natural.norm(dim=1,keepdim=True)
            assert (natural_norm>0).all()
            g=output['shared_global'].float()
            tensors=[getattr(evidence,name) for name in ('cnn','transformer','mamba')]
            regions=torch.stack(tensors,dim=1).float().cpu()
            assert regions.shape[1:]==(3,3,16,128) and torch.isfinite(regions).all()
            caches['regions'].append(regions)
            for key in KEYS:
                caches[key].append(output[key].float().cpu())
            case_vectors['natural'].append(output['fused'].float().cpu())
            for name,permutation in (('within_group',within),('cross_group_same_norm',cross),
                                     ('slot_mean_same_norm',None)):
                if name=='slot_mean_same_norm':
                    parts=[x.mean(2,keepdim=True).expand_as(x) for x in tensors]
                else:
                    parts=[x.index_select(2,permutation) for x in tensors]
                changed=type(evidence)(cnn=parts[0],transformer=parts[1],mamba=parts[2],
                                       positions=evidence.positions)
                c=model.evidence_model.read_evidence(changed).float()
                assert torch.isfinite(c).all()
                if name=='within_group':
                    error=float((c-natural).abs().max())
                    errors['within_c_max_abs']=max(errors['within_c_max_abs'],error)
                    assert error<=1e-5
                else:
                    changed_norm=c.norm(dim=1,keepdim=True)
                    assert (changed_norm>0).all()
                    c=c*(natural_norm/changed_norm)
                    error=float(((c.norm(dim=1,keepdim=True)-natural_norm).abs()/natural_norm).max())
                    errors['same_norm_max_relative']=max(errors['same_norm_max_relative'],error)
                    assert error<=1e-5
                delta=c-natural
                correction_changes[name].append(torch.stack((delta.abs().amax(1),
                    delta.norm(dim=1)/natural_norm[:,0],
                    (c*natural).sum(1)/(c.norm(dim=1)*natural_norm[:,0])),dim=1).cpu())
                case_vectors[name].append(F.normalize(g+gain*c,dim=1).cpu())
    handle.remove()
    assert not captured
    caches={key:torch.cat(rows) for key,rows in caches.items()}
    cases={name:torch.cat(rows) for name,rows in case_vectors.items()}
    assert all(x.shape[0]==protocol['counts'][split] and torch.isfinite(x).all()
               for x in (*caches.values(),*cases.values()))
    summaries={}
    for name,rows in correction_changes.items():
        values=torch.cat(rows)
        assert torch.isfinite(values).all() and len(values)==protocol['counts'][split]
        summaries[name]={key:dict(count=len(values),mean=float(values[:,i].mean()),
            min=float(values[:,i].min()),max=float(values[:,i].max()))
            for i,key in enumerate(('max_abs_change','relative_l2_change','cosine_to_natural'))}
    return caches,cases,dict(encoding_batches=calls,correction_changes=summaries,**errors)


def diagnose(args,seal):
    started=time.perf_counter()
    prior=require_inputs(seal)
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    row=next(row for row in prior['rows'] if row['exposure']==args.exposure)
    original=Path(row['run_dir']);output=args.output_dir
    assert not output.exists()
    values=argparse.Namespace(dataset='RGBNT201',variant='semantic',recipe='semantic',exposure=args.exposure,
        mode='evaluate',seed=42,epochs=50,protocol=Path(prior['protocol']),signal_source=Path(prior['signal_source']),
        clip_weight=Path(prior['clip_weight']),initialization=Path(row['initialization']),output_dir=original)
    values.baseline_sha256=runner.sha256(values.clip_weight)
    protocol=runner.read_protocol(values.protocol,'RGBNT201')
    producer.configure();model,_cfg,binding=producer.inner.foundation.build(values,protocol)
    training=json.loads((original/'training.json').read_text())
    receipt=json.loads((original/'official_metrics.json').read_text())
    assert binding==row['initializer']==training['initializer']
    assert receipt['schema']==producer.SCHEMA and receipt['status']=='COMPLETE'
    assert receipt['condition']==producer.condition(values) and receipt['training_epochs']==50
    selected=max(training['history'],key=lambda r:(r['official_fused']['mAP'],r['epoch']))
    payload=producer.inner.foundation.load(original/'best_map.pth',model,values)
    assert payload['epoch']==receipt['selected_epoch']==selected['epoch']==row['selected_epoch']==8
    assert payload['metrics']==selected['official_fused'] and receipt['metrics']==row['metrics']
    assert receipt['checkpoint_sha256']==row['checkpoint_sha256']
    model.eval();before=runner._module_state_sha256(model)
    assert all(p.grad is None for p in model.parameters())
    query,q_cases,q_record=extract(model,protocol,'query')
    gallery,g_cases,g_record=extract(model,protocol,'gallery')
    after=runner._module_state_sha256(model)
    assert before==after and all(p.grad is None for p in model.parameters())
    cached_root=Path(seal['cached_features_root'])/args.exposure
    cached_errors={}
    for split,features in (('query',query),('gallery',gallery)):
        previous=torch.load(cached_root/f'{split}_features.pt',map_location='cpu',weights_only=True)
        cached_errors[split]={key:float((features[key]-previous[key]).abs().max()) for key in KEYS}
        assert all(error<=1e-5 for error in cached_errors[split].values())
    data=base.metadata(protocol)
    old=torch.load(original/'official_distances.pt',map_location='cpu',weights_only=False)
    assert all(np.array_equal(data[key],old[key]) for key in data)
    distances={name:runner.distance_matrix(q_cases[name],g_cases[name]) for name in CASES}
    global_distance=runner.distance_matrix(query['shared_global'],gallery['shared_global'])
    original_error=float((distances['natural']-old['fused']).abs().max())
    within_error=float((distances['within_group']-distances['natural']).abs().max())
    assert original_error<=1e-5 and within_error<=1e-5
    scores={'global':base.score(global_distance,data,'RGBNT201',values.signal_source)}
    scores.update({name:base.score(distance,data,'RGBNT201',values.signal_source) for name,distance in distances.items()})
    assert all(abs(scores['natural']['metrics'][k]-receipt['metrics'][k])<=1e-5 for k in receipt['metrics'])
    assert all(abs(scores['within_group']['metrics'][k]-scores['natural']['metrics'][k])<=1e-5
               for k in receipt['metrics'])
    changes={name:{'versus_global':comparisons.damage(scores['global'],scores[name],data['query_ids']),
                   'versus_natural':comparisons.damage(scores['natural'],scores[name],data['query_ids'])}
             for name in CASES}
    output.mkdir(parents=True)
    for split,features in (('query',query),('gallery',gallery)):
        torch.save(features,output/f'{split}_regions_and_features.pt')
    torch.save({'readout_weight':model.evidence_model.readout.weight.detach().cpu(),
                'gain':float(model.evidence_model.readout_gain.detach().cpu())},output/'readout_state.pt')
    torch.save(dict(data,global_feature=global_distance,**distances),output/'diagnostic_distances.pt')
    result={'schema':SCHEMA,'status':'COMPLETE','exposure':args.exposure,'dataset':'RGBNT201','selected_epoch':8,
        'optimizer_updates':0,'new_training':False,'encoding_batches':{'query':q_record,'gallery':g_record},
        'cached_feature_max_abs_errors':cached_errors,'original_natural_distance_max_abs_error':original_error,
        'within_group_distance_max_abs_error':within_error,'scores':scores,'comparisons':changes,
        'model_state_before_sha256':before,'model_state_after_sha256':after,
        'input_checkpoint_sha256':row['checkpoint_sha256'],'input_seal_sha256':runner.sha256(args.seal),
        'elapsed_seconds':time.perf_counter()-started,'completed_at':datetime.now().astimezone().isoformat(),
        'artifacts':{name:{'bytes':(output/name).stat().st_size,'sha256':runner.sha256(output/name)}
                     for name in ('query_regions_and_features.pt','gallery_regions_and_features.pt',
                                  'readout_state.pt','diagnostic_distances.pt')},
        'boundary':'Post-final-role readout interventions only, not removing upstream bridges or retraining. Both sides changed identically; original global/weights/gain fixed, cross-group/mean correction norms matched per sample. Descriptive frozen-best controls are not new trained methods, stronger teachers, natural-misalignment causes, independent ablations or cross-dataset gains. No test identity enters forward/intervention. Existing benchmark is consumed for development.'}
    require_inputs(seal)
    base.write(output/'DIAGNOSIS.json',result)
    print(json.dumps({'status':'COMPLETE','exposure':args.exposure,'metrics':{n:s['metrics'] for n,s in scores.items()}},allow_nan=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--exposure',required=True,choices=('gradient','random'))
    args=parser.parse_args()
    diagnose(args,json.loads(args.seal.read_text()))

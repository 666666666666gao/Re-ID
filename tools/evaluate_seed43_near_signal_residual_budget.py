"""Evaluate a fixed seed43 readout using saved actual distances and the author scorer."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'modeling'))
from trifusion.residual_budget import apply_residual_budget,cached_budget_distance
from tools import run_official_three_dataset_roles as legacy
from tools.analyze_correspondence_distances import camera_scores,scene_scores
from tools.official_three_dataset_model import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('RGBNT100','MSVR310'),required=True)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir=args.output_dir.resolve()
    assert not args.output_dir.exists();args.output_dir.mkdir(parents=True)
    torch.set_num_threads(2);started=time.perf_counter()
    reg=json.loads(args.registration.read_text());comparison=json.loads(args.comparison.read_text())
    assert comparison['status']=='COMPLETE_PROVISIONAL_CACHED_DISTANCE_ADAPTATION'
    assert comparison['registration_sha256']==sha256(args.registration)
    assert reg['ratios']==[.25,.5,.75]
    for name,digest in reg['source_sha256'].items():assert sha256(ROOT/name)==digest,name
    row=next(r for r in reg['rows'] if r['dataset']==args.dataset)
    chosen=next(r for r in comparison['rows'] if r['dataset']==args.dataset)['selected']
    ratio=chosen['ratio'];assert ratio in reg['ratios']
    for name,digest in row['input_sha256'].items():assert sha256(Path(name))==digest,name
    receipt=json.loads(Path(row['metrics_path']).read_text())
    train_path=Path(row['metrics_path']).parent/'training.json'
    training=json.loads(train_path.read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert training['training']['epochs']==50 and training['seed']==43
    assert training['selected_epoch']==row['selected_epoch']==receipt['selected_epoch']
    assert training['checkpoint_state_sha256']==receipt['model_state_sha256']
    source=ROOT/'comparators/Signal-cd1b0a6'
    model_args=argparse.Namespace(dataset=args.dataset,method='R2',seed=43,
        signal_source=source,clip_weight=ROOT/'pertrained-model/ViT-B-16.pt',
        signal_checkpoint=Path(row['author_checkpoint']),
        signal_sha256=row['input_sha256'][row['author_checkpoint']],epochs=50)
    protocol=legacy.read_protocol(Path(row['protocol']),args.dataset)
    model,_,_,binding=legacy.initialize(model_args,protocol)
    assert binding['author_checkpoint_sha256']==training['initializer']['author_checkpoint_sha256']
    assert binding['signal_state_sha256']==training['initializer']['signal_state_sha256']
    payload=torch.load(row['role_checkpoint'],map_location='cpu',weights_only=True)
    assert payload['schema']=='trifusion-official-role-checkpoint-v1'
    assert payload['dataset']==args.dataset and payload['method']=='R2'
    assert payload['protocol_sha256']==sha256(Path(row['protocol']))
    assert payload['author_checkpoint_sha256']==model_args.signal_sha256
    assert set(payload['role_state_dict'])==legacy.checkpoint_names(model,'R2')
    state=model.state_dict();state.update(payload['role_state_dict']);model.load_state_dict(state,strict=True)
    from tools.run_signal_preserving_v5 import _module_state_sha256
    expected=receipt['model_state_sha256'];assert payload['final_state_sha256']==expected
    assert _module_state_sha256(model)==expected
    from utils import metrics as upstream
    assert Path(upstream.__file__).resolve()==(source/'utils/metrics.py').resolve()
    encode_start=time.perf_counter()
    query=legacy.extract(model,protocol,'query','R2')
    gallery=legacy.extract(model,protocol,'gallery','R2')
    encode_seconds=time.perf_counter()-encode_start
    assert _module_state_sha256(model)==expected and all(p.grad is None for p in model.parameters())
    cached=torch.load(row['distance_path'],map_location='cpu',weights_only=False)
    for split in ('query','gallery'):
        records=protocol['records'][split]
        for field,key in (('ids','identity'),('cameras','camera'),('scenes','scene')):
            assert np.array_equal(cached[split+'_'+field],np.asarray([r[key] for r in records]))
    arrays={name:legacy.distance_matrix(query[name],gallery[name]) for name in ('baseline_only','fused')}
    control_errors={name:float((arrays[name]-cached['distances'][name]).abs().max()) for name in arrays}
    assert max(control_errors.values())<1e-5,control_errors
    block_checks={}
    for split,features in (('query',query),('gallery',gallery)):
        fused=features['fused'];width=reg['baseline_width']
        assert fused.shape[1]==reg['fused_width']
        raw_gnorm=fused[:,:width].norm(dim=1);raw_rnorm=fused[:,width:].norm(dim=1)
        unit=torch.nn.functional.normalize(fused,dim=1)
        gnorm=unit[:,:width].norm(dim=1);rnorm=unit[:,width:].norm(dim=1)
        error=float((gnorm-rnorm).abs().max())
        assert bool((gnorm>0).all()) and bool((rnorm>0).all()) and error<1e-5
        prefix=torch.nn.functional.normalize(fused[:,:width],dim=1)
        baseline=torch.nn.functional.normalize(features['baseline_only'],dim=1)
        prefix_error=float((prefix-baseline).abs().max());assert prefix_error<1e-5
        raw_error=float((raw_gnorm-raw_rnorm).abs().max())
        block_checks[split]=dict(nonzero=True,normalized_equal_norm_max_difference=error,
            raw_float32_norm_max_difference=raw_error,original_raw_check_would_pass=raw_error<1e-5,
            raw_baseline_norm_min=float(raw_gnorm.min()),raw_baseline_norm_max=float(raw_gnorm.max()),
            normalized_baseline_energy_min=float(gnorm.square().min()),normalized_baseline_energy_max=float(gnorm.square().max()),
            normalized_bank_energy_min=float(rnorm.square().min()),normalized_bank_energy_max=float(rnorm.square().max()),
            baseline_prefix_max_difference=prefix_error)
    adjusted_q=apply_residual_budget(query['fused'],reg['baseline_width'],ratio)
    adjusted_g=apply_residual_budget(gallery['fused'],reg['baseline_width'],ratio)
    arrays['adapted']=legacy.distance_matrix(adjusted_q,adjusted_g)
    predicted=cached_budget_distance(cached['distances']['baseline_only'].double(),cached['distances']['fused'].double(),ratio)
    adapted_error=float((arrays['adapted'].double()-predicted).abs().max());assert adapted_error<1e-5
    qids,gids=(cached[k] for k in ('query_ids','gallery_ids'))
    qc,gc=(cached[k] for k in ('query_cameras','gallery_cameras'))
    qs,gs=(cached[k] for k in ('query_scenes','gallery_scenes'))
    path=args.output_dir/'official_distances.pt'
    torch.save(dict(distances={'adapted':arrays['adapted']},
        **{key:value for key,value in cached.items() if key!='distances'}),path)
    saved_actual=torch.load(path,map_location='cpu',weights_only=False)
    arrays['adapted']=saved_actual['distances']['adapted']
    cached_metric_delta={}
    scores={};scorer=camera_scores if args.dataset=='RGBNT100' else scene_scores
    for name,distance in arrays.items():
        sub=args.output_dir/name;sub.mkdir();os.chdir(sub)
        score=scorer(distance.numpy(),qids,gids,qc if args.dataset=='RGBNT100' else qs,gc if args.dataset=='RGBNT100' else gs)
        cmc,map_value=(upstream.eval_func(distance.numpy(),qids,gids,qc,gc) if args.dataset=='RGBNT100'
            else upstream.eval_func_msrv(distance.numpy(),qids,gids,qc,gc,qs,gs))
        assert abs(score['metrics']['mAP']-map_value*100)<1e-10
        assert all(abs(score['metrics'][f'Rank-{k}']-cmc[k-1]*100)<1e-5 for k in (1,5,10))
        reference=chosen['metrics'] if name=='adapted' else receipt['outputs'][name]['metrics']
        if name=='adapted':
            cached_metric_delta={key:score['metrics'][key]-reference[key] for key in reference}
        else:
            assert all(abs(score['metrics'][key]-reference[key])<1e-5 for key in reference),(name,score['metrics'],reference)
        scores[name]=score
    actual=scores['adapted']['metrics'];baseline=scores['baseline_only']['metrics']
    ranks=np.asarray(scores['adapted']['first_match_rank']);old=np.asarray(scores['baseline_only']['first_match_rank'])
    result=dict(status='COMPLETE_ACTUAL_SAVED_DISTANCE_AUTHOR_EVALUATION',dataset=args.dataset,seed=43,ratio=ratio,
        original_selected_epoch=row['selected_epoch'],training_epochs=50,new_training_epochs=0,optimizer_updates=0,
        frozen_query_gallery_encodings=1,verification_attempt=2,prior_failed_full_query_gallery_encodings=1,
        prior_diagnostic_query_batches=0,model_state_sha256=expected,state_equal=True,all_parameter_grads_none=True,
        query_count=len(qids),gallery_count=len(gids),control_distance_max_difference=control_errors,
        adapted_cached_distance_max_difference=adapted_error,block_checks=block_checks,outputs=scores,
        delta_vs_signal={key:actual[key]-baseline[key] for key in actual},
        rank1_repairs=int(((old!=1)&(ranks==1)).sum()),rank1_new_errors=int(((old==1)&(ranks!=1)).sum()),
        all_reported_metrics_strictly_above_signal=all(actual[key]>baseline[key]+1e-10 for key in actual),
        independent_upstream_metrics_equal=True,cached_metric_delta_pp=cached_metric_delta,
        old_cached_metric_parity_pass=all(abs(value)<1e-5 for value in cached_metric_delta.values()),
        saved_actual_distance_is_scoring_input=True,encoding_seconds=encode_seconds,elapsed_seconds=time.perf_counter()-started,
        distance_path=str(path),distance_sha256=sha256(path),control_distance_path=row['distance_path'],
        control_distance_sha256=row['input_sha256'][row['distance_path']],registration_sha256=sha256(args.registration),
        comparison_sha256=sha256(args.comparison),training_receipt_sha256=sha256(train_path),input_sha256=row['input_sha256'],
        driver_sha256=sha256(Path(__file__)),completed_at=datetime.now().astimezone().isoformat(),
        boundary='Second actual encoding of the same fixed seed43 E19 ratio.25 after one separately preserved FAILED cached-metric-parity verification. The next registered evaluation scores its saved actual float32 distance and requires independent author scorer agreement; cached metric difference is diagnostic, not an acceptance criterion. All original state, control-distance, normalized-block and adapted-distance bounds retained. This corrects the evidenced false premise that a close cached distance guarantees identical discrete AP; it does not change the author metric or the old failure. No additional ratio, training, seed mean, architecture change, mechanism causality or unbiased SOTA claim.')
    (args.output_dir/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],dataset=args.dataset,ratio=ratio,metrics=actual,encoding_seconds=encode_seconds)))


if __name__=='__main__':main()

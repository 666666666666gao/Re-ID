"""CPU-only all-query comparisons and complete training traces for fixed-best diagnosis."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def table(path, rows):
    assert rows
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    assert array.size and np.isfinite(array).all()
    return dict(count=int(array.size), mean=float(array.mean()),
                percentiles=dict(zip(('min','p05','p25','median','p75','p95','max'),
                    np.percentile(array, (0,5,25,50,75,95,100)).tolist(), strict=True)))


def score(data, dataset):
    environment = 'scenes' if dataset=='MSVR310' else 'cameras'
    scorer = scene_scores if dataset=='MSVR310' else camera_scores
    return scorer(data['fused'].numpy(), data['query_ids'], data['gallery_ids'],
                  data['query_'+environment], data['gallery_'+environment])


def compare(control, candidate, identities, folder):
    ap0, ap1 = (np.asarray(s['average_precision']) for s in (control,candidate))
    rank0, rank1 = (np.asarray(s['first_match_rank']) for s in (control,candidate))
    assert ap0.shape==ap1.shape==rank0.shape==rank1.shape==identities.shape
    assert np.isfinite(ap0).all() and np.isfinite(ap1).all()
    delta = 100*(ap1-ap0)
    repairs = int(((rank0!=1)&(rank1==1)).sum())
    errors = int(((rank0==1)&(rank1!=1)).sum())
    metrics = {k:candidate['metrics'][k]-control['metrics'][k] for k in control['metrics']}
    assert abs(metrics['mAP']-delta.mean())<1e-5
    assert abs(metrics['Rank-1']-100*(repairs-errors)/len(delta))<1e-5
    identity_rows = [dict(identity=int(identity), queries=int((identities==identity).sum()),
        control_macro_ap_points=float(100*ap0[identities==identity].mean()),
        candidate_macro_ap_points=float(100*ap1[identities==identity].mean()),
        delta_macro_ap_points=float(delta[identities==identity].mean()))
        for identity in np.unique(identities)]
    query_rows = [dict(query_index=i, identity=int(identity), control_ap=float(ap0[i]),
        candidate_ap=float(ap1[i]), delta_ap_points=float(delta[i]),
        control_first_rank=int(rank0[i]), candidate_first_rank=int(rank1[i]))
        for i,identity in enumerate(identities)]
    folder.mkdir()
    table(folder/'query_changes.csv', query_rows)
    table(folder/'identity_changes.csv', identity_rows)
    result = dict(control_metrics=control['metrics'], candidate_metrics=candidate['metrics'],
        delta_metrics=metrics, rank1_repairs=repairs, rank1_new_errors=errors,
        query_ap_delta_points=distribution(delta),
        identity_macro_delta_ap_points=distribution([r['delta_macro_ap_points'] for r in identity_rows]),
        query_ap_improved=int((delta>1e-6).sum()), query_ap_worsened=int((delta<-1e-6).sum()),
        boundary='All queries of the same selected fixed model. Identity averages do not measure training-seed variation; no checkpoint, gain or threshold selection.')
    write(folder/'COMPARISON.json', result)
    return result


def training_trace(row, job, output):
    run = Path(row['run_dir'])
    paths = [run/name for name in ('training.json','training_steps.jsonl','training_batch_order.jsonl')]
    training = json.loads(paths[0].read_text())
    steps = [json.loads(line) for line in paths[1].read_text().splitlines()]
    batches = [json.loads(line) for line in paths[2].read_text().splitlines()]
    history = training['history']
    assert [r['epoch'] for r in history]==list(range(1,51))
    assert len(steps)==len(batches)==sum(r['steps'] for r in history)
    assert [(r['epoch'],r['batch']) for r in steps]==[(r['epoch'],r['batch']) for r in batches]
    assert training['best_epoch']==row['best_epoch']
    heads = len(row['initializer']['head_names'])
    columns = []
    for step in steps:
        assert set(step)==set(steps[0])
        assert len(step['head_losses'])==len(step['global_head_losses'])==heads
        values = {name:float(value) for name,value in step.items()
                  if name not in ('epoch','batch','lr','head_losses','global_head_losses')}
        values.update({f'role_head_{i}_loss':float(v) for i,v in enumerate(step['head_losses'])})
        values.update({f'global_head_{i}_loss':float(v) for i,v in enumerate(step['global_head_losses'])})
        values.update(lr_min=float(min(step['lr'])),lr_max=float(max(step['lr'])))
        assert all(np.isfinite(v) for v in values.values())
        columns.append(values)
    epoch_rows = []
    for historical in history:
        selected = [v for step,v in zip(steps,columns,strict=True) if step['epoch']==historical['epoch']]
        assert len(selected)==historical['steps']
        means = {key:float(np.mean([r[key] for r in selected])) for key in columns[0]}
        assert abs(means['loss']-historical['mean_loss'])<1e-8
        epoch_rows.append(dict(epoch=historical['epoch'], steps=len(selected),
            training_loop_seconds=historical['seconds'], **historical['official_fused'], **means))
    table(output/'complete_50_epoch_trace.csv', epoch_rows)
    best = next(r for r in history if r['epoch']==row['best_epoch'])
    assert all(abs(best['official_fused'][k]-row['metrics'][k])<1e-5 for k in row['metrics'])
    intervals = {s['mode']:(datetime.fromisoformat(s['completed_at'])-
        datetime.fromisoformat(s['started_at'])).total_seconds() for s in job['steps']}
    result = dict(epochs=50, formal_steps=len(steps),
        input_sha256={str(p):sha(p) for p in paths},
        full_step_distributions={key:distribution([r[key] for r in columns]) for key in columns[0]},
        best_epoch=row['best_epoch'], best_metrics=row['metrics'], last_metrics=history[-1]['official_fused'],
        best_minus_last={k:row['metrics'][k]-history[-1]['official_fused'][k] for k in row['metrics']},
        training_loop_seconds_sum=sum(r['seconds'] for r in history),
        recorded_child_intervals_seconds=intervals,
        boundary='All recorded batches and all50 epochs. Loop seconds exclude epoch evaluation/save; train-child interval includes them. Recorded norms do not establish task-gradient magnitude or calibrated correspondence.')
    write(output/'TRAINING.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.diagnosis,args.output = args.diagnosis.resolve(),args.output.resolve()
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    summary_path = args.diagnosis/'SUMMARY.json'
    summary = json.loads(summary_path.read_text())
    seal = json.loads((args.diagnosis/'INPUT_SEAL.json').read_text())
    assert summary['schema']==seal['schema']=='fixed-best-row-transport-diagnosis-v1'
    assert summary['status']=='COMPLETE' and summary['accepted_models']==6 and summary['deployment_modes']==18
    assert all(sha(ROOT/name)==digest for name,digest in seal['diagnostic_source_sha256'].items())
    assert all(sha(ROOT/name)==digest for name,digest in seal['original_source_sha256'].items())
    campaign = json.loads((Path(seal['original_campaign'])/'campaign.json').read_text())
    matrix = json.loads((Path(seal['original_campaign'])/'accepted_matrix.json').read_text())
    assert campaign['status']=='COMPLETE' and campaign['report_exit_code']==0 and matrix['accepted']==6
    assert not args.output.exists()
    args.output.mkdir()
    results = []
    for endpoint in summary['rows']:
        dataset,variant = endpoint['dataset'],endpoint['variant']
        folder = args.diagnosis/(dataset+'_'+variant)
        output = args.output/(dataset+'_'+variant)
        output.mkdir()
        row = next(r for r in matrix['rows'] if (r['dataset'],r['variant'])==(dataset,variant))
        job = next(j for j in campaign['jobs'] if (j['phase'],j['dataset'],j['variant'])==('full',dataset,variant))
        assert sha(Path(row['run_dir'])/'best_map.pth')==row['checkpoint_sha256']
        assert sha(Path(row['run_dir'])/'official_distances.pt')==row['distance_sha256']
        assert sha(Path(row['run_dir'])/'official_metrics.json')==row['receipt_sha256']
        accepted_data = torch.load(Path(row['run_dir'])/'official_distances.pt',
                                   map_location='cpu',weights_only=False)
        scores,data = {},{}
        for mode in ('original','opposite','self_only'):
            result = json.loads((folder/mode/'RESULT.json').read_text())
            assert sha(folder/mode/'distances.pt')==result['distance_sha256']
            assert sha(folder/mode/'diagnostic_arrays.pt')==result['arrays_sha256']
            data[mode] = torch.load(folder/mode/'distances.pt',map_location='cpu',weights_only=False)
            assert all(np.array_equal(data[mode][key],data['original'][key])
                       for key in data[mode] if key!='fused')
            scores[mode] = score(data[mode],dataset)
            assert all(abs(scores[mode]['metrics'][k]-result['metrics'][k])<1e-5 for k in result['metrics'])
        assert all(abs(scores['original']['metrics'][k]-row['metrics'][k])<1e-5 for k in row['metrics'])
        assert all(np.array_equal(data['original'][key],accepted_data[key])
                   for key in accepted_data if key!='fused')
        scores['own_global'] = json.loads((folder/'original/own_global.json').read_text())
        scores['own_correction'] = json.loads((folder/'original/own_correction.json').read_text())
        pairs = {(a+'__'+b):compare(scores[a],scores[b],data['original']['query_ids'],output/(a+'__'+b))
                 for a,b in (('original','opposite'),('original','self_only'),
                             ('own_global','original'),('own_global','own_correction'))}
        arrays = torch.load(folder/'original/diagnostic_arrays.pt',map_location='cpu',weights_only=False)
        q = len(data['original']['query_ids'])
        count = q+len(data['original']['gallery_ids'])
        assert arrays['statistics'].shape==(count,6,9) and arrays['norms'].shape==(count,5)
        assert arrays['peer_norms'].shape==(count,3,4)
        measurements = {name:distribution(arrays['statistics'][:,:,i].numpy())
                        for i,name in enumerate(arrays['statistics_columns'])}
        measurements.update({name:distribution(arrays['norms'][:,i].numpy())
                             for i,name in enumerate(arrays['norm_columns'])})
        measurements.update({name:distribution(arrays['peer_norms'][:,:,i].numpy())
                             for i,name in enumerate(arrays['peer_norm_columns'])})
        assert bool((arrays['norms'][:,0]>0).all())
        measurements['actual_scaled_correction_global_ratio'] = distribution(
            (arrays['norms'][:,2]/arrays['norms'][:,0]).numpy())
        result = dict(dataset=dataset,variant=variant,best_epoch=row['best_epoch'],
            pairs=pairs,full_query_gallery_measurements=measurements,
            weight_statistics=endpoint['weight_statistics'],training=training_trace(row,job,output))
        write(output/'ENDPOINT.json',result)
        results.append(result)
    assert all(sha(ROOT/name)==digest for name,digest in seal['diagnostic_source_sha256'].items())
    assert all(sha(ROOT/name)==digest for name,digest in seal['original_source_sha256'].items())
    result = dict(status='COMPLETE',schema=seal['schema'],completed_at=datetime.now().astimezone().isoformat(),
        fixed_models=6,deployment_modes=18,all_query_comparisons=24,rows=results,
        diagnosis_summary_sha256=sha(summary_path),
        boundary='Descriptive fixed-best single-seed diagnosis, no retraining, hyperparameter selection, geometric correspondence truth or SOTA/stability claim. All endpoints and query rows retained.')
    write(args.output/'SUMMARY.json',result)
    print(json.dumps(dict(status='COMPLETE',fixed_models=6,all_query_comparisons=24)))


if __name__=='__main__':
    main()

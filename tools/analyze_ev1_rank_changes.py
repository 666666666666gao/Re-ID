"""CPU-only decomposition of closed EV1 saved rankings; no model execution."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def scores(data, dataset):
    distance = data['fused'].numpy()
    query_ids, gallery_ids = np.asarray(data['query_ids']), np.asarray(data['gallery_ids'])
    environment = 'scenes' if dataset == 'MSVR310' else 'cameras'
    query_env, gallery_env = np.asarray(data['query_'+environment]), np.asarray(data['gallery_'+environment])
    ap, first, margins = [], [], []
    for index, order in enumerate(np.argsort(distance, axis=1)):
        keep = ~((gallery_ids[order] == query_ids[index]) & (gallery_env[order] == query_env[index]))
        kept = order[keep]
        matches = gallery_ids[kept] == query_ids[index]
        positions = np.flatnonzero(matches)
        assert positions.size and (~matches).any()
        ap.append(float(np.mean(np.cumsum(matches)[positions] / (positions+1))))
        first.append(int(positions[0]+1))
        margins.append(float(distance[index,kept[~matches]].min()-distance[index,kept[matches]].min()))
    return np.asarray(ap), np.asarray(first), np.asarray(margins)


def group(delta, mask, total):
    values = delta[mask]
    return {'queries':int(mask.sum()), 'ap_improved':int((values>1e-8).sum()),
            'ap_worsened':int((values < -1e-8).sum()),
            'mean_delta_ap_points':float(values.mean()*100) if values.size else None,
            'contribution_to_official_delta_map_points':float(values.sum()*100/total)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    report_path=ROOT/'results/semantic_native_evidence_complete_20261002/SUMMARY.json'
    report=json.loads(report_path.read_text())
    assert report['status']=='COMPLETE' and report['accepted']==6
    rows=report['rows']+report['prior_control_rows']+report['historical_native_high_rows']
    results=[]
    query_rows={}
    for dataset in ('RGBNT201','RGBNT100','MSVR310'):
        selected={name:next(r for r in rows if r['dataset']==dataset and r['variant']==name)
                  for name in ('semantic','combined','roles','native_high')}
        arrays, scored={}, {}
        for name,row in selected.items():
            path=Path(row['run_dir'])/'official_distances.pt'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['distance_sha256']
            arrays[name]=torch.load(path,map_location='cpu',weights_only=False)
            scored[name]=scores(arrays[name],dataset)
            ap,rank,_=scored[name]
            metrics={'mAP':float(ap.mean()*100), **{f'Rank-{k}':float((rank<=k).mean()*100) for k in (1,5,10)}}
            assert all(abs(metrics[k]-row['metrics'][k])<1e-5 for k in metrics)
            for key in ('query_ids','gallery_ids','query_cameras','gallery_cameras','query_scenes','gallery_scenes'):
                assert np.array_equal(arrays['semantic'][key], arrays[name][key])
            assert arrays[name]['fused'].shape==arrays['semantic']['fused'].shape
        original_ap,original_rank,margin=scored['semantic']
        combined_ap,combined_rank,_=scored['combined']
        delta=combined_ap-original_ap
        total=len(delta)
        first_correct, second_correct=original_rank==1, combined_rank==1
        masks={'both_correct':first_correct & second_correct,
               'repaired':~first_correct & second_correct,
               'new_error':first_correct & ~second_correct,
               'both_wrong':~first_correct & ~second_correct}
        groups={name:group(delta,mask,total) for name,mask in masks.items()}
        assert sum(row['queries'] for row in groups.values())==total
        assert abs(sum(row['contribution_to_official_delta_map_points'] for row in groups.values())-delta.mean()*100)<1e-10
        pair=next(p['paired_diagnosis'] for p in report['pairs']
                  if p['dataset']==dataset and p['paired_diagnosis']['control']=='semantic')
        assert groups['repaired']['queries']==pair['rank1_repairs']
        assert groups['new_error']['queries']==pair['rank1_new_errors']
        cuts=np.quantile(margin[first_correct],[.25,.5,.75])
        quartile=np.searchsorted(cuts,margin,side='right')
        fragility=[]
        for index in range(4):
            mask=first_correct & (quartile==index)
            fragility.append({'quartile':index+1, **group(delta,mask,total),
                'new_rank1_errors':int((mask & ~second_correct).sum()),
                'new_rank1_error_rate':float((mask & ~second_correct).sum()/mask.sum()) if mask.any() else None})
        prior_delta=scored['native_high'][0]-scored['roles'][0]
        persistent={'both_detail_interventions_ap_worse':int(((prior_delta < -1e-8)&(delta < -1e-8)).sum()),
                    'n1_worse_ev1_better':int(((prior_delta < -1e-8)&(delta > 1e-8)).sum()),
                    'n1_better_ev1_worse':int(((prior_delta > 1e-8)&(delta < -1e-8)).sum()),
                    'both_detail_interventions_ap_better':int(((prior_delta > 1e-8)&(delta > 1e-8)).sum())}
        qids=np.asarray(arrays['semantic']['query_ids'])
        query_rows[dataset]=[{
            'query_index':int(index),'identity':int(qids[index]),
            'semantic_ap':float(original_ap[index]),'combined_ap':float(combined_ap[index]),
            'semantic_first_rank':int(original_rank[index]),'combined_first_rank':int(combined_rank[index]),
            'semantic_margin':float(margin[index]),'roles_ap':float(scored['roles'][0][index]),
            'native_high_ap':float(scored['native_high'][0][index])}
            for index in range(total)]
        identities=[{'identity':int(identity),'queries':int((qids==identity).sum()),
                     'ev1_mean_delta_ap_points':float(delta[qids==identity].mean()*100),
                     'n1_mean_delta_ap_points':float(prior_delta[qids==identity].mean()*100)}
                    for identity in np.unique(qids)]
        persistent['identities_both_worse']=sum(row['ev1_mean_delta_ap_points'] < -1e-6 and row['n1_mean_delta_ap_points'] < -1e-6 for row in identities)
        results.append({'dataset':dataset,'queries':total,'delta_map_points':float(delta.mean()*100),
            'delta_rank1_points':float((second_correct.mean()-first_correct.mean())*100),
            'rank1_groups':groups,'semantic_correct_margin_quartile_cuts':cuts.tolist(),
            'semantic_correct_margin_quartiles':fragility,'cross_intervention_overlap':persistent,
            'identity_changes':identities,
            'bindings':{name:{k:row[k] for k in ('run_dir','best_epoch','checkpoint_sha256','distance_sha256')} for name,row in selected.items()}})
        del arrays,scored
    output={'status':'CLOSED_EV1_CPU_RANK_DECOMPOSITION_COMPLETE',
        'created_at':datetime.now().astimezone().isoformat(),'rows':results,
        'source_report':str(report_path),'cuda_initialized':torch.cuda.is_initialized(),
        'model_runs':0,'training_runs':0,'weights_generated':0,
        'boundary':'Saved selected-checkpoint full-gallery real-GT rankings only. New decomposition is descriptive, not new model inference, evaluation selection, causal attribution or seed replication. Baseline margin uses ground truth for diagnosis; not a deployable gate/router. N1/EV1 overlap compares different intervention controls and selected epochs, not a common-control causal effect. Old failures and new native untrained status unchanged.'}
    assert not output['cuda_initialized']
    args.output_dir.mkdir(parents=True)
    for dataset, values in query_rows.items():
        with (args.output_dir/(dataset+'_QUERY_CHANGES.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (args.output_dir/'SUMMARY.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'rows':[{k:r[k] for k in ('dataset','queries','delta_map_points','rank1_groups')} for r in results]}),flush=True)


if __name__=='__main__':
    main()

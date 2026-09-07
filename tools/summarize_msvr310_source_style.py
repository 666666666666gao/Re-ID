#!/usr/bin/env python3
"""Local text-only complete paired ranking and identity analysis; no model files."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES=('baseline_only','fused','cnn','transformer','mamba')
ENDS=('control','source_style')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root=args.directory
    summary=json.loads((root/'q1_summary.json').read_bytes())
    verification=json.loads((root/'q1_cpu.json').read_bytes())
    resume=json.loads((root/'resume_verification.json').read_bytes())
    protocol=json.loads(args.protocol.read_bytes())
    assert verification['status']=='PASS_COMPLETE_MSVR_STYLE_Q1'
    assert verification['summary_sha256']==sha(root/'q1_summary.json')
    assert resume['status']=='PASS_FIXED_ORIGINAL_END_REUSE_AND_EXACT_CPU_DISTANCE'
    assert resume['summary_sha256']==sha(root/'q1_summary.json')
    assert resume['complete_cpu_verification_sha256']==sha(root/'q1_cpu.json')
    assert resume['original_training_updates_reused']==summary['original_training_updates_reused']==260
    assert resume['new_optimizer_steps']==summary['new_optimizer_steps']==1300
    assert summary['distance_cpu_threads']==56 and summary['training_cpu_threads']==4
    assert sum(r['training_reused_from_original_run'] for f in summary['folds'] for r in f['endpoints'].values())==1
    assert summary['folds'][0]['endpoints']['control']['training_reused_from_original_run']
    assert verification['checked_training_steps']==summary['optimizer_steps']==1560
    assert summary['heldout_record_forwards']==2064
    assert summary['official_image_reads']==summary['rgbnt201_dev_image_reads']==0
    queries=[];costs={end:dict(training_seconds=0.,updates=0,peak_reserved_mib=0.,active_style_steps=0,extra_visual_passes=0) for end in ENDS}
    for f,fold in zip(summary['folds'],protocol['folds'],strict=True):
        assert f['fold']==fold['fold']
        gallery=[protocol['records'][i] for i in fold['gallery_record_indices']]
        ranks={}
        for end in ENDS:
            path=root/f"fold_{fold['fold']}_{end}_rankings.json"
            row=f['endpoints'][end]
            assert sha(path)==row['retrieval']['rankings_sha256']
            ranks[end]=json.loads(path.read_bytes())
            tr=row['training'];cost=costs[end]
            cost['training_seconds']+=sum(e['elapsed_seconds'] for e in tr['history'])
            cost['updates']+=tr['optimizer_steps'];cost['peak_reserved_mib']=max(cost['peak_reserved_mib'],tr['peak_reserved_mib'])
            cost['active_style_steps']+=tr['active_style_steps'];cost['extra_visual_passes']+=tr['additional_visual_passes']
        for qi,q in enumerate(fold['query_rows']):
            source=gallery[q['gallery_position']]
            assert source['identity']==q['identity']
            item=dict(fold=fold['fold'],record_index=q['record_index'],identity=q['identity'],camera=source['camera'],scene=source['scene'],outputs={})
            for name in NAMES:
                values={}
                for end in ENDS:
                    order=ranks[end][name][qi]
                    assert sorted(order)==list(range(len(gallery)))
                    legal=[g for g in order if not(gallery[g]['identity']==source['identity'] and gallery[g]['scene']==source['scene'])]
                    positive=[k+1 for k,g in enumerate(legal) if gallery[g]['identity']==source['identity']]
                    assert len(positive)==q['valid_positives']>0
                    ap=sum((i+1)/position for i,position in enumerate(positive))/len(positive)
                    logged=f['endpoints'][end]['retrieval']['outputs'][name]
                    assert abs(ap-logged['average_precision'][qi])<1e-14 and positive[0]==logged['first_match_rank'][qi]
                    first=gallery[legal[0]]
                    values[end]=dict(ap=ap,first_match_rank=positive[0],first_gallery={k:first[k] for k in ('index','identity','camera','scene','paths')})
                item['outputs'][name]=values
            queries.append(item)
    assert len(queries)==600
    identities=np.array([r['identity'] for r in queries]);unique=np.unique(identities);assert len(unique)==60
    changes={};all_metrics={end:{} for end in ENDS}
    for name in NAMES:
        aps={end:np.array([r['outputs'][name][end]['ap'] for r in queries]) for end in ENDS}
        first={end:np.array([r['outputs'][name][end]['first_match_rank'] for r in queries]) for end in ENDS}
        for end in ENDS:
            metrics=dict(mAP=float(aps[end].mean()*100),**{f'Rank-{k}':float((first[end]<=k).mean()*100) for k in (1,5,10)})
            for k,v in metrics.items():assert abs(v-summary['comparison']['endpoints'][end]['metrics'][name][k])<1e-10
            all_metrics[end][name]=metrics
        delta=(aps['source_style']-aps['control'])*100
        assert abs(float(delta.mean())-summary['comparison']['matched_gains_mAP'][name])<1e-10
        rows=[dict(identity=int(i),queries=int((identities==i).sum()),gain_mAP=float(delta[identities==i].mean()),
                   query_weighted_contribution_pp=float(delta[identities==i].sum()/600)) for i in unique]
        repaired=(first['control']>1)&(first['source_style']==1);broken=(first['control']==1)&(first['source_style']>1)
        broken_rows=[queries[i] for i in np.flatnonzero(broken)]
        sums=np.array([delta[identities==i].sum() for i in unique]);counts=np.array([(identities==i).sum() for i in unique])
        sample=np.random.default_rng(42).integers(0,60,size=(10000,60))
        lower=float(np.quantile(sums[sample].sum(1)/counts[sample].sum(1),.025,method='linear'))
        if name=='fused':assert abs(lower-summary['comparison']['paired_bootstrap_lower_pp'])<1e-10
        changes[name]=dict(matched_gain_mAP=float(delta.mean()),bootstrap_lower_pp=lower,
                           identity_improved=sum(r['gain_mAP']>0 for r in rows),identity_declined=sum(r['gain_mAP']<0 for r in rows),
                           all_identity_contributions=rows,rank1_repaired=int(repaired.sum()),rank1_new_errors=int(broken.sum()),
                           new_errors_same_camera=sum(r['camera']==r['outputs'][name]['source_style']['first_gallery']['camera'] for r in broken_rows),
                           new_errors_same_scene=sum(r['scene']==r['outputs'][name]['source_style']['first_gallery']['scene'] for r in broken_rows),
                           all_new_error_record_indices=[r['record_index'] for r in broken_rows])
    for cost in costs.values():
        assert cost['updates']==780 and cost['extra_visual_passes']==2340
        cost['seconds_per_update']=cost['training_seconds']/780
    assert costs['control']['active_style_steps']==0 and costs['source_style']['active_style_steps']==393
    result=dict(status='PASS_COMPLETE_LOCAL_MSVR_STYLE_RANKING_AND_IDENTITY_ANALYSIS',scientific_status=summary['status'],
                generated_at=datetime.now().astimezone().isoformat(),summary_sha256=sha(root/'q1_summary.json'),
                remote_cpu_verification_sha256=sha(root/'q1_cpu.json'),script_sha256=sha(__file__),
                aggregate_metrics=all_metrics,changes=changes,costs=costs,all600_queries=queries,
                original_training_updates_reused=260,new_optimizer_steps=1300,
                original_failed_run_files_unchanged=summary['original_failed_run_files_unchanged'],
                resume_verification_sha256=sha(root/'resume_verification.json'),
                scientific_checks=summary['comparison']['paired_checks'],vehicle_checks=summary['comparison']['endpoints']['source_style']['scientific_checks'],
                boundaries=['Internal full-path identity-isolated comparison, not official or zero-shot transfer.',
                            'All600 queries and60 eligible identities included; no per-test-identity tuning.',
                            'Camera/scene correlations do not establish visual cause. No original images read locally.',
                            'Rank and scalar verification here; full feature/distance/checkpoint verification is bound remote CPU evidence.'])
    assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('all600_queries','changes')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

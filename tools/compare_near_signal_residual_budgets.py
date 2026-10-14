"""A preregistered three-value inference adaptation on two close R2 endpoints."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'modeling'))
from trifusion.residual_budget import apply_residual_budget,cached_budget_distance
from tools.analyze_correspondence_distances import camera_scores,scene_scores
from tools.collect_correspondence_roles import sha

RATIOS=(.25,.5,.75)


def component_check():
    generator=torch.Generator().manual_seed(42)
    g=torch.nn.functional.normalize(torch.randn(7,5,generator=generator,dtype=torch.float64),dim=1)
    r=torch.nn.functional.normalize(torch.randn(7,9,generator=generator,dtype=torch.float64),dim=1)
    original=torch.cat((g,r),dim=1)/2**.5
    distance=lambda x:(x[:,None]-x[None,:]).square().sum(-1)
    checks=[]
    for ratio in (*RATIOS,1.):
        actual=distance(apply_residual_budget(original,5,ratio))
        expected=cached_budget_distance(distance(g),distance(original),ratio)
        error=float((actual-expected).abs().max())
        assert error<1e-12
        checks.append(dict(ratio=ratio,maximum_difference=error))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();assert not args.output_dir.exists()
    torch.set_num_threads(1)
    registration=json.loads(args.registration.read_text())
    assert registration['ratios']==list(RATIOS) and registration['datasets']==['RGBNT100','MSVR310']
    for name,digest in registration['source_sha256'].items():assert sha(ROOT/name)==digest,name
    rows=[]
    for row in registration['rows']:
        for name,digest in row['input_sha256'].items():assert sha(Path(name))==digest,name
        receipt=json.loads(Path(row['metrics_path']).read_text())
        data=torch.load(row['distance_path'],map_location='cpu',weights_only=False)
        assert receipt['status']=='COMPLETE' and receipt['method']=='R2' and receipt['training_epochs']==50
        env='cameras' if row['dataset']=='RGBNT100' else 'scenes'
        scorer=camera_scores if env=='cameras' else scene_scores
        score=lambda d:scorer(d.numpy(),data['query_ids'],data['gallery_ids'],data['query_'+env],data['gallery_'+env])
        baseline=score(data['distances']['baseline_only']);original=score(data['distances']['fused'])
        for name,value in (('baseline_only',baseline),('fused',original)):
            assert all(abs(value['metrics'][key]-receipt['outputs'][name]['metrics'][key])<1e-5 for key in value['metrics'])
        candidates=[]
        for ratio in RATIOS:
            d=cached_budget_distance(data['distances']['baseline_only'].double(),data['distances']['fused'].double(),ratio)
            values=score(d)
            baseline_rank=np.asarray(baseline['first_match_rank']);candidate_rank=np.asarray(values['first_match_rank'])
            repairs=int(((baseline_rank!=1)&(candidate_rank==1)).sum())
            harms=int(((baseline_rank==1)&(candidate_rank!=1)).sum())
            candidates.append(dict(ratio=ratio,metrics=values['metrics'],
                delta_vs_signal={key:values['metrics'][key]-baseline['metrics'][key] for key in values['metrics']},
                rank1_repairs=repairs,rank1_new_errors=harms,
                all_reported_metrics_strictly_above_signal=all(values['metrics'][key]>baseline['metrics'][key]+1e-10 for key in values['metrics']),
                main_metrics_strictly_above_signal=all(values['metrics'][key]>baseline['metrics'][key]+1e-10 for key in ('mAP','Rank-1'))))
        selected=max(candidates,key=lambda v:(v['all_reported_metrics_strictly_above_signal'],
            v['metrics']['mAP'],v['metrics']['Rank-1'],v['ratio']))
        rows.append(dict(dataset=row['dataset'],signal_metrics=baseline['metrics'],original_r2_metrics=original['metrics'],
            candidates=candidates,selected=selected,selection_basis='all-four goal first, then mAP and R1',
            input_sha256=row['input_sha256']))
    result=dict(status='COMPLETE_PROVISIONAL_CACHED_DISTANCE_ADAPTATION',rows=rows,component_checks=component_check(),
        registration_sha256=sha(args.registration),at=datetime.now().astimezone().isoformat(),
        nn_calls=0,optimizer_updates=0,
        boundary='Fixed grid registered before scoring, close historical R2 endpoints only. Same original full50 best checkpoints; inference amplitude parameter, no new architecture or scientific mechanism claim. Cached algebra assumes nonzero equal-energy blocks and must be validated by actual frozen-model encoding before accepting adapted scores. Consumed official benchmark selection; not unbiased SOTA. All candidates reported, no seed mean or query-specific rule.')
    args.output_dir.mkdir()
    (args.output_dir/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],rows=[dict(dataset=r['dataset'],selected=r['selected']) for r in rows])))


if __name__=='__main__':main()

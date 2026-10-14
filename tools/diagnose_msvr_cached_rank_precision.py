"""Diagnose failed cached-metric parity from saved author top50 lists; no NN."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'modeling'))
from trifusion.residual_budget import cached_budget_distance
from tools.analyze_correspondence_distances import scene_scores
from tools.official_three_dataset_model import sha256


def parse(value):
    found=re.fullmatch(r'(-?\d+)_s(-?\d+)_v(-?\d+)',value);assert found,value
    return tuple(int(x) for x in found.groups())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--failed-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();assert not args.output_dir.exists();torch.set_num_threads(2)
    reg=json.loads(args.registration.read_text());comparison=json.loads(args.comparison.read_text())
    assert comparison['registration_sha256']==sha256(args.registration)
    row=next(r for r in reg['rows'] if r['dataset']=='MSVR310')
    chosen=next(r for r in comparison['rows'] if r['dataset']=='MSVR310')['selected'];assert chosen['ratio']==.25
    for name,digest in row['input_sha256'].items():assert sha256(Path(name))==digest,name
    data=torch.load(row['distance_path'],map_location='cpu',weights_only=False)
    qids,gids=data['query_ids'],data['gallery_ids'];qs,gs=data['query_scenes'],data['gallery_scenes']
    qc,gc=data['query_cameras'],data['gallery_cameras']
    proxy=cached_budget_distance(data['distances']['baseline_only'].double(),data['distances']['fused'].double(),.25)
    score=lambda d:scene_scores(d.numpy(),qids,gids,qs,gs)
    double_score=score(proxy);rounded=score(proxy.float())
    assert all(abs(double_score['metrics'][k]-chosen['metrics'][k])<1e-10 for k in chosen['metrics'])
    arrays=dict(baseline_only=data['distances']['baseline_only'],fused=data['distances']['fused'],adapted=proxy)
    reports={};text_hashes={};actual_patterns={}
    for name,distance in arrays.items():
        path=args.failed_run/name/'re.txt';text_hashes[name]=sha256(path)
        lines=path.read_text().splitlines();assert len(lines)==1+2*len(qids) and lines[0]=='rank list file'
        changed=[];patterns=[]
        for i in range(len(qids)):
            assert parse(lines[1+2*i].rstrip(':'))==(int(qids[i]),int(qs[i]),int(qc[i]))
            actual=[parse(x) for x in lines[2+2*i].split()]
            order=np.argsort(distance[i].numpy());keep=~((gids[order]==qids[i])&(gs[order]==qs[i]))
            kept=order[keep][:50];assert len(actual)==len(kept)
            expected=[(int(gids[j]),int(gs[j]),int(gc[j])) for j in kept]
            a=[x[0]==qids[i] for x in actual];b=[x[0]==qids[i] for x in expected]
            if a!=b:changed.append(dict(query_index=i,actual_positive_ranks=[k+1 for k,v in enumerate(a) if v],
                cached_positive_ranks=[k+1 for k,v in enumerate(b) if v]))
            patterns.append(a)
        if name!='adapted':assert not changed,(name,changed)
        cmc={f'Rank-{k}':float(np.mean([any(pattern[:k]) for pattern in patterns])*100) for k in (1,5,10)}
        reports[name]=dict(actual_top50_cmc=cmc,positive_pattern_changed_queries=len(changed),changes=changed)
        actual_patterns[name]=patterns
    old=np.asarray([a[0] for a in actual_patterns['baseline_only']]);new=np.asarray([a[0] for a in actual_patterns['adapted']])
    result=dict(status='COMPLETE_SAVED_AUTHOR_RANK_TEXT_PRECISION_DIAGNOSIS',dataset='MSVR310',ratio=.25,
        original_double_proxy_metrics=double_score['metrics'],float32_rounded_proxy_metrics=rounded['metrics'],
        rounded_minus_double_mAP=rounded['metrics']['mAP']-double_score['metrics']['mAP'],
        quantization_distance_max_difference=float((proxy-proxy.float().double()).abs().max()),
        author_top50=reports,actual_rank1_repairs=int((~old&new).sum()),actual_rank1_new_errors=int((old&~new).sum()),
        nn_calls=0,optimizer_updates=0,original_failed_run=str(args.failed_run),rank_text_sha256=text_hashes,
        driver_sha256=sha256(Path(__file__)),registration_sha256=sha256(args.registration),
        boundary='Original actual966 MSVR metric-parity failure remains FAILED and no threshold is relaxed. Cast is a fixed-ratio numeric diagnosis, not a new candidate or accepted model distance. Saved text contains only top50 identity/scene/camera tuples; it cannot reconstruct full AP, exact gallery-instance orders or identify all tail swaps. No new NN/weight/seed/parameter selection.',
        completed_at=datetime.now().astimezone().isoformat())
    args.output_dir.mkdir();(args.output_dir/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

import base64
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
assert not (w/'SEED43_MSVR_GOLD_COLD_VERIFIED.json').exists()
observed=json.loads((w/'SEED43_MSVR_GOLD_OBSERVED.json').read_bytes())
assert observed['status']=='TERMINAL' and observed['exit']['exit_code']==0
assert observed['summary']['status']=='COMPLETE_ACTUAL_SAVED_DISTANCE_AUTHOR_EVALUATION'
code=r'''
import base64,hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools.analyze_correspondence_distances import scene_scores
source=root/'comparators/Signal-cd1b0a6';sys.path.insert(0,str(source))
from utils import metrics as upstream
assert Path(upstream.__file__).resolve()==(source/'utils/metrics.py').resolve()
torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=root/'refine-logs/seed43_msvr_gold_20261014_978';reg=json.loads((base/'ACTUAL_REGISTRATION.json').read_text())
assert reg['cached_metric_parity_acceptance_criterion'] is False
assert all(sha(root/name)==digest for name,digest in reg['source_sha256'].items())
row=reg['rows'][0]
assert all(sha(Path(name))==digest for name,digest in row['input_sha256'].items())
out=root/'results/seed43_msvr_gold978';summary_path=out/'SUMMARY.json';summary=json.loads(summary_path.read_text())
assert summary['registration_sha256']==sha(base/'ACTUAL_REGISTRATION.json')
assert summary['comparison_sha256']==sha(base/'ACTUAL_COMPARISON.json')
assert summary['driver_sha256']==reg['driver_sha256']==sha(root/'tools/evaluate_seed43_near_signal_residual_budget.py')
assert summary['model_state_sha256']==json.loads(Path(row['metrics_path']).read_text())['model_state_sha256']
assert summary['status']=='COMPLETE_ACTUAL_SAVED_DISTANCE_AUTHOR_EVALUATION'
assert summary['seed']==43 and summary['ratio']==.25 and summary['original_selected_epoch']==19
assert summary['state_equal'] and summary['all_parameter_grads_none'] and summary['independent_upstream_metrics_equal']
assert summary['verification_attempt']==2 and summary['prior_failed_full_query_gallery_encodings']==1
assert summary['all_reported_metrics_strictly_above_signal'] and summary['old_cached_metric_parity_pass'] is False
distance_path=Path(summary['distance_path']);assert sha(distance_path)==summary['distance_sha256']
data=torch.load(distance_path,map_location='cpu',weights_only=False)
control=torch.load(row['distance_path'],map_location='cpu',weights_only=False)
for key in ('protocol_sha256','query_ids','gallery_ids','query_cameras','gallery_cameras','query_scenes','gallery_scenes'):
    assert np.array_equal(data[key],control[key]),key
d=data['distances']['adapted'];assert tuple(d.shape)==(591,1055)
qids,gids=data['query_ids'],data['gallery_ids'];qc,gc=data['query_cameras'],data['gallery_cameras'];qs,gs=data['query_scenes'],data['gallery_scenes']
actual=scene_scores(d.numpy(),qids,gids,qs,gs)
baseline=scene_scores(control['distances']['baseline_only'].numpy(),qids,gids,qs,gs)
diagnosis_out=root/'results/seed43_msvr_gold_cold979';assert not diagnosis_out.exists();diagnosis_out.mkdir();os.chdir(diagnosis_out)
cmc,map_value=upstream.eval_func_msrv(d.numpy(),qids,gids,qc,gc,qs,gs)
assert abs(actual['metrics']['mAP']-map_value*100)<1e-10
assert all(abs(actual['metrics'][f'Rank-{k}']-cmc[k-1]*100)<1e-5 for k in (1,5,10))
assert all(abs(value-summary['outputs']['adapted']['metrics'][key])<1e-10 for key,value in actual['metrics'].items())
assert np.array_equal(actual['first_match_rank'],summary['outputs']['adapted']['first_match_rank'])
assert np.allclose(actual['average_precision'],summary['outputs']['adapted']['average_precision'],rtol=0,atol=1e-12)
first=np.asarray(actual['first_match_rank']);old=np.asarray(baseline['first_match_rank'])
ap_delta=np.asarray(actual['average_precision'])-np.asarray(baseline['average_precision'])
repairs=np.flatnonzero((old!=1)&(first==1));harms=np.flatnonzero((old==1)&(first!=1))
examples=[]
for q in harms:
    order=np.argsort(d[q].numpy());kept=order[~((gids[order]==qids[q])&(gs[order]==qs[q]))];negative=int(kept[0])
    examples.append(dict(query_index=int(q),identity=int(qids[q]),scene=int(qs[q]),new_first_positive_rank=int(first[q]),
        first_wrong_identity=int(gids[negative]),first_wrong_scene=int(gs[negative]),same_scene_wrong_identity=bool(gs[negative]==qs[q])))
record=dict(status='COLD_SAVED_ACTUAL_DISTANCE_AND_AUTHOR_METRICS_VERIFIED',dataset='MSVR310',seed=43,selected_epoch=19,ratio=.25,
    actual_metrics=actual['metrics'],signal_metrics=baseline['metrics'],delta_pp={key:value-baseline['metrics'][key] for key,value in actual['metrics'].items()},
    all_four_strictly_above_signal=all(value>baseline['metrics'][key] for key,value in actual['metrics'].items()),
    rank1_repairs=len(repairs),rank1_new_errors=len(harms),rank1_harm_examples=examples,
    ap_improved_queries=int((ap_delta>1e-12).sum()),ap_declined_queries=int((ap_delta<-1e-12).sum()),
    source_seal_count=len(reg['source_sha256']),input_sha256=row['input_sha256'],actual_distance_sha256=sha(distance_path),
    actual_distance_bytes=distance_path.stat().st_size,actual_summary_sha256=sha(summary_path),author_metric_source_sha256=sha(source/'utils/metrics.py'),
    control_distance_max_difference=summary['control_distance_max_difference'],adapted_cached_distance_max_difference=summary['adapted_cached_distance_max_difference'],
    normalized_block_checks=summary['block_checks'],cached_metric_delta_pp=summary['cached_metric_delta_pp'],old_cached_metric_parity_pass=False,
    nn_calls=0,optimizer_updates=0,gpu_queries=0,other_running_dataset_queries=0,cumulative_actual_full_encodings=2,
    completed_at=datetime.now().astimezone().isoformat(),
    boundary='Independent subsequent CPU reload/hash/full-author-scoring of the saved actual distance under the explicitly preregistered corrected criterion. The original cache-parity run remains FAILED. Fixed complete epoch19/seed43/ratio.25 endpoint, no seed mean or metric-wise mixture. Original-Signal plus extra role-training and official benchmark selection disclosed; not own-module causal gain or unbiased SOTA.')
assert record['all_four_strictly_above_signal'] and record['rank1_repairs']==10 and record['rank1_new_errors']==7
assert all(sha(root/name)==digest for name,digest in reg['source_sha256'].items())
target=diagnosis_out/'VERIFIED.json';target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(record=record,files={str(p.relative_to(root)):dict(sha256=sha(p),data=base64.b64encode(p.read_bytes()).decode()) for p in (target,distance_path)})))
'''
run=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code.encode(),capture_output=True)
(w/'SEED43_MSVR_GOLD_COLD.stdout').write_bytes(run.stdout);(w/'SEED43_MSVR_GOLD_COLD.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);dest=w/'seed43_msvr_gold_cold';dest.mkdir()
for name,item in value.pop('files').items():
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
    path=dest/name;assert path.resolve().is_relative_to(dest.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'SEED43_MSVR_GOLD_COLD_VERIFIED.json').write_text(json.dumps(value['record'],indent=2)+'\n',encoding='utf-8')
print(json.dumps(value['record']))

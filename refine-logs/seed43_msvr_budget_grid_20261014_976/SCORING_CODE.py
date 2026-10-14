import base64
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
registration_name='refine-logs/seed43_msvr_terminal_20261014_975/BUDGET_REGISTRATION.json'
registration_sha=hashlib.sha256((r/registration_name).read_bytes()).hexdigest()
assert not (w/'SEED43_MSVR_BUDGETS.json').exists()
code='registration_name='+repr(registration_name)+'\nregistration_sha='+repr(registration_sha)+'\n'+r'''
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'modeling'))
from trifusion.residual_budget import cached_budget_distance
from tools.analyze_correspondence_distances import scene_scores
from tools.compare_near_signal_residual_budgets import component_check
torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=root/registration_name;assert sha(path)==registration_sha
reg=json.loads(path.read_text());assert reg['status']=='REGISTERED_BEFORE_SEED43_BUDGET_SCORING'
assert reg['dataset']=='MSVR310' and reg['seed']==43 and reg['ratios']==[.25,.5,.75]
assert all(sha(root/name)==digest for name,digest in reg['source_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in reg['input_sha256'].items())
receipt=json.loads(Path(reg['metrics_path']).read_text())
assert receipt['status']=='COMPLETE' and receipt['seed']==43 and receipt['training_epochs']==50
data=torch.load(reg['distance_path'],map_location='cpu',weights_only=False)
score=lambda d:scene_scores(d.numpy(),data['query_ids'],data['gallery_ids'],data['query_scenes'],data['gallery_scenes'])
baseline=score(data['distances']['baseline_only']);original=score(data['distances']['fused'])
for name,value in (('baseline_only',baseline),('fused',original)):
    assert all(abs(value['metrics'][key]-receipt['outputs'][name]['metrics'][key])<1e-5 for key in value['metrics'])
rank=np.asarray(baseline['first_match_rank']);candidates=[]
for ratio in reg['ratios']:
    d=cached_budget_distance(data['distances']['baseline_only'].double(),data['distances']['fused'].double(),ratio)
    values=score(d);candidate_rank=np.asarray(values['first_match_rank'])
    candidates.append(dict(ratio=ratio,metrics=values['metrics'],
        delta_vs_signal={key:value-baseline['metrics'][key] for key,value in values['metrics'].items()},
        rank1_repairs=int(((rank!=1)&(candidate_rank==1)).sum()),rank1_new_errors=int(((rank==1)&(candidate_rank!=1)).sum()),
        all_reported_metrics_strictly_above_signal=all(value>baseline['metrics'][key]+1e-10 for key,value in values['metrics'].items()),
        average_precision=values['average_precision'],first_match_rank=values['first_match_rank']))
selected=max(candidates,key=lambda v:(v['all_reported_metrics_strictly_above_signal'],v['metrics']['mAP'],v['metrics']['Rank-1'],v['ratio']))
assert all(sha(root/name)==digest for name,digest in reg['source_sha256'].items())
result=dict(status='COMPLETE_PROVISIONAL_SEED43_MSVR_CACHED_BUDGET_GRID',dataset='MSVR310',seed=43,selected_epoch=receipt['selected_epoch'],
    signal_metrics=baseline['metrics'],original_r2_metrics=original['metrics'],candidates=candidates,
    selected_ratio=selected['ratio'],selected_metrics=selected['metrics'],selected_all_four_pass=selected['all_reported_metrics_strictly_above_signal'],
    component_checks=component_check(),registration_sha256=registration_sha,at=datetime.now().astimezone().isoformat(),
    nn_calls=0,optimizer_updates=0,gpu_queries=0,other_running_dataset_queries=0,
    boundary='Three preregistered original ratios only, one newly closed seed43 whole checkpoint. All candidates disclosed; cached values provisional pending unchanged strict actual verifier. Not an independent validation set or causal module gain.')
out=root/'results/seed43_msvr_budget_grid976';assert not out.exists();out.mkdir()
(out/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,allow_nan=False))
'''
run=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code.encode(),capture_output=True)
(w/'SEED43_MSVR_BUDGETS.stdout').write_bytes(run.stdout);(w/'SEED43_MSVR_BUDGETS.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
result=json.loads(run.stdout)
(w/'SEED43_MSVR_BUDGETS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key!='candidates'}))
print(json.dumps([dict(ratio=row['ratio'],metrics=row['metrics'],repairs=row['rank1_repairs'],harms=row['rank1_new_errors'],all_four=row['all_reported_metrics_strictly_above_signal']) for row in result['candidates']]))

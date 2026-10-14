import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
assert not (w/'TOP50_AP_CONTRIBUTION_OBSERVED.json').exists()
failed=(w/'budget_fixed_actual_observation/logs/near_signal_residual_budget_actual966/MSVR310/stderr').read_text(encoding='utf-8')
line=next(x.removeprefix('AssertionError: ') for x in failed.splitlines() if x.startswith("AssertionError: ('adapted',"))
name,actual,cached=ast.literal_eval(line);assert name=='adapted'
diagnosis=json.loads((w/'rank_precision/results/msvr_cached_rank_precision969/SUMMARY.json').read_bytes())
registration=json.loads((r/'refine-logs/near_signal_residual_budget_20261014_963/REGISTRATION.json').read_bytes())
row=next(x for x in registration['rows'] if x['dataset']=='MSVR310')
request=dict(distance_path=row['distance_path'],distance_sha256=row['input_sha256'][row['distance_path']],
    diagnosis=diagnosis,actual=actual,cached=cached,traceback_sha256=hashlib.sha256(failed.encode()).hexdigest())
code='request='+repr(request)+'\n'+r'''
from datetime import datetime
import hashlib,json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
path=Path(request['distance_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==request['distance_sha256']
data=torch.load(path,map_location='cpu',weights_only=False)
qids,gids=data['query_ids'],data['gallery_ids'];qs,gs=data['query_scenes'],data['gallery_scenes']
def partial(ranks,total):return sum((i+1)/rank for i,rank in enumerate(ranks))/total
rows=[]
for change in request['diagnosis']['author_top50']['adapted']['changes']:
    index=change['query_index'];total=int(((gids==qids[index])&(gs!=qs[index])).sum())
    actual=partial(change['actual_positive_ranks'],total);cached=partial(change['cached_positive_ranks'],total)
    rows.append(dict(query_index=index,identity=int(qids[index]),scene=int(qs[index]),legal_positive_count=total,
        actual_positive_ranks=change['actual_positive_ranks'],cached_positive_ranks=change['cached_positive_ranks'],
        actual_top50_ap_contribution=actual,cached_top50_ap_contribution=cached,
        delta_ap_top50=actual-cached,delta_mAP_top50=(actual-cached)*100/len(qids)))
known=sum(row['delta_mAP_top50'] for row in rows)
observed=request['actual']['mAP']-request['cached']['mAP']
result=dict(status='COMPLETE_SAVED_TOP50_AP_CONTRIBUTION_ANALYSIS',query_count=len(qids),changes=rows,
    known_top50_delta_mAP=known,observed_full_delta_mAP=observed,unexplained_aggregate_delta_mAP=observed-known,
    actual_observed_metrics=request['actual'],cached_metrics=request['cached'],input_distance_sha256=request['distance_sha256'],
    source_traceback_sha256=request['traceback_sha256'],rank_text_sha256=request['diagnosis']['rank_text_sha256'],
    nn_calls=0,optimizer_updates=0,completed_at=datetime.now().astimezone().isoformat(),
    boundary='Partial AP contributions use all legal-positive denominators and original591-query mean. Only saved top50 is known; unexplained remainder cannot locate individual tail swaps or prove a unique floating-point cause. Original metric-parity failure remains failed; no tolerance, ranking parameter or accepted endpoint changes.')
print(json.dumps(result,allow_nan=False))
'''
p=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code.encode(),capture_output=True)
(w/'TOP50_AP_CONTRIBUTION.stdout').write_bytes(p.stdout);(w/'TOP50_AP_CONTRIBUTION.stderr').write_bytes(p.stderr)
assert p.returncode==0,p.stderr.decode()
result=json.loads(p.stdout)
(w/'TOP50_AP_CONTRIBUTION_OBSERVED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))

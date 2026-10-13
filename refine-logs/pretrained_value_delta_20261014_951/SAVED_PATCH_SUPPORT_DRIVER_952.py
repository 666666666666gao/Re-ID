"""Describe saved local-only disagreements with the preserved global ranking."""
import hashlib,json
from pathlib import Path

w=Path(__file__).resolve().parent
source=w/'native_patch950/results/native_clip_patch950/SUMMARY.json'
value=json.loads(source.read_bytes());assert value['status']=='COMPLETE_FROZEN_PATCH_SOURCE_DIAGNOSIS'
g=value['global_score'];rows=[]
for name,item in value['scores'].items():
    for method,score in item['scores'].items():
        assert len(score['average_precision'])==len(g['average_precision'])==836
        repairs=sum(a!=1 and b==1 for a,b in zip(g['first_match_rank'],score['first_match_rank']))
        harms=sum(a==1 and b!=1 for a,b in zip(g['first_match_rank'],score['first_match_rank']))
        delta=[(b-a)*100 for a,b in zip(g['average_precision'],score['average_precision'])]
        rows.append(dict(source=name,method=method,local_metrics=score['metrics'],
            delta_mAP_from_global=sum(delta)/836,rank1_repairs=repairs,rank1_new_errors=harms,
            positive_query_ap_changes=sum(x>1e-10 for x in delta),negative_query_ap_changes=sum(x<-1e-10 for x in delta)))
result=dict(status='COMPLETE_SAVED_PATCH_SUPPORT_DESCRIPTION',input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    rows=rows,neural_forwards=0,optimizer_updates=0,
    boundary='Local-only relations versus G, not fused scores or selected query gates/teacher targets. No fusion coefficient or new ranking search.')
path=w/'PATCH_SUPPORT_DESCRIPTION_952.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

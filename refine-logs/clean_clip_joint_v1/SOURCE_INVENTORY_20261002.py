"""Training-file inventory only; not sampled-batch or meta-learning evidence."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/gaob/Re-ID/Trifusion')
FOLDER=ROOT/'logs/official_three_dataset_protocols_20260923'
OUTPUT=ROOT/'.codex_tmp/train_environment_support_20261002.json'
assert not OUTPUT.exists()
result={'observed_at':datetime.now().astimezone().isoformat(),'scope':'Actual training record inventory only; no sampling, neural call, score or meta update. Environment is camera for201/100 and original scene/time block forMSVR310.','datasets':{}}
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
    path=FOLDER/f'{dataset}.json'
    protocol=json.loads(path.read_text())
    rows=protocol['records']['train']
    field='scene' if dataset=='MSVR310' else 'camera'
    identities=defaultdict(Counter)
    env=defaultdict(Counter)
    for row in rows:
        identities[row['identity']][row[field]]+=1
        env[row[field]][row['identity']]+=1
    eligible_ids={identity for identity,counts in identities.items() if len(counts)>1}
    eligible_anchors=sum(sum(identities[identity].values()) for identity in eligible_ids)
    pairs=[]
    for a,ac in sorted(env.items()):
        for b,bc in sorted(env.items()):
            if a==b:continue
            supported=set(ac)&set(bc)
            supported={identity for identity in supported if len(set(bc)-{identity})>0}
            pairs.append({'A':a,'B':b,'identities_with_legal_cross_environment_positive_and_B_negative':len(supported),
                          'A_anchors_with_such_dataset_support':sum(ac[i] for i in supported)})
    result['datasets'][dataset]={'protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'environment_field':field,'training_records':len(rows),'training_identities':len(identities),
        'environments':{str(k):{'records':sum(v.values()),'identities':len(v)} for k,v in sorted(env.items())},
        'identities_with_multiple_environments':len(eligible_ids),'records_of_these_identities':eligible_anchors,
        'identity_environment_count_distribution':dict(sorted(Counter(len(v) for v in identities.values()).items())),
        'ordered_environment_pair_support':pairs,
        'boundary':'Dataset-level possibilities only; does not show actual PLAIN_V8 batches or identity availability in a specific meta split.'}
OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(OUTPUT),'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
       'inventory':{name:{k:row[k] for k in ('training_records','training_identities','identities_with_multiple_environments','records_of_these_identities')} for name,row in result['datasets'].items()}}))

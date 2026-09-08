import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
sources={}
for name in ('remote_inventory.stdout','remote_provenance.stdout'):
    x=json.loads((O/name).read_bytes());sources.update({p: x['files'][p] for p in x['texts']})
proof=[]
for path,item in sources.items():
    dest=O/'snapshots'/'remote'/path.lstrip('/')
    actual=hashlib.sha256(dest.read_bytes()).hexdigest()
    proof.append({'source':path,'snapshot':str(dest.relative_to(O)),'source_sha256':item['sha256'],'snapshot_sha256':actual,'byte_exact':actual==item['sha256']})
(O/'snapshot_validation.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps({'snapshots':len(proof),'mismatches':[p for p in proof if not p['byte_exact']]},indent=2))

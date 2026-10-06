from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
c=root/'logs/independent_role_heads_v1_20261006_873'
j=root/'logs/independent_role_heads_launch_20261006_873'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
state=json.loads((c/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert json.loads((j/'EXIT.json').read_text())['exit_code']==0
assert not (Path('/proc')/str(state['controller_pid'])).exists()
old_path=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(old_path)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
old=json.loads(old_path.read_text())
artifacts=dict(old['artifact_sha256'])
assert len(artifacts)==187 and all(sha(Path(n))==d for n,d in artifacts.items())
parent_path=root/'refine-logs/independent_role_heads_v1/SOURCE_SCOPE.json'
parent=json.loads(parent_path.read_text())
assert len(parent['source_sha256'])==385
assert all(sha(root/n)==d for n,d in parent['source_sha256'].items())
matrix=json.loads((c/'accepted_matrix.json').read_text())
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==3
rows=list(matrix['rows'])
for r in matrix['rows']:
 acceptance_path=c/'acceptance'/f"{r['dataset']}_semantic.json"
 acceptance=json.loads(acceptance_path.read_text())
 assert acceptance['row']==r and acceptance['status']=='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT'
 assert all(sha(Path(n))==d for n,d in acceptance['artifact_sha256'].items())
 assert not Path(acceptance['probe']['path']).exists()
 artifacts.update(acceptance['artifact_sha256'])
 artifacts[str(acceptance_path)]=sha(acceptance_path)
 control=next(x for x in old['rows'] if (x['dataset'],x['variant'])==(r['dataset'],'global_only'))
 rows.append(control)
 assert control['initializer']['protocol_sha256']==r['initializer']['protocol_sha256']
for p in (c/'manifest.json',c/'campaign.json',c/'accepted_matrix.json',c/'probe_retirement.jsonl',j/'EXIT.json',
          old_path,parent_path,root/'results/independent_role_heads_v1_20261006_873/SUMMARY.json'):
 artifacts[str(p)]=sha(p)
for n,d in json.loads((c/'manifest.json').read_text())['initialization_sha256'].items():
 assert sha(Path(n))==d;artifacts[n]=d
for r in matrix['rows']:
 p=root/'logs/training_feature_scale_protocols_20261002'/f"{r['dataset']}.json"
 assert sha(p)==r['initializer']['protocol_sha256'];artifacts[str(p)]=sha(p)
clip=root/'pertrained-model/ViT-B-16.pt'
assert sha(clip)=='5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
artifacts[str(clip)]=sha(clip)
assert {(r['dataset'],r['variant']) for r in rows}=={(d,v) for d in ('RGBNT201','MSVR310','RGBNT100') for v in ('semantic','global_only')}
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows,artifact_sha256=artifacts,
 parent_source_sha256=parent['source_sha256'],parent_source_scope_sha256=sha(parent_path),
 free_bytes=shutil.disk_usage(root).free,nn_forwards=0,optimizer_updates=0,
 boundary='Read-only seal construction from accepted original artifacts after NN terminal. No source deployment, model, scoring, update or deletion.')))

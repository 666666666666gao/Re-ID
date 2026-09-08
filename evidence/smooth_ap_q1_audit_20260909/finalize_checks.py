from pathlib import Path
from collections import Counter
import json,hashlib,shutil
OUT=Path(__file__).parent
REPO=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (OUT/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
remote={}
for name,p in [('arrays','remote_array_checks_attempt2.result.jsonl'),('statistics','remote_statistics_checks.result.jsonl')]:
 rows=[json.loads(line) for line in (OUT/p).read_text(encoding='utf8').splitlines()]
 result=rows[-1];assert result['status']=='PASS';write('independent_remote_'+name+'_results.json',result);remote[name]=result
identity=[]
for row in json.loads((OUT/'prior_snapshot_identity.json').read_bytes()):
 a=Path(row['path']).read_bytes();b=Path(row['snapshot']).read_bytes()
 kind='exact' if a==b else 'newline_only' if a.replace(b'\r\n',b'\n')==b.replace(b'\r\n',b'\n') else 'content_change'
 identity.append(dict(path=row['path'],snapshot=row['snapshot'],kind=kind,current_sha256=hashlib.sha256(a).hexdigest(),snapshot_sha256=hashlib.sha256(b).hexdigest()))
write('snapshot_difference_classification.json',identity)
der=json.loads(Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_positive_derivative_complete_20260909\result.json').read_bytes())
differences=[];groups=0
for r,ref in zip(remote['statistics']['endpoints'],der['endpoints']):
 assert r['endpoint']==ref['endpoint'] and r['steps']==ref['steps']==260
 for phase,cats in r['phases'].items():
  for cat,count in cats.items():
   target=ref['phases'][phase][cat]['smooth_distance_derivative'];assert count['count']==target['count'];groups+=1
   for k in ['negative','zero','positive']:
    if count[k]!=target[k]:differences.append(dict(endpoint=r['endpoint'],phase=phase,category=cat,key=k,numpy_float64=count[k],executor_float32=target[k]))
write('independent_derivative_census_comparison.json',dict(count_groups=groups,sign_count_differences=differences,scope='Float64 analytic counts versus executor Float32 autograd; distance objective equivalence independently checked separately for every position in both dtypes.'))
inputs=json.loads((OUT/'input_hashes_initial.json').read_bytes())
for p in REPO.glob('results/MSVR310_SMOOTH_AP_V1_Q1_2026-09-09.md'):inputs[str(p)]=None
for base in [Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_positive_derivative_complete_20260909'),Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_ranking_analysis_20260909')]:
 for p in base.rglob('*'):
  if p.is_file():inputs[str(p)]=None
for name in ['smooth_ap_q1_source_log_analysis_20260909.json','smooth_ap_q1_aggregate_source_20260909.json','smooth_ap_positive_derivative_attempt1_failure_20260909.md','smooth_ap_positive_derivative_durable_observation1_20260909.json']:
 p=Path(r'C:\Users\gb\.codex_tmp')/name
 if p.exists():inputs[str(p)]=None
final={};drift=[]
for name,old in inputs.items():
 p=Path(name);actual=dict(bytes=p.stat().st_size,sha256=sha(p));final[name]=actual
 if old is not None and old!=actual:drift.append(dict(path=name,before=old,after=actual))
 # Preserve all Q1 primary text and current contract/source files; no remote binary copied.
 relative=Path('repo')/p.relative_to(REPO) if p.is_relative_to(REPO) else Path('local_text')/p.relative_to(Path(r'C:\Users\gb\.codex_tmp'))
 target=OUT/'snapshots'/relative;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();target.write_bytes(p.read_bytes())
write('input_hashes_final.json',final);write('input_drift_during_audit.json',drift)
small={k:remote['arrays'][k] for k in ['status','elapsed_seconds','source_path_inventory','source_checks','model_forwards','optimizer_updates','image_reads','torch_version','numpy_version']}
small['endpoints']=[{k:v for k,v in r.items() if k!='retrieval'} for r in remote['arrays']['endpoints']]
small['retrieval_numeric']=[dict(fold=r['fold'],endpoint=r['endpoint'],retrieval=r['retrieval']) for r in remote['arrays']['endpoints']]
small['snapshots']=dict(Counter(r['kind'] for r in identity));small['input_drift']=drift;small['derivative_census_sign_differences']=differences
write('verification_compact_summary.json',small)
print(json.dumps(small,indent=2))

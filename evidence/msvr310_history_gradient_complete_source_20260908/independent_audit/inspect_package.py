from pathlib import Path
import json, hashlib
R=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
T=Path('C:/Users/gb/.codex_tmp')
S=T/'history_gradient_complete_source_20260908'
for p in [R/'protocols/msvr310_train_oof_v1.json', S/'source/summary.json',S/'source/cpu_verification.json',T/'history_gradient_complete_analysis_20260908/complete_text_reaggregation.json',T/'history_gradient_complete_processing_20260908/all_statistics_postcheck.json']:
 d=json.loads(p.read_bytes());print(p.name, 'KEYS',list(d));print('SMALL', {k:v for k,v in d.items() if not isinstance(v,(list,dict))})
 if 'records' in d: print('RECORD',d['records'][0]); print('FOLD KEYS',list(d['folds'][0]))
 if 'states' in d:print('STATE',str(d['states'][0])[:3000])
 if 'environment' in d:print('ENV',d['environment'])
p=S/'source/fold_0_initial/steps.jsonl'
rows=[json.loads(x) for x in p.read_text().splitlines()]
print('ROW 67',json.dumps(rows[66],indent=2)[-6500:])
configs=['TriFusion-history-candidate-gradient-v1.json','TriFusion-fresh-coordinate-paired-v1.json','TriFusion-instance-memory-paired-v1.json','TriFusion-source-style-paired-v1-r2.json','Signal-source-oof-v1.json']
out=[]
for n in configs:
 p=R/'configs/MSVR310'/n;d=json.loads(p.read_bytes())
 for key in ['project_file_sha256','project_source_file_sha256']:
  for name,expected in d.get(key,{}).items():
   q=R/name;b=q.read_bytes();actual=hashlib.sha256(b).hexdigest();lf=hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()
   out.append(dict(config=n,path=name,expected=expected,actual=actual,exact=actual==expected,lf_match=lf==expected))
print('BINDINGS',len(out),'EXACT',sum(x['exact'] for x in out));print('MISMATCH',json.dumps([x for x in out if not x['exact']],indent=2))
(Path(__file__).parent/'project_binding_checks.json').write_text(json.dumps(out,indent=2)+'\n')

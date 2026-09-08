from pathlib import Path
import json,hashlib
out=Path(__file__).parent
repo=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
snap=repo/'evidence/smooth_ap_m0_audit_20260909/snapshots/remote'
checks={('/'+p.relative_to(snap).as_posix()):hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file() and '/TriFusion-ReID/' in p.as_posix() and '/refine-logs/' not in p.as_posix()}
body='''import os,sys,json,hashlib,subprocess,datetime
from pathlib import Path
sys.dont_write_bytecode=True
root=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
result={'observed_at':datetime.datetime.now().astimezone().isoformat(),'model_forwards':0,'optimizer_updates':0,'official_image_reads':0,'snapshot_identity':[]}
for name,digest in expected.items():
 p=Path(name);actual=sha(p) if p.is_file() else None
 result['snapshot_identity'].append(dict(path=name,expected=digest,actual=actual,match=actual==digest))
summary=json.loads((run/'q1/summary.json').read_bytes())
result['summary_sha256']=sha(run/'q1/summary.json')
result['processes']={str(p):Path('/proc/'+str(p)).exists() for p in [48170,49157,58836]}
result['head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
result['file_tree']=[str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()]
result['config_chain']={}
p=root/'configs/MSVR310/TriFusion-smooth-ap-paired-v1.json'
while True:
 c=json.loads(p.read_bytes());result['config_chain'][str(p)]=c
 nextkey=next((k for k in ['previous_config','coordinate_config','memory_config','base_config'] if k in c),None)
 if nextkey is None:break
 p=root/c[nextkey]
result['source_config']=c
result['base_config']=json.loads((root/c['BASELINE']['CONFIG']).read_bytes())
result['source_metadata_path']=str(root/c['SOURCE_METADATA']['PATH'])
result['source_baseline_summary_path']=c['BASELINE']['SUMMARY']
result['baseline_summary']=json.loads(Path(c['BASELINE']['SUMMARY']).read_bytes())
result['source_metadata_schema']={k:len(v) if isinstance(v,(list,dict)) else v for k,v in json.loads((root/c['SOURCE_METADATA']['PATH']).read_bytes()).items()}
result['training_example']=summary['folds'][0]['endpoints']['control']['training']['steps'][0]
result['retrieval_keys']=list(summary['folds'][0]['endpoints']['control']['retrieval'])
result['project_pins']=[]
for cfgname,cfg in result['config_chain'].items():
 for key in ['project_file_sha256','project_source_file_sha256','fixed_file_sha256']:
  for name,digest in cfg.get(key,{}).items():
   p=Path(name) if name.startswith('/') else root/name
   actual=sha(p)
   result['project_pins'].append(dict(config=cfgname,path=str(p),expected=digest,actual=actual,match=actual==digest))
result['commits']={}
tracked=['configs/MSVR310/TriFusion-smooth-ap-paired-v1.json','tools/msvr_smooth_ap.py','tools/train_msvr_smooth_ap.py','tools/verify_msvr_smooth_ap.py','tools/run_msvr_smooth_ap.py']
for commit in ['2e947a4325144e37fed638105ac954e7e54b5fe5',summary['project_commit'],result['head']]:
 result['commits'][commit]={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=root)).hexdigest() for n in tracked}
print(json.dumps(result,ensure_ascii=False))
'''
(out/'remote_probe.py').write_text('expected='+repr(checks)+'\n'+body,encoding='utf8',newline='\n')
print(len(checks))

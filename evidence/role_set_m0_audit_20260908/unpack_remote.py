"""Extract remote text snapshots and compare their byte manifests independently."""
import ast
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
remote=json.loads((OUT/'remote_inventory.stdout.txt').read_bytes())
local=json.loads((OUT/'local_inventory.json').read_bytes())
for path,content in remote.pop('texts').items():
    target=OUT/'snapshots/remote'/path.lstrip('/')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(content,encoding='utf-8',newline='')
intake=json.loads((OUT/'snapshots/project/evidence/msvr310_role_set_m0_complete_20260908/intake_manifest.json').read_bytes())
checks=[]
for item in intake['files']:
    actual=remote['files'][intake['remote']+'/'+item['path']]
    checks.append({'path':item['path'],'sha_match':actual['sha256']==item['sha256'],'bytes_match':actual['bytes']==item['bytes']})
remote['intake_comparison']=checks
differences=[]
for path,item in local['hashes'].items():
    if not path.startswith('C:\\Users\\gb\\.trifusion_github_publish_22c3bee\\'):continue
    relative=path.split('.trifusion_github_publish_22c3bee\\',1)[1].replace('\\','/')
    remote_path='/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'+relative
    if remote_path in remote['files'] and item['sha256']!=remote['files'][remote_path]['sha256']:
        a=(OUT/item['snapshot']).read_bytes();b=(OUT/'snapshots/remote'/remote_path.lstrip('/')).read_bytes()
        differences.append({'path':relative,'local_sha256':item['sha256'],'remote_sha256':remote['files'][remote_path]['sha256'],'LF_normalized_equal':a.replace(b'\r\n',b'\n')==b.replace(b'\r\n',b'\n'),'ast_equal':ast.dump(ast.parse(a.decode('utf-8-sig')))==ast.dump(ast.parse(b.decode('utf-8-sig'))) if relative.endswith('.py') else None})
remote['local_remote_differences']=differences
(OUT/'remote_inventory.json').write_text(json.dumps(remote,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'remote_files':len(remote['files']),'bindings':len(remote['bindings']),'binding_failures':[x for x in remote['bindings'] if not x['pass']],'intake_all_match':all(x['sha_match'] and x['bytes_match'] for x in checks),'local_remote_differences':differences,'original_processes':remote['original_processes'],'dataset_root_metadata_files':remote['dataset_root_metadata_files']},ensure_ascii=False,indent=2))

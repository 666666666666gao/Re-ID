import json, hashlib, subprocess
from pathlib import Path
from inspect_inputs import OUT, REPO, record

data=json.loads(record(OUT/'remote_sources.stdout.json'))
mapping={}
for remote,item in data['documents'].items():
 rel=Path(remote).relative_to('/root/autodl-tmp/trifusion-v2')
 path=OUT/'remote_sources'/rel
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes(item['text'].encode('utf-8'))
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 mapping[remote]=str(path)
(OUT/'remote_source_path_map.json').write_text(json.dumps(mapping,indent=2)+'\n')
remote=json.loads(record(OUT/'remote_inventory.stdout.json'))
manifest=json.loads(record(Path('C:/Users/gb/.codex_tmp/history_gradient_q1_complete_20260908/intake_manifest.json')))
intake=[]
for r in manifest['files']:
 name=manifest['run']+'/'+r['path'];wanted=remote['read_hashes'].get(name,remote['files'].get(name))
 assert wanted=={k:r[k] for k in ('bytes','sha256')};intake.append(name)
code=[]
for name,r in remote['project_files'].items():
 p=REPO/name
 if not p.is_file():continue
 b=record(p);code.append(dict(path=name,local_sha256=hashlib.sha256(b).hexdigest(),remote_sha256=r['sha256'],raw_match=hashlib.sha256(b).hexdigest()==r['sha256'],lf_match=hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()==r['lf_sha256'],execution_commit_match=r['execution_commit_match']))
initial=remote['pipeline']['code_commit'];q1=json.loads(record(Path('C:/Users/gb/.codex_tmp/history_gradient_q1_complete_20260908/q1/summary.json')))['project_commit']
diff=subprocess.check_output(['git','-C',str(REPO),'diff','--name-status',initial,q1],text=True)
result=dict(local_remote_intake_files_equal=len(intake),live_remote_process_presence=remote['process_presence'],pipeline=remote['pipeline'],remote_head=remote['head'],remote_git_status=remote['status'],project_file_checks=code,local_lf_mismatches=[r for r in code if not r['lf_match']],execution_commit_changed_paths=[r['path'] for r in code if not r['execution_commit_match']],q1_project_commit=q1,execution_to_q1_commit_diff=diff,actual_signal_head=data['signal_head'],actual_signal_diff_sha256=data['signal_diff_sha256'])
(OUT/'provenance_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('project_file_checks','pipeline')},indent=2))

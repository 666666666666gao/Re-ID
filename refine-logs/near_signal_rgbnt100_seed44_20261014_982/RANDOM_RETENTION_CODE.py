import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
assert not (w/'SEED44_RANDOM_RETENTION.json').exists()
result=next(row for row in json.loads(Path('C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_conditioned_evidence_best930/INPUT_SEAL.json').read_bytes())['rows'] if row['exposure']=='random')
request=dict(path=result['run_dir']+'/best_map.pth',sha256=result['checkpoint_sha256'],metrics=result['metrics'])
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
code='request='+repr(request)+'\n'+r'''
import hashlib,json
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');p=Path(request['path']).resolve()
assert p==root/'trained-model/global_conditioned_evidence_20261013_926_full_random_RGBNT201/best_map.pth'
campaign=json.loads((root/'logs/global_conditioned_evidence_full_20261013_926/campaign.json').read_text());assert campaign['status']=='COMPLETE_TWO_FULL50_FIRST_STRICT'
assert json.loads((root/'logs/global_conditioned_evidence_full_20261013_926/CONTROLLER_EXIT.json').read_text())['exit_code']==0
assert hashlib.sha256(p.read_bytes()).hexdigest()==request['sha256']
print(json.dumps(dict(**request,bytes=p.stat().st_size)))
'''
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code.encode(),capture_output=True)
assert run.returncode==0,run.stderr.decode();artifact=json.loads(run.stdout)
(w/'SEED44_RANDOM_ARCHIVE_PLAN.json').write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8')
target=Path('D:/Program Files/UserCache/gb/codex_tmp/trifusion_resume_20261014/closed_value_dependencies/closed_full_weights/global_conditioned_random926_best_map.pth').resolve()
assert target.drive=='D:' and not target.exists()
print(json.dumps(dict(status='COPYING_EXACT_CLOSED_BEST',bytes=artifact['bytes'],expected_seconds=180)),flush=True)
run=subprocess.run(['scp','-q',*options,'-P','2026','gaob@172.19.12.138:'+artifact['path'],str(target)],capture_output=True)
(w/'SEED44_RANDOM_COPY.json').write_text(json.dumps(dict(exit_code=run.returncode,stderr=run.stderr.decode()))+'\n',encoding='utf-8')
assert run.returncode==0,run.stderr.decode()
assert target.stat().st_size==artifact['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==artifact['sha256']
artifact['archive_path']=str(target)
(w/'SEED44_RANDOM_ARCHIVE_VERIFIED.json').write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8')
request=artifact
code='request='+repr(request)+'\n'+r'''
from datetime import datetime
import hashlib,json,shutil
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=Path(request['path']).resolve();assert p==root/'trained-model/global_conditioned_evidence_20261013_926_full_random_RGBNT201/best_map.pth'
assert p.is_relative_to(root/'trained-model')
state=json.loads((root/'logs/global_conditioned_evidence_full_20261013_926/campaign.json').read_text());assert state['status']=='COMPLETE_TWO_FULL50_FIRST_STRICT'
terminal=json.loads((root/'logs/global_conditioned_evidence_full_20261013_926/CONTROLLER_EXIT.json').read_text());assert terminal['exit_code']==0
assert json.loads((root/'logs/global_conditioned_evidence_best930/CONTROLLER_EXIT.json').read_text())['exit_code']==0
protected={}
for dataset,seed,label in (('RGBNT201',42,'horizon50_v1'),('RGBNT100',42,'horizon50_v1'),('MSVR310',43,'near_signal_followup_20261014_970')):
    path=root/f'trained-model/official_extra_seed{seed}_{dataset}_R2_{label}_bestmap_e50_20260926/{dataset}_R2_seed{seed}/roles_best_map.pth'
    metrics=json.loads((path.parent/'official_metrics.json').read_text());assert sha(path)==metrics['role_checkpoint_sha256']
    protected[str(path)]=metrics['role_checkpoint_sha256']
assert str(p) not in protected and sha(p)==request['sha256'] and p.stat().st_size==request['bytes']
before=shutil.disk_usage(root).free;p.unlink()
assert not p.exists() and all(sha(Path(name))==digest for name,digest in protected.items())
record=dict(status='EXACT_D_CLOSED_RANDOM_BEST_ARCHIVED_REMOTE_DUPLICATE_RETIRED',artifact=request,
    selected_strong_bests_sha256=protected,remote_free_before=before,remote_free_after=shutil.disk_usage(root).free,
    at=datetime.now().astimezone().isoformat(),nn_calls=0,optimizer_updates=0,gpu_queries=0,running_training_queries=0,
    restore_obligation='Restore this exact D best before old random926/diagnosis930 weight consumers. All original official distances/results/history and scientific old best retained. Active RGBNT100 seed43 files and selected three strong R2 bests untouched.')
(root/'logs/global_conditioned_evidence_full_20261013_926/BEST_RETENTION_FOR_SEED44.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code.encode(),capture_output=True)
(w/'SEED44_RANDOM_RETENTION.stdout').write_bytes(run.stdout);(w/'SEED44_RANDOM_RETENTION.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);(w/'SEED44_RANDOM_RETENTION.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value),flush=True)

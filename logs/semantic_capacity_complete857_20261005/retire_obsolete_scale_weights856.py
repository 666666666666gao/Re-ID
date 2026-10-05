"""Retire four completed, rejected scale candidates outside current dependencies."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'semantic_capacity_obsolete_scale_weight_retirement856'
inventory=json.loads((base/'semantic_capacity_obsolete_scale_weights856/stdout.json').read_bytes())
rows=[r for r in inventory['rows'] if r['dataset'] in ('RGBNT201','RGBNT100')
      and r['condition']['recipe'] in ('raw','metric_raw')]
assert len(rows)==4 and all(not r['protected_current_raw187'] for r in rows)
assert all(r['best_sha256']==r['checkpoint_receipt_sha256'] and r['epochs']==50
           and r['training_status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
           and r['evaluation_status']=='COMPLETE' and r['original_distance_present'] for r in rows)
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/obsolete_scale_weight_retirement_20261005_856'
assert not journal.exists()
dependencies=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
sources=json.loads((root/'refine-logs/semantic_capacity_control_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows={rows!r}
for row in rows:
 target=Path(row['best_path']);folder=target.parent
 assert target.resolve().is_relative_to((root/'trained-model').resolve()) and target.name=='best_map.pth'
 assert folder.name.startswith(('training_feature_scale_20261003_v2_full_raw_','metric_feature_scale_20261003_v1_full_metric_raw_'))
 assert str(target) not in dependencies and target.relative_to(root).as_posix() not in sources
 assert sha(target)==row['best_sha256'] and target.stat().st_size==row['best_bytes']
 assert sha(folder/'training.json')==row['training_sha256'] and sha(folder/'official_metrics.json')==row['evaluation_sha256']
 t=json.loads((folder/'training.json').read_text());e=json.loads((folder/'official_metrics.json').read_text())
 assert e['dataset'] in ('RGBNT201','RGBNT100') and e['condition']['recipe'] in ('raw','metric_raw')
 assert len(t['history'])==50 and e['status']=='COMPLETE' and t['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
 assert t['best_epoch']==e['selected_epoch'] and e['checkpoint_sha256']==row['best_sha256']
 assert sha(folder/'official_distances.pt')==e['distance_sha256']
 assert sha(folder/'best_epoch_distances.pt')==e['training_best_distance_sha256']
 row['retained_formal_distance_sha256']=e['distance_sha256']
 row['retained_best_epoch_distance_sha256']=e['training_best_distance_sha256']
 assert not Path('/proc/'+str(t.get('controller_pid',-1))).exists()
journal.mkdir();(journal/'PREPARE.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows),indent=2)+'\\n')
before=shutil.disk_usage(root).free
for row in rows:
 target=Path(row['best_path']);target.unlink();assert not target.exists()
 assert (target.parent/'official_distances.pt').is_file() and (target.parent/'official_metrics.json').is_file()
after=shutil.disk_usage(root).free
value=dict(status='FOUR_OBSOLETE_NEGATIVE_SCALE_BESTS_RETIRED',at=datetime.now().astimezone().isoformat(),
 rows=rows,retired_files=4,retired_bytes=sum(r['best_bytes'] for r in rows),disk_free_before=before,disk_free_after=after,
 boundary='Four finished F2/F3 raw or metric_raw candidates on201/100 rejected by primary mAP. Not required by current345/187. Original metrics/history/distances/SHA preserved; old weight-reload paths explicitly retired. No normalized controls, MSVR positive candidates, author weights, initialization or current best removed. Current NN/source/GPU untouched.')
(journal/'RETIREMENT.json').write_text(json.dumps(value,indent=2)+'\\n');print(json.dumps(value))
'''
compile(code,'retire_scale.py','exec')
assert not packet.exists();packet.mkdir();(packet/'REMOTE_SOURCE.py').write_bytes(code.encode())
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();stdout.channel.settimeout(300)
data,error=stdout.read(),stderr.read();status=stdout.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_bytes((json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n').encode())
assert status==0,error.decode();record=json.loads(data)
print(json.dumps({k:record[k] for k in ('status','at','retired_files','retired_bytes','disk_free_before','disk_free_after')}))

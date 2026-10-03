from pathlib import Path
from datetime import datetime
import json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
prior=json.loads((base/'owned_m0_probe_inventory777/stdout.json').read_bytes())
candidates=[r for r in prior['rows'] if Path(r['path']).parent.name.startswith('correspondence_context_identity_20260929_')]
assert len(candidates)==15
protected=json.loads((base/'closed_probe_inventory777/INVENTORY.json').read_bytes())['protected_sha256']
target=base/'closed_context_probe_inventory786'
assert not target.exists()
target.mkdir()
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
code=f'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion').resolve(strict=True)
campaign=root/'logs/correspondence_context_identity_20260929'
candidates={candidates!r}
protected={protected!r}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
state=json.loads((campaign/'campaign.json').read_text())
accepted=json.loads((campaign/'accepted_matrix.json').read_text())
assert state['status']=='COMPLETE'
assert accepted['expected_endpoints']==accepted['verified_complete']==len(accepted['rows'])==15
assert all(r['status']=='VERIFIED_COMPLETE' for r in accepted['rows'])
assert not (Path('/proc')/str(state['controller_pid'])).exists()
assert all(sha(Path(n))==h for n,h in protected.items())
rows=[];pids=[state['controller_pid']]
for r in candidates:
 probe=Path(r['path']);full=probe.parent.with_name(probe.parent.name[:-3]+'_full')
 assert probe.resolve(strict=True)==probe and probe.is_relative_to(root/'trained-model') and probe.name=='m0_reload_probe.pth'
 assert probe.stat().st_size==r['bytes'] and sha(probe)==r['probe_sha256']
 m0=json.loads((probe.parent/'training.json').read_text())
 train=json.loads((full/'training.json').read_text());official=json.loads((full/'official_metrics.json').read_text())
 assert m0['status']=='M0_PASS' and m0['m0']['reload_max_abs_difference']<=1e-5
 assert train['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE'
 assert [h['epoch'] for h in train['history']]==list(range(1,51))
 row=[a for a in accepted['rows'] if a['run_dir']==str(full)]
 assert len(row)==1;row=row[0]
 assert sha(full/'best_map.pth')==row['checkpoint_sha256']==official['checkpoint_sha256']
 assert sha(full/'official_distances.pt')==row['distance_sha256']==official['distance_sha256']
 assert sorted(p.name for p in full.glob('*.pth'))==['best_map.pth']
 child=Path(row['campaign_dir']);childstate=json.loads((child/'campaign.json').read_text())
 assert childstate['status']=='COMPLETE'
 assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in childstate['jobs'])
 cp=[childstate['controller_pid']]+[j['pid'] for j in childstate['jobs']]
 assert all(not (Path('/proc')/str(pid)).exists() for pid in cp);pids.extend(cp)
 rows.append({{**r,'training_path':str(probe.parent/'training.json'),'training_sha256':sha(probe.parent/'training.json'),
 'formal_best_path':str(full/'best_map.pth'),'formal_best_sha256':row['checkpoint_sha256'],
 'formal_distance_path':str(full/'official_distances.pt'),'formal_distance_sha256':row['distance_sha256'],
 'formal_receipt_path':str(full/'official_metrics.json'),'formal_receipt_sha256':sha(full/'official_metrics.json'),
 'dataset':row['dataset'],'variant':row['variant'],'metrics':row['metrics']}})
source_names=['tools/collect_correspondence_context_identity.py','tools/queue_correspondence_context_identity.py']
print(json.dumps({{'status':'EXACT_FIFTEEN_CLOSED_CONTEXT_PROBES_VERIFIED','at':datetime.now().astimezone().isoformat(),
 'candidates':rows,'retirable_bytes':sum(r['bytes'] for r in rows),'absent_pids':sorted(set(pids)),
 'protected_sha256':protected,'source_sha256':{{n:sha(root/n) for n in source_names}},
 'campaign_path':str(campaign/'campaign.json'),'campaign_sha256':sha(campaign/'campaign.json'),
 'accepted_path':str(campaign/'accepted_matrix.json'),'accepted_sha256':sha(campaign/'accepted_matrix.json'),
 'free_bytes':shutil.disk_usage(root).free,
 'boundary':'Read-only metadata/hash dependency check. Fifteen completed context15 M0 binaries only. Collector consumes M0 training.json, not probe binary; all full50/best/reload/distance receipts retained. No model/scorer/removal.'}},indent=2))
'''
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();rc=stdout.channel.recv_exit_status();client.close()
(target/'stdout.json').write_bytes(data);(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':rc,'at':datetime.now().astimezone().isoformat()})+'\n')
assert rc==0,error.decode()
(target/'INVENTORY.json').write_bytes(data)
record=json.loads(data)
print(json.dumps({k:record[k] for k in ('status','at','retirable_bytes','free_bytes','boundary')}))

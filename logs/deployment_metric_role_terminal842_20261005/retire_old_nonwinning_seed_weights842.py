"""Retire lower-mAP historical seed20 candidates; keep each group's winner."""
from pathlib import Path
from datetime import datetime
import hashlib,json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
source=base/'old_seed_weight_receipts842';assert json.loads((source/'EXIT.json').read_bytes())['exit_code']==0
record=json.loads((source/'stdout.json').read_bytes());groups={}
for row in record['rows']:
 tr=json.loads(row['files']['training.json']['text']);ev=json.loads(row['files']['official_metrics.json']['text'])
 assert tr['status']=='FIXED_EPOCH20_TRAINING_COMPLETE' and ev['status']=='COMPLETE' and ev['fixed_epoch']==20
 assert tr['checkpoint_sha256']==ev['role_checkpoint_sha256']==row['checkpoint_sha256']
 assert tr['seed']==ev['seed'] and ev['independent_upstream_metrics_equal'] and not ev['reranking']
 key=(ev['dataset'],ev['method'],ev['protocol_sha256'],ev['author_checkpoint_sha256'])
 groups.setdefault(key,[]).append(row)
winners=[max(rows,key=lambda r:json.loads(r['files']['official_metrics.json']['text'])['outputs']['fused']['metrics']['mAP']) for rows in groups.values()]
winnerpaths={r['path'] for r in winners}
losers=[r for r in record['rows'] if r['path'] not in winnerpaths]
assert losers
packet=base/'old_nonwinning_seed_retirement842';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');losers=LOSERS;winners=WINNERS
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
sealpath=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json';sealdigest=sha(sealpath)
protected=set(json.loads(sealpath.read_text())['artifact_sha256'])
for row in losers+winners:
 p=Path(row['path']);assert p.resolve().is_relative_to(root/'trained-model') and p.name=='roles_epoch20.pth'
 assert row['path'] not in protected and p.stat().st_size==row['bytes'] and sha(p)==row['checkpoint_sha256']
 for name,receipt in row['files'].items():assert sha(p.parent/name)==receipt['sha256']
 ev=json.loads(row['files']['official_metrics.json']['text'])
 assert sha(Path(ev['distance_arrays']))==ev['distance_arrays_sha256']
archive=root/'logs/old_nonwinning_seed_weight_retirement_20261005_842';assert not archive.exists();archive.mkdir()
cert=dict(status='HISTORICAL_LOWER_MAP_SEED20_WEIGHTS_VERIFIED_BEFORE_RETIREMENT',at=datetime.now().astimezone().isoformat(),
 losers=losers,winners=winners,control_seal_sha256=sealdigest,
 boundary='User best-only storage instruction. Within same dataset/method/protocol/author group among25 nonprimary seeds, preserve highest fusedmAP;retain seed42 and every outside-subset weight. Original results/distance/receipts retained. Not new performance evaluation or unbiased seed selection.')
accept=archive/'ACCEPTANCE.json';accept.write_text(json.dumps(cert,indent=2)+'\\n');before=shutil.disk_usage(root).free
with (archive/'RETIREMENT.jsonl').open('x') as journal:
 for row in losers:
  Path(row['path']).unlink()
  journal.write(json.dumps(dict(path=row['path'],bytes=row['bytes'],checkpoint_sha256=row['checkpoint_sha256'],acceptance_sha256=sha(accept),at=datetime.now().astimezone().isoformat()))+'\\n');journal.flush()
assert all(not Path(r['path']).exists() for r in losers)
assert all(sha(Path(r['path']))==r['checkpoint_sha256'] for r in winners)
assert sha(sealpath)==sealdigest
summary=dict(status='HISTORICAL_NONWINNING_SEED20_WEIGHTS_RETIRED',at=datetime.now().astimezone().isoformat(),retired=len(losers),retired_bytes=sum(r['bytes'] for r in losers),
 retained_group_winners=len(winners),disk_free_before=before,disk_free_after=shutil.disk_usage(root).free,acceptance_sha256=sha(accept))
(archive/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\\n')
print(json.dumps(dict(summary=summary,files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=sha(p)) for p in archive.iterdir()})))
'''.replace('LOSERS',repr(losers)).replace('WINNERS',repr(winners))
compile(code,'remote_old_nonwinning_weights842.py','exec');packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
assert rc==0,error.decode();value=json.loads(data)
for name,row in value['files'].items():
 target=packet/'received'/Path(name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(row['text'].encode('utf-8'))
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
(packet/'SUMMARY.json').write_text(json.dumps(value['summary'],indent=2)+'\n',encoding='utf-8')
print(json.dumps(value['summary']))

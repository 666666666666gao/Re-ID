"""Retire four consumed old diagnostic feature caches; preserve distances and bests."""
from pathlib import Path
from datetime import datetime
import hashlib,json
import paramiko
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft');rows=[]
for packet,family in (('native_fixed_best812_complete','native_fixed_best_20261004_811'),('role_input_detach_fixed_best_complete','role_input_detach_fixed_best_20261004_v1')):
 complete=json.loads((base/packet/'stdout.json').read_bytes())
 assert complete['status']=='SIX_FIXED_BEST_READ_ONLY_DIAGNOSES_VERIFIED' and complete['campaign']['status']=='COMPLETE'
 assert complete['launch_exit']['exit_code']==0
 for variant in ('semantic','native'):
  relative=f'results/{family}/RGBNT100_{variant}/DIAGNOSIS.json'
  diagnosis=json.loads((base/packet/'texts'/f'RGBNT100_{variant}_DIAGNOSIS.json').read_bytes())
  assert diagnosis['status']=='COMPLETE'
  artifact=diagnosis['artifacts']['gallery_features.pt']
  rows.append(dict(path='/data/gaob/Re-ID/Trifusion/'+relative.removesuffix('DIAGNOSIS.json')+'gallery_features.pt',
                   bytes=artifact['bytes'],sha256=artifact['sha256'],diagnosis_path='/data/gaob/Re-ID/Trifusion/'+relative,
                   diagnosis_sha256=complete['files'][relative]['sha256'],distance_sha256=diagnosis['artifacts']['diagnostic_distances.pt']['sha256']))
assert len(rows)==4 and sum(r['bytes'] for r in rows)==843795290
packet=base/'consumed_gallery_feature_retirement843';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=ROWS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected=set(json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256'])
for row in rows:
 p=Path(row['path']);assert p.resolve().is_relative_to(root/'results') and p.name=='gallery_features.pt' and str(p) not in protected
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
 assert sha(Path(row['diagnosis_path']))==row['diagnosis_sha256']
 assert sha(p.parent/'diagnostic_distances.pt')==row['distance_sha256']
archive=root/'logs/consumed_gallery_feature_retirement_20261005_843';assert not archive.exists();archive.mkdir()
cert=dict(status='FOUR_CONSUMED_HISTORICAL_GALLERY_FEATURE_CACHES_VERIFIED',at=datetime.now().astimezone().isoformat(),rows=rows,
 boundary='Old native/detach fixed-best diagnoses already consumed;current187 inputs exclude these caches. Preserve original diagnostic distances,query features,receipts and all model bests. Old cached gallery inspection retired;no NN replay,power-temperature action or current global-task diagnosis deletion.')
accept=archive/'ACCEPTANCE.json';accept.write_text(json.dumps(cert,indent=2)+'\\n');before=shutil.disk_usage(root).free
with (archive/'RETIREMENT.jsonl').open('x') as journal:
 for row in rows:
  Path(row['path']).unlink();journal.write(json.dumps(dict(**row,at=datetime.now().astimezone().isoformat()))+'\\n');journal.flush()
assert all(not Path(r['path']).exists() and sha(Path(r['diagnosis_path']))==r['diagnosis_sha256'] for r in rows)
summary=dict(status='FOUR_CONSUMED_GALLERY_CACHES_RETIRED',at=datetime.now().astimezone().isoformat(),retired_bytes=sum(r['bytes'] for r in rows),disk_free_before=before,
 disk_free_after=shutil.disk_usage(root).free,acceptance_sha256=sha(accept))
(archive/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\\n')
print(json.dumps(dict(summary=summary,files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=sha(p)) for p in archive.iterdir()})))
'''.replace('ROWS',repr(rows))
compile(code,'remote_consumed_gallery_retirement843.py','exec');packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close();(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n');assert rc==0,error.decode();value=json.loads(data)
for name,row in value['files'].items():
 target=packet/'received'/Path(name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(row['text'].encode('utf-8'));assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
(packet/'SUMMARY.json').write_text(json.dumps(value['summary'],indent=2)+'\n',encoding='utf-8');print(json.dumps(value['summary']))

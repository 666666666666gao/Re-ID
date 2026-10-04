"""Read back completed remote retirement; never invoke cleanup or unlink again."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import paramiko

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
original=base/'role_input_detach_m0_retirement'
assert original.exists() and not (original/'EXIT.json').exists()
(original/'EXIT.json').write_text(json.dumps({'exit_code':1,'observation':'Native exec session16834 ended with socket TimeoutError in stdout.read(); no remote exit status received','observed_at':datetime.now().astimezone().isoformat()})+'\n',encoding='utf-8')
(original/'CLIENT_TIMEOUT.json').write_text(json.dumps({'channel_timeout_seconds':60,'exception':'TimeoutError','operation':'stdout.read()','remote_operation_not_replayed':True,'source_sha256':hashlib.sha256(Path('C:/Users/gb/.codex_tmp/retire_role_input_detach_closed_m0.py').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
packet=base/'role_input_detach_m0_retirement_received'
assert not packet.exists();packet.mkdir()
code='''
from datetime import datetime
from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
path=root/'logs/role_input_detach_m0_retirement_20261004/RETIREMENT.json'
record=json.loads(path.read_text())
assert record['status']=='RETIRED_EXACT_6_CLOSED_ROLE_INPUT_DETACH_M0' and len(record['candidates'])==6
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
for item in record['candidates']:
 assert item['deleted'] and not Path(item['path']).exists()
 assert sha(Path(item['m0_training_receipt_path']))==item['m0_training_receipt_sha256']
assert all(Path(item['path']).stat().st_size==item['bytes'] and sha(Path(item['path']))==item['sha256'] for item in record['protected_formal_best'])
assert sha(root/'pertrained-model/ViT-B-16.pt')==record['public_clip_sha256']
seal=json.loads((root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['source_sha256'])==324 and all(sha(root/name)==digest for name,digest in seal['source_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
controls=json.loads((root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(controls['artifact_sha256'])==61 and all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())
print(json.dumps({'status':'REMOTE_RETIREMENT_COMPLETE_READ_BACK_AFTER_CLIENT_TIMEOUT','at':datetime.now().astimezone().isoformat(),'retirement':record,'retirement_sha256':sha(path),'formal_best_source324_artifact124_control61_unchanged':True,'remote_cleanup_invocations':1,'boundary':'Read-only result receipt and hashes after a real local transport timeout. No cleanup replay, unlink, model, training or retired M0 verification.'}))
'''
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(180)
data,error=out.read(),err.read();status=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':status})+'\n',encoding='utf-8')
client.close();assert status==0,error.decode()
record=json.loads(data)
(packet/'RETIREMENT.json').write_text(json.dumps(record['retirement'],indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:record[key] for key in ('status','at','retirement_sha256','formal_best_source324_artifact124_control61_unchanged')}))

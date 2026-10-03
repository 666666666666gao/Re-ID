"""Retire only the exact prelisted old closed M0 probes after SHA checks."""
from pathlib import Path
import json
import paramiko

out=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement')
spec=json.loads((out/'SPEC.json').read_bytes())
assert spec['port']==2026 and len(spec['candidates'])==24
assert not (out/'RETIREMENT.json').exists()
code='''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path(spec['root']).resolve(strict=True)
receipt_path=Path(spec['remote_receipt'])
assert not receipt_path.exists() and not receipt_path.parent.exists()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
processes=subprocess.check_output(['ps','-u','gaob','-o','pid=,args='],text=True)
assert not any(name in processes for name in ('queue_visual_update_control.py','run_visual_update_control.py',
 'queue_clean_clip_joint.py','run_clean_clip_joint.py','queue_training_feature_scale.py','run_training_feature_scale.py'))
for row in spec['closed_summaries']:
 assert digest(Path(row['remote']))==row['sha256']
 summary=json.loads(Path(row['remote']).read_text())
 assert summary['accepted']==row['accepted'] and all(r['status']=='VERIFIED_COMPLETE' for r in summary['rows'])
 campaign=json.loads((Path(row['campaign'])/'campaign.json').read_text())
 assert campaign['status']=='COMPLETE' and not (Path('/proc')/str(campaign['controller_pid'])).exists()
verified=[]
for row in spec['candidates']:
 path=Path(row['path'])
 assert path.name=='m0_reload_probe.pth' and not path.is_symlink()
 assert path.resolve(strict=True)==path and path.is_relative_to(root/'trained-model')
 assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
 training_path=path.parent/'training.json';training=json.loads(training_path.read_text())
 assert training['status']=='M0_PASS' and training['m0']['reload_probe_sha256']==row['sha256']
 verified.append({**row,'original_training_receipt':training,'training_receipt_sha256':digest(training_path),'deleted':False})
for row in spec['protected_best']:
 assert digest(Path(row['path']))==row['sha256']
for row in spec['protected_f3']:
 assert Path(row['path']).stat().st_size==row['bytes'] and digest(Path(row['path']))==row['sha256']
assert digest(Path(spec['public_clip']['path']))==spec['public_clip']['sha256']
assert all(digest(root/name)==expected for name,expected in spec['f3_source_sha256'].items())
record={'status':'VERIFIED_NOT_YET_RETIRED','started_at':datetime.now().astimezone().isoformat(),
        'candidates':verified,'protected_best':spec['protected_best'],'protected_f3':spec['protected_f3'],
        'closed_summaries':spec['closed_summaries'],'public_clip':spec['public_clip'],
        'free_bytes_before':shutil.disk_usage(root).free,'boundary':spec['boundary']}
receipt_path.parent.mkdir();receipt_path.write_text(json.dumps(record,indent=2)+'\\n')
for row in verified:
 Path(row['path']).unlink();row.update(deleted=True,deleted_at=datetime.now().astimezone().isoformat())
 receipt_path.write_text(json.dumps(record,indent=2)+'\\n')
assert all(not Path(row['path']).exists() for row in verified)
assert all(Path(row['path']).is_file() for row in spec['protected_best']+spec['protected_f3'])
record.update(status='RETIRED_EXACT_24_CLOSED_M0',completed_at=datetime.now().astimezone().isoformat(),
              retired_bytes=sum(row['bytes'] for row in verified),free_bytes_after=shutil.disk_usage(root).free)
receipt_path.write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write('spec='+repr(spec)+'\n'+code);stdin.channel.shutdown_write()
data,error=stdout.read(),stderr.read();exit_code=stdout.channel.recv_exit_status()
(out/'stdout.json').write_bytes(data);(out/'stderr.txt').write_bytes(error)
(out/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
record=json.loads(data)
(out/'RETIREMENT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps({key:record[key] for key in ('status','retired_bytes','free_bytes_before','free_bytes_after','completed_at')},indent=2))

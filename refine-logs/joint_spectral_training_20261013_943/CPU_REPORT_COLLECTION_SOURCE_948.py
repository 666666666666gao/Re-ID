"""Collect the original CPU report; do not invoke it again."""
import base64, hashlib, json, subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent
request=json.loads((w/'REPORT_REQUEST_948.json').read_bytes())
assert json.loads((w/'REPORT_TRANSPORT_EXIT_948.json').read_bytes())['exit_code']==1
assert not (w/'REPORT_COLLECTED_948.json').exists()
code='request='+repr(request)+'\n'+r'''
import base64,hashlib,json,shutil
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
camp=Path(request['campaign']);admin=Path(request['admin']);out=Path(request['output'])
state=json.loads((camp/'campaign.json').read_text());receipt=json.loads((admin/'EXIT.json').read_text())
assert state['status']=='COMPLETE' and state['accepted']==state['expected']==3
assert receipt['exit_code']==0 and receipt['report_invocations']==1 and receipt['neural_forwards']==0
assert json.loads((out/'SUMMARY.json').read_text())['status']=='COMPLETE_SAVED_DISTANCE_DIAGNOSIS'
assert all(sha(root/n)==v for n,v in request['source_sha256'].items())
assert sha(root/'tools/report_joint_spectral_delta.py')==request['report_sha256']
files=[admin/'REQUEST.json',admin/'EXIT.json',admin/'stdout.log',admin/'stderr.log',
       camp/'campaign.json',camp/'ACCEPTED_MATRIX.json',camp/'SOURCE_SEAL.json',out/'SUMMARY.json',out/'REPORT.md']
inventory=[]
for arm in state['arms']:
    directory=Path(arm['campaign']);accepted=arm['result']
    files.extend(directory/n for n in ('manifest.json','campaign.json','ACCEPTED_RESULT.json','M0_ACCEPTED.json'))
    full=Path(accepted['run_dir']);files.extend(full/n for n in ('training.json','official_metrics.json'))
    for folder in (full,Path(accepted['m0_dir'])):
        inventory.extend(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in folder.glob('*.pth'))
print(json.dumps(dict(exit_code=receipt['exit_code'],weight_inventory=inventory,remote_free_bytes=shutil.disk_usage(root).free,
    report_invocations=1,collection_only=True,
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size,data=base64.b64encode(p.read_bytes()).decode()) for p in files})))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code,text=True,capture_output=True)
(w/'REPORT_COLLECTION_948.stdout').write_bytes(run.stdout.encode());(w/'REPORT_COLLECTION_948.stderr').write_bytes(run.stderr.encode())
(w/'REPORT_COLLECTION_EXIT_948.json').write_text(json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout);destination=w/'report948';assert not destination.exists();destination.mkdir()
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(w/'REPORT_COLLECTED_948.json').write_text(json.dumps(dict(exit_code=value['exit_code'],weight_inventory=value['weight_inventory'],
    remote_free_bytes=value['remote_free_bytes'],report_invocations=1,collection_only=True,
    original_collection_exit_code=1,original_native_terminal_chunk='154e5c',
    original_failure='Collector requested child SOURCE_SEAL.json; canonical SOURCE_SEAL.json is on parent, per-arm seal is manifest.json.',
    files={name:{k:v for k,v in item.items() if k!='data'} for name,item in value['files'].items()}),indent=2)+'\n')
print((destination/'results/joint_spectral_report948/REPORT.md').read_text())

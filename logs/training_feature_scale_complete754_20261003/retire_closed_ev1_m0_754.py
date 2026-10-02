"""Retire exactly six completed EV1 engineering probes; retain final and current dependencies."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import paramiko

base = Path('C:/Users/gb/.codex_tmp')
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
old = repo/'logs/semantic_native_complete734_20261002/raw'
summary = json.loads((old/'results/semantic_native_evidence_complete_20261002/SUMMARY.json').read_bytes())
assert summary['status'] == 'COMPLETE' and summary['accepted'] == 6
assert summary['formal_epochs'] == 300
f2 = json.loads((base/'training_feature_scale_complete754/INTAKE.json').read_bytes())
root = '/data/gaob/Re-ID/Trifusion'
candidates = []
best = []
for row in summary['rows']:
    relative = Path(row['m0_dir']).relative_to(root).as_posix()
    receipt = json.loads((old/relative/'training.json').read_bytes())
    assert receipt['status'] == 'M0_PASS'
    candidates.append({'path':row['m0_dir']+'/m0_reload_probe.pth',
                       'sha256':receipt['m0']['reload_probe_sha256']})
    best.append({'path':row['run_dir']+'/best_map.pth', 'sha256':row['checkpoint_sha256']})
assert len(candidates) == 6 and len({x['path'] for x in candidates}) == 6
assert not {x['path'] for x in candidates} & {x['path'] for x in f2['binary']}
out = base/'closed_ev1_m0_retirement754'
assert not out.exists()
out.mkdir()
remote_receipt = root+'/logs/closed_ev1_m0_retirement754_20261003/RETIREMENT.json'
spec = {'root':root, 'candidates':candidates, 'protected_best':best,
        'current_f2':f2['binary'], 'remote_receipt':remote_receipt,
        'original_summary_sha256':hashlib.sha256((old/'results/semantic_native_evidence_complete_20261002/SUMMARY.json').read_bytes()).hexdigest(),
        'public_clip_path':summary['rows'][0]['initializer']['public_clip_path'],
        'public_clip_sha256':summary['rows'][0]['initializer']['public_clip_sha256']}
(out/'REQUEST.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
code = '''
from datetime import datetime
from pathlib import Path
import hashlib, json, shutil, subprocess
spec = json.loads(SPEC)
root = Path(spec['root']).resolve(strict=True)
receipt_path = Path(spec['remote_receipt'])
assert not receipt_path.exists() and not receipt_path.parent.exists()
campaign = json.loads((root/'logs/semantic_native_evidence_20261002_v1/campaign.json').read_bytes())
assert campaign['status'] == 'COMPLETE'
assert len(campaign['jobs']) == 6 and all(row['status'] == 'COMPLETE' for row in campaign['jobs'])
processes = subprocess.check_output(['ps','-u','gaob','-o','pid=,args='],text=True)
assert 'queue_semantic_native_evidence.py' not in processes
assert 'run_semantic_native_evidence.py' not in processes
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()
original_summary = root/'results/semantic_native_evidence_complete_20261002/SUMMARY.json'
assert digest(original_summary) == spec['original_summary_sha256']
verified = []
for row in spec['candidates']:
    p = Path(row['path'])
    assert p.name == 'm0_reload_probe.pth' and not p.is_symlink()
    assert p.resolve(strict=True) == p and p.is_relative_to(root/'trained-model')
    assert digest(p) == row['sha256'], str(p)
    verified.append({**row,'bytes':p.stat().st_size,'deleted':False})
for row in spec['protected_best']:
    p = Path(row['path'])
    assert p.is_file() and digest(p) == row['sha256'], str(p)
for row in spec['current_f2']:
    p = Path(row['path'])
    assert p.is_file() and p.stat().st_size == row['bytes'], str(p)
public = Path(spec['public_clip_path'])
assert public.is_file() and digest(public) == spec['public_clip_sha256']
receipt = {'started_at':datetime.now().astimezone().isoformat(),'status':'VERIFIED_NOT_YET_RETIRED',
           'campaign_status':campaign['status'],'original_summary_sha256':spec['original_summary_sha256'],
           'candidates':verified,
           'protected_ev1_best':spec['protected_best'],'current_f2_paths_preserved':spec['current_f2'],
           'public_clip_preserved':{'path':str(public),'sha256':spec['public_clip_sha256']},
           'free_bytes_before':shutil.disk_usage(root).free,
           'boundary':'Only six completed EV1 engineering M0 probe binaries retired after original receipt/SHA verification. EV1 final best, author/public weights, all F2 binaries and all original text/distance evidence retained. Historical EV1 M0 binary replay is no longer available; original receipts remain unchanged. No model, scoring or report executed.'}
receipt_path.parent.mkdir()
receipt_path.write_text(json.dumps(receipt,indent=2)+'\\n')
for row in verified:
    Path(row['path']).unlink()
    row['deleted'] = True
    row['deleted_at'] = datetime.now().astimezone().isoformat()
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\\n')
assert all(not Path(row['path']).exists() for row in verified)
assert all(Path(row['path']).is_file() for row in spec['protected_best']+spec['current_f2'])
receipt.update(status='RETIRED_EXACT_SIX_CLOSED_EV1_M0',completed_at=datetime.now().astimezone().isoformat(),
               retired_bytes=sum(row['bytes'] for row in verified),free_bytes_after=shutil.disk_usage(root).free)
receipt_path.write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''.replace('SPEC',repr(json.dumps(spec)))
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(out/'stdout.txt').write_bytes(data)
(out/'stderr.txt').write_bytes(error)
assert status == 0, error.decode()
actual = json.loads(data)
assert actual['status'] == 'RETIRED_EXACT_SIX_CLOSED_EV1_M0'
(out/'RETIREMENT.json').write_text(json.dumps(actual,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps({k:actual[k] for k in ('status','completed_at','retired_bytes','free_bytes_before','free_bytes_after')},indent=2))

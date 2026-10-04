"""Run only after the original once-only final report has been collected and published."""
from pathlib import Path
import json
import paramiko

private = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
verified = json.loads((private / 'role_input_detach_complete_report/stdout.json').read_bytes())
assert verified['status'] == 'ORIGINAL_ONCE_ONLY_FINAL_CPU_REPORT_AND_ARTIFACTS_VERIFIED'
assert verified['formal_completed'] == 6 and verified['report_exit_code'] == verified['launch_exit']['exit_code'] == 0
assert verified['m0_probe_dependencies_verified_before_cleanup']
diagnosis = json.loads((private / 'role_input_detach_fixed_best_complete/stdout.json').read_bytes())
assert diagnosis['status'] == 'SIX_FIXED_BEST_READ_ONLY_DIAGNOSES_VERIFIED' and diagnosis['all_model_parameter_and_buffer_sha_unchanged']
target = private / 'role_input_detach_m0_retirement'
assert not target.exists()
code = '''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion').resolve(strict=True)
campaign=root/'logs/role_input_detach_v1_20261004_813'
report_root=root/'results/role_input_detach_v1_complete_20261004_813'
receipt_path=root/'logs/role_input_detach_m0_retirement_20261004/RETIREMENT.json'
assert not receipt_path.parent.exists()
diag_root=root/'results/role_input_detach_fixed_best_20261004_v1'
diag_launch=root/'logs/role_input_detach_fixed_best_launch_20261004_v1'
diag_state=json.loads((diag_root/'campaign.json').read_text())
diag_exit=json.loads((diag_launch/'EXIT.json').read_text())
assert diag_state['status']=='COMPLETE' and diag_exit['exit_code']==0
assert len(diag_state['jobs'])==6 and all(job['status']=='COMPLETE' and job['exit_code']==0 for job in diag_state['jobs'])
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert not (Path('/proc')/str(state['controller_pid'])).exists()
assert len(state['jobs'])==12 and all(job['status']=='COMPLETE' and job['exit_code']==0 for job in state['jobs'])
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
for name,info in verified['files'].items():
    assert sha(root/name)==info['sha256']
report=json.loads((report_root/'SUMMARY.json').read_text())
assert report['status']=='COMPLETE' and report['accepted']==6 and report['formal_epochs']==300 and report['formal_steps']==12968
manifest=json.loads((campaign/'manifest.json').read_text())
assert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
seal=json.loads((root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(seal['source_sha256'])==324 and len(seal['rows'])==9
assert all(sha(root/name)==digest for name,digest in seal['source_sha256'].items())
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
controls=json.loads((root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(controls['artifact_sha256'])==61 and all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())
protected_control_artifact_sha=controls['artifact_sha256']
protected=[]
candidates=[]
assert len(verified['artifacts'])==len(matrix['rows'])==6
for row in matrix['rows']:
    key=row['dataset']+'/'+row['variant']
    evidence=verified['artifacts'][key]
    best=Path(row['run_dir'])/'best_map.pth'
    assert best.name=='best_map.pth' and sha(best)==evidence['best_map.pth']['sha256']==row['checkpoint_sha256']
    protected.append({'path':str(best),'sha256':row['checkpoint_sha256'],'bytes':best.stat().st_size})
    probe=Path(row['m0_dir'])/'m0_reload_probe.pth'
    expected=root/f"trained-model/role_input_detach_v1_20261004_813_m0_{row['variant']}_{row['dataset']}/m0_reload_probe.pth"
    assert probe==expected and probe.resolve(strict=True)==expected and not probe.is_symlink()
    assert probe.is_relative_to(root/'trained-model')
    proof=evidence['m0_reload_probe.pth']
    assert str(probe)==proof['path'] and probe.stat().st_size==proof['bytes'] and sha(probe)==proof['sha256']
    training_path=probe.parent/'training.json'
    training=json.loads(training_path.read_text())
    assert training['status']=='M0_PASS' and training['m0']['reload_probe_sha256']==proof['sha256']
    candidates.append({'path':str(probe),'bytes':proof['bytes'],'sha256':proof['sha256'],
        'm0_training_receipt_path':str(training_path),'m0_training_receipt_sha256':sha(training_path),'deleted':False})
public_clip=root/'pertrained-model/ViT-B-16.pt'
clip_digest=sha(public_clip)
assert all(row['initializer']['public_clip_sha256']==clip_digest for row in matrix['rows'])
record={'status':'VERIFIED_NOT_YET_RETIRED','started_at':datetime.now().astimezone().isoformat(),
    'candidates':candidates,'protected_formal_best':protected,'public_clip_sha256':clip_digest,
    'original_report_sha256':sha(report_root/'SUMMARY.json'),'original_report_invocations':1,'original_report_exit_code':0,
    'free_bytes_before':shutil.disk_usage(root).free,
    'boundary':'Delete only6 exact own successful M0 reload probes after original finalCPU report/verify dependency has closed. Keep6 formal mAPbest, authors/inputs/all text/distance/failed evidence. Do not run old M0-dependent verification afterward or relabel missing binaries as replayable; retired probeSHAs and receipts remain.'}
receipt_path.parent.mkdir()
receipt_path.write_text(json.dumps(record,indent=2)+'\\n')
for candidate in candidates:
    Path(candidate['path']).unlink()
    candidate.update(deleted=True,deleted_at=datetime.now().astimezone().isoformat())
    receipt_path.write_text(json.dumps(record,indent=2)+'\\n')
assert all(not Path(candidate['path']).exists() for candidate in candidates)
assert all(sha(Path(best['path']))==best['sha256'] for best in protected)
assert all(Path(candidate['m0_training_receipt_path']).is_file() and sha(Path(candidate['m0_training_receipt_path']))==candidate['m0_training_receipt_sha256'] for candidate in candidates)
assert sha(public_clip)==clip_digest
assert all(sha(Path(name))==digest for name,digest in protected_control_artifact_sha.items())
assert all(sha(Path(name))==digest for name,digest in seal['artifact_sha256'].items())
record.update(status='RETIRED_EXACT_6_CLOSED_ROLE_INPUT_DETACH_M0',completed_at=datetime.now().astimezone().isoformat(),
    retired_bytes=sum(candidate['bytes'] for candidate in candidates),free_bytes_after=shutil.disk_usage(root).free)
receipt_path.write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
compile(code, '<retire-exact-closed-native-m0>', 'exec')
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command('/usr/bin/python3 -B -')
stdin.write('verified='+repr(verified)+'\n'+code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data,error=stdout.read(),stderr.read()
exit_code=stdout.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n',encoding='utf-8')
assert exit_code==0,error.decode()
record=json.loads(data)
(target/'RETIREMENT.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
client.close()
print(json.dumps({key:record[key] for key in ('status','retired_bytes','free_bytes_before','free_bytes_after','completed_at')},indent=2))

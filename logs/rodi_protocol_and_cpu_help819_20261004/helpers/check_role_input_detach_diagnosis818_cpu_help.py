from datetime import datetime
from pathlib import Path
import json
import shlex
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_input_detach_diagnosis818_cpu_help')
assert not packet.exists()
packet.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''
from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
path=root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/SOURCE_SCOPE.json'
scope=json.loads(path.read_text())
assert scope['status']=='SOURCE_SCOPE_REGISTERED_INPUT_ARTIFACT_SEAL_PENDING_FULL_SIX'
assert len(scope['source_sha256'])==324
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in scope['source_sha256'].items())
assert not (path.parent/'INPUT_SEAL.json').exists()
state=json.loads((root/'logs/role_input_detach_v1_20261004_813/campaign.json').read_text())
print(json.dumps({'source_count':324,'source_sha_unchanged':True,'campaign_status':state['status'],'formal_completed':sum(r['phase']=='full' and r['status']=='COMPLETE' for r in state['jobs']),'artifact_seal_created':False}))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(30)
data, error = stdout.read(), stderr.read()
scope_exit = stdout.channel.recv_exit_status()
(packet / 'scope_stdout.json').write_bytes(data)
(packet / 'scope_stderr.txt').write_bytes(error)
(packet / 'scope_EXIT.json').write_text(json.dumps({'exit_code': scope_exit}) + '\n', encoding='utf-8')
assert scope_exit == 0, error.decode()
command = ['env', 'CUDA_VISIBLE_DEVICES=', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B',
    '/data/gaob/Re-ID/Trifusion/tools/diagnose_role_input_detach_best.py', '--help']
_, stdout, stderr = client.exec_command(shlex.join(command))
stdout.channel.settimeout(60)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(packet / 'help_stdout.txt').write_bytes(data)
(packet / 'help_stderr.txt').write_bytes(error)
record = {'status': 'CPU_ONLY_ENTRY_IMPORT_AND_ARGUMENT_HELP_PASS' if exit_code == 0 else 'CPU_ONLY_ENTRY_IMPORT_AND_ARGUMENT_HELP_FAILED',
    'at': datetime.now().astimezone().isoformat(), 'exit_code': exit_code, 'command': command,
    'scope': json.loads((packet / 'scope_stdout.json').read_bytes()),
    'boundary': 'CPU import and CLI help only with empty CUDA_VISIBLE_DEVICES. No model construction, forward, artifact seal, M0/reload, score or training execution. Existing six-endpoint queue unchanged; Goal active/unmet.'}
(packet / 'EXIT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
client.close()
assert exit_code == 0, error.decode()
assert all(name.encode() in data for name in ('--campaign', '--seal', '--output-dir', '--dataset', '--variant'))
print(json.dumps(record, indent=2))

"""Intake sealed executed text source only; do not interpret experimental results."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import paramiko

private = Path('C:/Users/gb/.codex_tmp')
launch = json.loads((private / 'training_feature_scale_deploy746/LAUNCH.json').read_bytes())
expected = launch['source_sha256']
assert len(expected) == 259
root = '/data/gaob/Re-ID/Trifusion'
output = private / 'training_feature_scale_source_intake749'
assert not output.exists()
output.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
sftp = client.open_sftp()
actual = {}
for name, digest in expected.items():
    destination = output / 'source' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sftp.open(root + '/' + name, 'rb') as stream:
        data = stream.read()
    actual[name] = hashlib.sha256(data).hexdigest()
    assert actual[name] == digest, name
    destination.write_bytes(data)
campaign = json.loads(sftp.open(launch['campaign'] + '/campaign.json').read())
sftp.close()
code = "from pathlib import Path;import json; p=Path('/proc')/str(" + str(launch['controller_pid']) + ");print(json.dumps({'controller_live':p.is_dir(),'controller_cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode() if p.is_dir() else None}))"
_, stdout, stderr = client.exec_command('/usr/bin/python3 -B -c ' + __import__('shlex').quote(code))
process = json.loads(stdout.read())
error = stderr.read()
assert stdout.channel.recv_exit_status() == 0, error.decode()
client.close()
receipt = {'status': 'EXECUTED_SOURCE_TEXT_BYTES_EXACT',
           'observed_at': datetime.now().astimezone().isoformat(), 'port': 2026,
           'controller_pid': launch['controller_pid'], **process,
           'source_count': len(actual), 'source_sha256': actual,
           'campaign_status': campaign['status'], 'campaign_phase': campaign['phase'],
           'report_invocations': campaign['report_invocations'],
           'jobs': [{k: v for k, v in row.items() if k in ('phase', 'dataset', 'variant', 'status', 'gpu', 'pid', 'exit_code')}
                    for row in campaign['jobs']],
           'boundary': 'CPU-only exact source text collection; no scoring, model execution, optimizer, report or restart. Not an integrity judgment.'}
(output / 'SOURCE_INTAKE.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
print(json.dumps({k: v for k, v in receipt.items() if k != 'source_sha256'}, indent=2))

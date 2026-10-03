"""After source review, run one isolated CPU synthetic witness on server26."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex

import paramiko


root = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
review = json.loads((root/'source_review_cpu_20261003/VERDICT.json').read_bytes())
assert review['blocking_issues'] == [] and review['status'] in ('PASS', 'WARN')
assert review['agent_id'] == '/root/review_independent_native_component_cpu'
output = root/'cpu_component_witness'
assert not output.exists()
output.mkdir()
remote = '/data/gaob/Re-ID/Trifusion/logs/independent_evidence_component_cpu_20261003'
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = f'''
from pathlib import Path
import json
path=Path({remote!r})
assert not path.exists()
path.mkdir()
campaign=Path('/data/gaob/Re-ID/Trifusion/logs/metric_feature_scale_20261003_v1/campaign.json')
data=json.loads(campaign.read_bytes())
print(json.dumps({{'original_controller_pid':data['controller_pid'],
 'original_status':data['status'],'original_report_invocations':data['report_invocations']}}))
'''
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(output/'prepare.stdout.txt').write_bytes(data)
(output/'prepare.stderr.txt').write_bytes(error)
assert status == 0, error.decode()
sftp = client.open_sftp()
files = ('ImageNativeEvidenceReader.py', 'check_image_native_evidence.py')
source_hashes = {}
for name in files:
    content = (root/name).read_bytes()
    sftp.put(str(root/name), remote+'/'+name)
    with sftp.open(remote+'/'+name, 'rb') as stream:
        assert stream.read() == content
    source_hashes[name] = hashlib.sha256(content).hexdigest()
command = ['/usr/bin/nice', '-n', '10', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=',
           'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
           '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B',
           remote+'/check_image_native_evidence.py', '--source',
           remote+'/ImageNativeEvidenceReader.py', '--output', remote+'/CPU_WITNESS.json']
start = datetime.now().astimezone().isoformat()
(output/'START.json').write_text(json.dumps({'started_at': start, 'port': 2026,
    'command': command, 'source_sha256': source_hashes,
    'scope': 'Four-thread CPU synthetic component witness only; no F3, canonical source, '
             'CUDA, actual dataset, model report or formal training invocation.'}, indent=2)+'\n',
    encoding='utf-8')
stdin, stdout, stderr = client.exec_command(shlex.join(command))
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(output/'stdout.txt').write_bytes(data)
(output/'stderr.txt').write_bytes(error)
(output/'EXIT.json').write_text(json.dumps({'ended_at': datetime.now().astimezone().isoformat(),
    'exit_code': status}, indent=2)+'\n', encoding='utf-8')
assert status == 0, error.decode()
sftp.get(remote+'/CPU_WITNESS.json', str(output/'CPU_WITNESS.json'))
receipt = json.loads((output/'CPU_WITNESS.json').read_bytes())
assert receipt['status'] == 'PASS_STANDALONE_CPU_ENGINEERING_ONLY'
assert receipt['device'] == 'cpu' and receipt['toy_updates'] == 8
sftp.close()
client.close()
print(json.dumps({key: receipt[key] for key in ('status', 'observed_at', 'parameter_count',
                 'trainable_tensors', 'toy_updates', 'scope')}, indent=2))

from pathlib import Path
from datetime import datetime
import hashlib
import json
import paramiko

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
target = base / 'native_installed_sources786_actual_python'
assert not target.exists()
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
code = '''from pathlib import Path
from datetime import datetime
import hashlib,json,sys
env=Path('/data/gaob/Re-ID/conda-envs/tri_reid')
site=env/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
assert site.is_dir()
names=['mamba_ssm/__init__.py','mamba_ssm/modules/mamba_simple.py','mamba_ssm/ops/selective_scan_interface.py']
metadata=list(site.glob('mamba_ssm-*.dist-info/METADATA'))
assert len(metadata)==1
names.append(metadata[0].relative_to(site).as_posix())
names.append('torch/nn/functional.py')
rows={}
for name in names:
 p=site/name
 assert p.is_file()
 rows[name]={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'interpreter':sys.executable,'python_version':sys.version,'files':rows,
 'boundary':'Read-only installed Python source and package metadata. No package import, Torch/CUDA/model/scorer/weight operations; compiled kernels not inspected.'}))
'''
stdin, stdout, stderr = client.exec_command('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
rc = stdout.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':rc,'at':datetime.now().astimezone().isoformat()})+'\n')
assert rc == 0, error.decode()
record = json.loads(data)
sftp = client.open_sftp()
for name, row in record['files'].items():
    dest = target/'source'/name
    dest.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(row['path'], str(dest))
    assert dest.stat().st_size == row['bytes']
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == row['sha256']
sftp.close()
client.close()
(target/'INTAKE.json').write_bytes(data)
print(json.dumps({'at':record['at'],'files':len(record['files']),'boundary':record['boundary']}))

from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_binary_tools789')
assert not target.exists()
target.mkdir()
code = '''from pathlib import Path
from datetime import datetime
import json,shutil,subprocess
paths=[p/'bin/cuobjdump' for p in Path('/usr/local').glob('cuda*') if (p/'bin/cuobjdump').is_file()]
packages={}
for name in ('mamba_ssm-2.2.6.post3.dist-info','causal_conv1d-1.6.0.dist-info'):
 from urllib.parse import urlsplit
 metadata=Path('/data/gaob/Re-ID/conda-envs/tri_reid/lib/python3.10/site-packages')/name/'direct_url.json'
 parsed=urlsplit(json.loads(metadata.read_bytes())['url'])
 packages[name]={'source_basename':Path(parsed.path).name,'source_top_level':Path(parsed.path).parts[1]}
nm=shutil.which('nm')
symbols={}
if nm:
 for name in ('selective_scan_cuda.cpython-310-x86_64-linux-gnu.so','causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so'):
  library=Path('/data/gaob/Re-ID/conda-envs/tri_reid/lib/python3.10/site-packages')/name
  result=subprocess.run([nm,'-D','-C','--defined-only',str(library)],capture_output=True,text=True)
  symbols[name]={'exit_code':result.returncode,'stderr':result.stderr,'lines':[line for line in result.stdout.splitlines() if any(term in line for term in ('selective_scan_bwd','causal_conv1d_bwd'))]}
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'nm':nm,'cuobjdump_on_path':shutil.which('cuobjdump'),'installed_cuobjdump_paths':[str(p) for p in paths],'package_source_summary':packages,'symbols':symbols,'boundary':'Read-only filesystem and static ELF symbol inspection. No CUDA kernels, library loading, Torch/model import, build, install or training.'}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':status,'at':datetime.now().astimezone().isoformat()})+'\n',encoding='utf-8')
client.close()
assert status == 0, error.decode()
record = json.loads(data)
(target/'INTAKE.json').write_bytes(data)
print(json.dumps({k:v for k,v in record.items() if k!='symbols'},indent=2))
for name,row in record['symbols'].items():
    print(name, row['exit_code'], len(row['lines']), row['lines'][:4])

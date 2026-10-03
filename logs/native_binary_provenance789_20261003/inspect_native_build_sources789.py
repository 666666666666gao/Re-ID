from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_build_sources789')
assert not target.exists()
target.mkdir()
code = '''from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit,unquote
import hashlib,json,subprocess
site=Path('/data/gaob/Re-ID/conda-envs/tri_reid/lib/python3.10/site-packages')
packages=[]
for name in ('mamba_ssm-2.2.6.post3.dist-info','causal_conv1d-1.6.0.dist-info'):
 direct=json.loads((site/name/'direct_url.json').read_bytes())
 parsed=urlsplit(direct['url'])
 assert parsed.scheme=='file' and not parsed.netloc
 source=Path(unquote(parsed.path))
 row={'package':name,'source_basename':source.name,'source_exists':source.is_dir()}
 if source.is_dir():
  for command,key in ((['rev-parse','HEAD'],'git_head'),(['status','--porcelain','--untracked-files=no'],'git_tracked_status')):
   result=subprocess.run(['git','-C',str(source),*command],capture_output=True,text=True)
   row[key]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
  paths=[]
  for relative in ('setup.py','pyproject.toml'):
   path=source/relative
   if path.is_file():
    data=path.read_bytes()
    paths.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()})
  if (source/'csrc').is_dir():
   row['csrc_inventory']=[str(p.relative_to(source)) for p in sorted((source/'csrc').rglob('*')) if p.is_file() and p.suffix in ('.cu','.cuh','.cpp','.h')]
  if source.name=='mamba':
   for relative in ('mamba_ssm/modules/mamba_simple.py','mamba_ssm/ops/selective_scan_interface.py','csrc/selective_scan/selective_scan.cpp','csrc/selective_scan/selective_scan_bwd_kernel.cuh','csrc/selective_scan/selective_scan.h'):
    path=source/relative
    if path.is_file():
     data=path.read_bytes()
     paths.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()})
  row['files']=paths
  row['build_directory_exists']=(source/'build').is_dir()
  if row['build_directory_exists']:
   row['build_records']=[str(p.relative_to(source)) for p in sorted((source/'build').rglob('*')) if p.is_file() and p.name in ('build.ninja','CMakeCache.txt','compile_commands.json')]
 packages.append(row)
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'packages':packages,'boundary':'Existing local package source directories only. Source contents and build-record inventory read without Torch, CUDA, extension load, build, install or training. Current source files alone do not prove the installed binary was compiled from them.'}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(target / 'stdout.json').write_bytes(data)
(target / 'stderr.txt').write_bytes(error)
(target / 'EXIT.json').write_text(json.dumps({'exit_code': status, 'at': datetime.now().astimezone().isoformat()}) + '\n', encoding='utf-8')
client.close()
assert status == 0, error.decode()
record = json.loads(data)
(target / 'INTAKE.json').write_bytes(data)
for row in record['packages']:
    print(json.dumps({k:v for k,v in row.items() if k != 'files'}))
print(record['boundary'])

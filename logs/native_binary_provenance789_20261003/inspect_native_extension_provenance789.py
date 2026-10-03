from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_extension_provenance789')
assert not target.exists()
target.mkdir()
code = '''from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit
import base64,csv,hashlib,json
site=Path('/data/gaob/Re-ID/conda-envs/tri_reid/lib/python3.10/site-packages')
assert site.is_dir()
packages={}
for pattern in ('mamba_ssm-*.dist-info','causal_conv1d-*.dist-info'):
 rows=[]
 for info in sorted(site.glob(pattern)):
  metadata=(info/'METADATA').read_text()
  version=[line.removeprefix('Version: ') for line in metadata.splitlines() if line.startswith('Version: ')]
  assert len(version)==1
  record=list(csv.reader((info/'RECORD').read_text().splitlines()))
  extensions=[]
  for name,digest,size in record:
   if name.endswith('.so'):
    path=(site/name).resolve()
    assert path.is_file()
    actual=hashlib.sha256(path.read_bytes()).digest()
    encoded=base64.urlsafe_b64encode(actual).decode().rstrip('=')
    extensions.append({'record_name':name,'record_digest':digest,'record_size':size,
       'actual_path':str(path),'actual_size':path.stat().st_size,'actual_sha256':actual.hex(),
       'matches_record_digest':digest=='sha256='+encoded,'matches_record_size':size==str(path.stat().st_size)})
  row={'dist_info':info.name,'version':version[0],'metadata_sha256':hashlib.sha256((info/'METADATA').read_bytes()).hexdigest(),
       'wheel':(info/'WHEEL').read_text(),'installer':(info/'INSTALLER').read_text(),'extensions':extensions,
       'direct_url_present':(info/'direct_url.json').is_file()}
  if row['direct_url_present']:
   direct=json.loads((info/'direct_url.json').read_bytes())
   parsed=urlsplit(direct['url'])
   row['direct_url_summary']={'scheme':parsed.scheme,'hostname':parsed.hostname,
      'path_basename':Path(parsed.path).name,'archive_info':direct.get('archive_info'),
      'vcs_info':direct.get('vcs_info'),'dir_info':direct.get('dir_info')}
  rows.append(row)
 packages[pattern]=rows
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'site':str(site),'packages':packages,
 'boundary':'Read-only package metadata and extension bytes hashed on2026. No Torch/model/CUDA imports, '
 'library loading, compiled binary download, install, scorer, training or weight change. '
 'RECORD equality establishes package file integrity, not compiled source equivalence or runtime selection.'}))
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
(target / 'EXIT.json').write_text(json.dumps({'exit_code': status,
    'at': datetime.now().astimezone().isoformat()}) + '\n', encoding='utf-8')
client.close()
assert status == 0, error.decode()
record = json.loads(data)
(target / 'INTAKE.json').write_bytes(data)
print(json.dumps({'at': record['at'], 'packages': record['packages'], 'boundary': record['boundary']}, indent=2))

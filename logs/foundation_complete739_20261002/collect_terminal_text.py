"""Receive complete F1 primary text and remote binary hashes after the sole CPU report."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex

import paramiko

ROOT = '/data2/gb/Re-ID/Trifusion'
OUT = Path('C:/Users/gb/.codex_tmp/foundation_recipe_complete_20261002')
SOURCE = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002/source_intake737')
assert not OUT.exists()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2025,username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
code = f'''
from datetime import datetime
import hashlib,json
from pathlib import Path
root=Path({ROOT!r})
campaign=root/'logs/foundation_recipe_20261002_v1'
report=root/'results/foundation_recipe_complete_20261002'
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert len(state['jobs'])==12 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
manifest=json.loads((campaign/'manifest.json').read_text())
source=manifest['source_sha256']
assert len(source)==249 and source=={{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source}}
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
summary=json.loads((report/'SUMMARY.json').read_text())
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==6
assert summary['accepted']==6 and summary['formal_epochs']==300
text=list(campaign.rglob('*.json'))+list(campaign.rglob('*.log'))+list(report.glob('*.json'))+list(report.glob('*.md'))
text.append(root/'logs/foundation_recipe_20261002_v1.controller.log')
binary=[]
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
 for recipe in ('author','current'):
  for phase in ('m0','full'):
   run=root/'trained-model'/f'foundation_recipe_20261002_v1_{{phase}}_{{recipe}}_{{dataset}}'
   text.extend(run.glob('*.json'));text.extend(run.glob('*.jsonl'))
   names=('m0_reload_probe.pth',) if phase=='m0' else ('best_map.pth','best_epoch_distances.pt','official_distances.pt')
   for name in names:
    path=run/name
    digest=hashlib.sha256()
    with path.open('rb') as stream:
     for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    binary.append({{'path':str(path),'bytes':path.stat().st_size,'sha256':digest.hexdigest()}})
files={{str(path.relative_to(root)):{{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}} for path in text}}
assert len(files)==len(text)
print(json.dumps({{'status':'ALL6_FORMAL_AND_ONCE_CPU_REPORT_COMPLETE','observed_at':datetime.now().astimezone().isoformat(),
 'source_sha256':source,'text':files,'binary':binary,'controller_state':state,
 'accepted_matrix':matrix,'terminal_summary_sha256':files[str((report/'SUMMARY.json').relative_to(root))]['sha256']}}))
'''
_,stdout,stderr=client.exec_command(shlex.join(['/usr/bin/python3','-B','-c',code]))
raw,error=stdout.read(),stderr.read()
assert stdout.channel.recv_exit_status()==0,error.decode()
receipt=json.loads(raw)
assert all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
           for name,digest in receipt['source_sha256'].items())
OUT.mkdir()
sftp=client.open_sftp()
for name,entry in receipt['text'].items():
    data=sftp.open(ROOT+'/'+name,'rb').read()
    assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],name
    path=OUT/'raw'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
sftp.close()
client.close()
receipt['received_at']=datetime.now().astimezone().isoformat()
receipt['local_source_directory']=str(SOURCE)
receipt['boundary']='Complete executor evidence intake; independent integrity and claim review not yet performed. Binary checkpoints/distances stay remote.'
(OUT/'INTAKE.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'text_files':len(receipt['text']),
                  'remote_binary_records':len(receipt['binary']),'output':str(OUT)},indent=2),flush=True)

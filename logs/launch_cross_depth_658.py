from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
metadata = json.loads((root / '.git/cross_depth_source_658.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip() == metadata['head']
assert all(sha(root / name) == digest for name,digest in metadata['source_sha256'].items())
old_sources = json.loads((root / '.git/m3_source_sha_640.json').read_text())
old_manifest = json.loads((root / 'logs/correspondence_context_identity_20260929/manifest.json').read_text())
assert all(sha(root/name) == digest for name,digest in {**old_sources,**old_manifest['source_sha256']}.items())
prior = root / 'logs/correspondence_context_identity_20260929'
assert sha(prior / 'accepted_matrix.json') == '7fd21fa6398578ff22aba2f1812ca2c6a163287ae2447263811941b53180f958'
state = json.loads((prior / 'campaign.json').read_text())
assert state['status'] == 'COMPLETE' and all(j['status'] == 'COMPLETE' for j in state['jobs'])
assert shutil.disk_usage(root).free >= 20 * 1024**3
memory = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert len(memory.splitlines()) == 4 and all(int(row.split(',')[1]) < 500 for row in memory.splitlines())
campaign = root / 'logs/cross_depth_role_state_20260929'
assert not campaign.exists()
record_path = root / '.git/cross_depth_launch_658_20260929.json'
assert not record_path.exists()
command = [sys.executable,'-B',str(root/'tools/queue_cross_depth_role_state.py'),
           '--campaign',str(campaign),'--after-campaign',str(prior),
           '--after-matrix-sha256','7fd21fa6398578ff22aba2f1812ca2c6a163287ae2447263811941b53180f958']
with (root / '.git/cross_depth_controller_658_20260929.log').open('x') as log:
    process = subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record = {'status':'CONTROLLER_LAUNCHED_NOT_YET_VERIFIED','at':datetime.now().astimezone().isoformat(),
          'pid':process.pid,'command':command,'campaign':str(campaign),'gpu_memory_before':memory,
          'source_contract':metadata,'boundary':'Persistent nine-endpoint M0/full50/best/reload queue; '
          'launch is not M0 PASS or formal completion. Poll240; no preemption/retry/new coefficients.'}
record_path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'pid':process.pid,'at':record['at'],'campaign':str(campaign)}))

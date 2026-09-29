from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root/'logs/cross_depth_role_state_20260929'
state = json.loads((campaign/'campaign.json').read_text())
manifest = json.loads((campaign/'manifest.json').read_text())
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name) == digest for name,digest in manifest['source_sha256'].items())
pid = state['controller_pid']
controller = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'queue_cross_depth_role_state.py' in controller and str(campaign) in controller
rows = []
for job in state['jobs']:
    row = {name:job[name] for name in ('dataset','variant','status')}
    if job['status'] in ('RUNNING','COMPLETE','FAILED'):
        child = campaign/f"{campaign.name}_depth_{job['variant']}_{job['dataset']}"
        receipt = json.loads((child/'campaign.json').read_text())
        row.update(gpu=job['gpu'],child_status=receipt['status'],jobs=receipt['jobs'])
        for phase in receipt['jobs']:
            folder = Path(phase['output_dir'])
            if (folder/'training.json').exists():
                training = json.loads((folder/'training.json').read_text())
                phase['training_receipt_status'] = training['status']
                phase['recorded_epochs'] = len(training['history'])
                phase['initializer'] = training['initializer']
                if 'm0' in training:
                    phase['m0_evidence'] = training['m0']
        if job['status'] == 'RUNNING':
            child_pid = receipt['controller_pid']
            row['child_command'] = Path(f'/proc/{child_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    rows.append(row)
path = root/'.git/cross_depth_progress_659_20260929.json'
assert not path.exists()
record = {'status':'ACTUAL_CROSS_DEPTH_PROGRESS','at':datetime.now().astimezone().isoformat(),
          'campaign_status':state['status'],'controller_pid':pid,'controller_command':controller,
          'manifest_sha256':sha(campaign/'manifest.json'),'source_paths':len(manifest['source_sha256']),
          'rows':rows,'gpu_memory':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
           '--format=csv,noheader,nounits'],text=True),
          'boundary':'Actual process and receipts, not endpoint scores or expected completion.'}
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'at':record['at'],'campaign_status':state['status'],'controller_pid':pid,
                  'rows':[{k:r[k] for k in ('dataset','variant','status')} for r in rows],
                  'gpu_memory':record['gpu_memory']}))

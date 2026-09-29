from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() == '7af2afd73f6e8a81348aa609b43e6a2277b78363'
sources = json.loads((root / '.git/m3_source_sha_640.json').read_text())
assert all(sha(root / name) == digest for name, digest in sources.items())
validation = json.loads((root / '.git/context_identity_cli_652_20260929.json').read_text())
assert validation['status'] == 'REAL_ENV_CPU_IMPORT_AST_AND_REGISTERED_COMMAND_PASS'
assert len(validation['command_contracts']) == 45
assert all(sha(root / name) == digest for name, digest in validation['source_sha256'].items())
prior = root / 'logs/correspondence_m3_prediction_20260929'
assert sha(prior / 'manifest.json') == 'df7b3498db1b1dd423fe643681dec2f8dce495525c6eaa52b4721ecbadbba3db'
state = json.loads((prior / 'campaign.json').read_text())
assert state['status'] == 'RUNNING' and len(state['jobs']) == 12
assert not any(job['status'] in ('PENDING', 'FAILED') for job in state['jobs'])
campaign = root / 'logs/correspondence_context_identity_20260929'
assert not campaign.exists()
output = root / '.git/context_identity_launch_653_20260929.json'
assert not output.exists()
command = [sys.executable, '-B', str(root / 'tools/queue_correspondence_context_identity.py'),
           '--campaign', str(campaign), '--after-campaign', str(prior), '--after-pid', '2566593',
           '--after-manifest-sha256', sha(prior / 'manifest.json')]
with (root / '.git/context_identity_controller_653.log').open('x') as log:
    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = {'status': 'NEW_CONTEXT_CONTROLLER_LAUNCHED_NOT_M0_OR_FORMAL_RESULT',
          'at': datetime.now().astimezone().isoformat(), 'head': '7af2afd73f6e8a81348aa609b43e6a2277b78363',
          'controller_pid': process.pid, 'command': command, 'campaign': str(campaign),
          'parent_manifest_sha256': sha(prior / 'manifest.json'), 'registered_endpoints': 15,
          'boundary': 'All old jobs started, 201/MSVR verified, free-card nonpreemptive continuation; old GPU worker reserved across train/evaluate transition. Real M0 precedes each full50/best/reload. This launch does not establish M0 pass, formal results or scientific success.'}
output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))

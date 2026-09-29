from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path('/data/gaob/Re-ID/Trifusion')
head = '815a75e471c2a6b17e8add0b32b741d77ecbc41f'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() == head
source = json.loads((root / '.git/m3_source_sha_640.json').read_text())
for name, digest in source.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
sync = json.loads((root / '.git/correspondence_roles_sync_640_20260929.json').read_text())
assert sync['status'] == 'M3_SOURCE_DOC_AND_EVIDENCE_SYNC_VERIFIED' and sync['head'] == head
review = json.loads(subprocess.check_output(['git', 'show', 'HEAD:logs/correspondence_m3_code_review_640_20260929.json'], cwd=root))
assert review['verdict'] == 'PASS' and review['blocking'] == []
m0_path = root / 'trained-model/correspondence_m3_sanity_matched_predictor_RGBNT201_seed42_m0_20260929/training.json'
m0 = json.loads(m0_path.read_text())
assert m0['status'] == 'M0_PASS'
assert m0['m0']['nonzero_gradient_parameters'] == m0['m0']['trainable_parameters'] == 136
assert m0['m0']['frozen_signal_unchanged'] and m0['m0']['reload_max_abs_difference'] == 0.0
assert shutil.disk_usage(root).free >= 10 * 1024**3
gpu_memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
available = [int(row.split(',')[0]) for row in gpu_memory.splitlines() if int(row.split(',')[0]) in range(4) and int(row.split(',')[1]) < 500]
assert len(available) == 4, gpu_memory
campaign = root / 'logs/correspondence_m3_prediction_20260929'
log = root / '.git/correspondence_m3_panel_launch_641_20260929.log'
receipt = root / '.git/correspondence_m3_panel_launch_641_20260929.json'
prior_path = root / '.git/correspondence_refinement_accepted_m2_complete_639_20260929.json'
assert not campaign.exists() and not log.exists() and not receipt.exists() and not prior_path.exists()
prior = subprocess.check_output(['git', 'show', 'HEAD:logs/correspondence_refinement_accepted_m2_complete_639_20260929.json'], cwd=root)
prior_sha = 'ed3c488ecf4ae106ab257eb657c4a65cc6b90245bec239a19aacacf37ffdab71'
assert hashlib.sha256(prior).hexdigest() == prior_sha
prior_path.write_bytes(prior)
command = ['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', str(root / 'tools/queue_correspondence_role_prediction.py'),
           '--campaign', str(campaign), '--prior-matrix', str(prior_path), '--prior-matrix-sha256', prior_sha]
with log.open('x') as handle:
    process = subprocess.Popen(command, cwd=root, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
record = {'status': 'M3_FIXED_TWELVE_PANEL_COORDINATOR_STARTED', 'at': datetime.now().astimezone().isoformat(),
          'head': head, 'pid': process.pid, 'command': command, 'campaign': str(campaign), 'log': str(log),
          'source_sha256': source, 'prior_matrix_sha256': prior_sha,
          'sanity_receipt_sha256': hashlib.sha256(m0_path.read_bytes()).hexdigest(),
          'available_gpu_indices': available, 'boundary': 'Coordinator launch only; actual child M0 and full50 phases must be observed. Four free GPUs checked before launch. Fixed twelve jobs, no retry or tuning.'}
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))

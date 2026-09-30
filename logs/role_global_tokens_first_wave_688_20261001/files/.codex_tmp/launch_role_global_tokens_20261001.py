"""Launch the reviewed nine-end campaign once, using existing four-GPU runtime."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--head', required=True)
parser.add_argument('--review-sha', required=True)
args = parser.parse_args()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == args.head
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT).strip()
review_path = ROOT / 'refine-logs/role_global_tokens_v1/REVIEW.json'
assert hashlib.sha256(review_path.read_bytes()).hexdigest() == args.review_sha
review = json.loads(review_path.read_text())
assert review['schema'] == 'trifusion-role-global-tokens-source-preflight-review-v1'
assert review['source_contract_verdict'] == 'PASS'
assert review['verdict'] in ('PASS', 'WARN') and not review['mandatory_code_fixes']
assert review['review_independence'] == 'same-family' and review['acceptance_status'] == 'provisional'
preflight = json.loads((review_path.parent / 'PREFLIGHT_20261001.json').read_text())
assert preflight['status'] == 'CPU_AND_CLI_PASS'
assert all(item['exit_code'] == 0 for item in preflight['checks'])
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
           for name, digest in preflight['source_sha256'].items())
campaign = ROOT / 'logs/role_global_tokens_20261001_v1'
receipt = ROOT / 'logs/role_global_tokens_launch_20261001_v1.json'
log_path = ROOT / 'logs/role_global_tokens_controller_20261001_v1.log'
assert not campaign.exists() and not receipt.exists() and not log_path.exists()
gpu_inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
command = [sys.executable, '-B', str(ROOT / 'tools/queue_role_global_tokens.py'),
           '--campaign', str(campaign), '--after-campaign',
           str(ROOT / 'logs/slot_competition_fp32_roles_20261001_r2'),
           '--after-matrix-sha256', 'e3433db691dbb1070595386c542f27782cd88b45f5f6faa8660b246e7c2fd4b8']
with log_path.open('x') as log:
    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
record = {'launched_at': datetime.now().astimezone().isoformat(), 'pid': child.pid,
          'command': command, 'head': args.head, 'review_sha256': args.review_sha,
          'source_sha256': preflight['source_sha256'], 'gpu_inventory': gpu_inventory,
          'campaign': str(campaign), 'controller_log': str(log_path),
          'expected_endpoints': 9, 'seed': 42, 'epochs': 50,
          'status': 'DISPATCHED_NOT_ACCEPTED', 'poll_seconds': 240}
receipt.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))

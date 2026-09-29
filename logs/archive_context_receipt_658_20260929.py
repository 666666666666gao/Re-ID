from pathlib import Path
import hashlib
import json
import subprocess

root = Path(r'C:/Users/gb/.trifusion_github_publish_22c3bee')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
matrix = json.loads((root / 'logs/context_identity_accepted_658_20260929.json').read_bytes())
rows = [row for row in matrix['rows'] if row['status'] == 'VERIFIED_COMPLETE']
assert len(rows) == matrix['verified_complete'] == 3
row = next(row for row in rows if row['dataset'] == 'RGBNT201' and row['variant'] == 'context_none')
control = next(row for row in rows if row['dataset'] == 'RGBNT201' and row['variant'] == 'static_none')
assert row['initial_model_state_sha256'] == control['initial_model_state_sha256']
assert row['trainable_parameters'] == control['trainable_parameters']
target = root / 'logs/context_identity_context_none_RGBNT201_658_20260929'
target.mkdir()
names = ('training.json', 'official_metrics.json', 'training_steps.jsonl')
options = ['scp', '-P', '2026', '-i', 'C:/Users/gb/.ssh/id_ed25519',
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ProxyCommand=none']
subprocess.run(options + [f"gaob@172.19.12.138:{row['run_dir']}/{name}" for name in names] + [str(target)], check=True)
assert sha(target / 'official_metrics.json') == row['receipt_sha256']
training = json.loads((target / 'training.json').read_bytes())
assert [item['epoch'] for item in training['history']] == list(range(1, 51))
assert training['best_epoch'] == row['best_epoch']
assert training['initializer']['initial_model_state_sha256'] == row['initial_model_state_sha256']
assert len((target / 'training_steps.jsonl').read_text().splitlines()) == row['task_scalars']['logged_steps']
print(json.dumps({'status': 'CONTEXT201_TERMINAL_TEXT_RECEIPTS_ARCHIVED',
                  'files': {name: sha(target / name) for name in names}}))

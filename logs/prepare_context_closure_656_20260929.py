from pathlib import Path
import hashlib
import json
import subprocess

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
scratch = Path('C:/Users/gb/.codex_tmp')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
matrix = read(root / 'logs/context_identity_accepted_662_20260929.json')
previous = read(root / 'logs/context_identity_accepted_658_20260929.json')
rows = {(r['dataset'], r['variant']): r for r in matrix['rows'] if r['status'] == 'VERIFIED_COMPLETE'}
assert len(rows) == matrix['verified_complete'] == 5
assert all(rows[(r['dataset'], r['variant'])] == r for r in previous['rows'] if r['status'] == 'VERIFIED_COMPLETE')
old_matrix = read(root / 'logs/correspondence_m3_accepted_complete_656_20260929.json')
assert old_matrix['verified_complete'] == old_matrix['expected_endpoints'] == 12
old = next(r for r in old_matrix['rows'] if r['dataset'] == 'RGBNT100' and r['variant'] == 'matched_predictor')
scp = ['scp', '-P', '2026', '-i', 'C:/Users/gb/.ssh/id_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ProxyCommand=none']
ssh = ['ssh', '-p', '2026', '-i', 'C:/Users/gb/.ssh/id_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ProxyCommand=none', 'gaob@172.19.12.138']
files = ('training.json', 'official_metrics.json', 'training_steps.jsonl')
archive_records = {}
for row, dirname in ((old, 'correspondence_m3_matched_predictor_RGBNT100_656_20260929'),
                     (rows[('MSVR310', 'context_none')], 'context_identity_context_none_MSVR310_662_20260929'),
                     (rows[('RGBNT100', 'static_none')], 'context_identity_static_none_RGBNT100_662_20260929')):
    target = root / 'logs' / dirname
    assert not target.exists()
    target.mkdir()
    subprocess.run(scp + [f"gaob@172.19.12.138:{row['run_dir']}/{name}" for name in files] + [str(target)], check=True)
    assert sha(target / 'official_metrics.json') == row['receipt_sha256']
    training = read(target / 'training.json')
    assert [item['epoch'] for item in training['history']] == list(range(1, 51))
    assert training['best_epoch'] == row['best_epoch']
    archive_records[dirname] = {'run_dir': row['run_dir'], 'files': {name: sha(target / name) for name in files}}
report = scratch / 'context_closure_archive_656.json'
assert not report.exists()
report.write_text(json.dumps(archive_records, indent=2) + '\n')
source = (scratch / 'observe_context_662.py').read_text()
observer = scratch / 'observe_context_663.py'
assert not observer.exists()
observer.write_text(source.replace('context_identity_progress_662_20260929.json', 'context_identity_progress_663_20260929.json'))
subprocess.run(scp + [str(observer), 'gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/.git/observe_context_663.py'], check=True)
subprocess.run(ssh + ['cd /data/gaob/Re-ID/Trifusion && /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B .git/observe_context_663.py'], check=True)
subprocess.run(scp + ['gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/.git/context_identity_progress_663_20260929.json', str(root / 'logs/context_identity_progress_663_20260929.json')], check=True)
print(json.dumps({'status': 'NEW_ACCEPTED_FIVE_AND_THREE_TERMINAL_TEXT_ARCHIVES_VERIFIED', 'archives': archive_records}))

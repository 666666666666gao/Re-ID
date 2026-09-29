from pathlib import Path
import hashlib
import json
import subprocess

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
folder = root / 'logs/context_identity_factor_complete_657_20260929'
folder.mkdir()
scp = ['scp', '-P', '2026', '-i', 'C:/Users/gb/.ssh/id_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ProxyCommand=none']
subprocess.run(scp + ['gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/.git/context_identity_factor_complete_657_20260929/*.json', str(folder)], check=True)
summary = json.loads((folder / 'summary.json').read_bytes())
assert summary['status'] == 'COMPLETE_CONTEXT_LOCAL_15_ENDPOINT_DIAGNOSIS'
assert summary['source_sha256'] == sha(root / 'tools/analyze_correspondence_context_identity.py')
assert summary['comparison_source_sha256'] == sha(root / 'tools/analyze_correspondence_distances.py')
assert summary['matrix_sha256'] == sha(root / 'logs/context_identity_accepted_complete_670_20260929.json')
assert len(summary['pairs']) == 15 and len(list(folder.glob('*.json'))) == 14
sealed = 0
for pair in summary['pairs']:
    if pair['origin'] == 'sealed_report_reused':
        path = root / 'logs' / Path(pair['report']).name
        sealed += 1
    else:
        path = folder / Path(pair['report']).name
    assert sha(path) == pair['report_sha256']
assert sealed == 2
for dataset, group in summary['datasets'].items():
    print(json.dumps({'dataset': dataset, 'factors': group['factor_metrics_pp'],
                      'cells': [{k: row[k] for k in ('variant', 'best_epoch', 'metrics')} for row in group['cells']]}))
print(json.dumps({'status': 'FULL15_FACTOR_AND_PAIR_BYTES_ARCHIVED', 'sealed_reused': sealed,
                  'new_pair_files': 13, 'summary_sha256': sha(folder / 'summary.json')}))

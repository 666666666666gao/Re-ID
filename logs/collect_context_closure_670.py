from pathlib import Path
import json
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.collect_correspondence_roles import sha

campaign = root / 'logs/correspondence_context_identity_20260929'
state = json.loads((campaign / 'campaign.json').read_text())
assert state['status'] == 'COMPLETE' and len(state['jobs']) == 15
assert all(row['status'] == 'COMPLETE' for row in state['jobs'])
output = root / '.git/context_identity_accepted_complete_670_20260929.json'
assert not output.exists()
accepted_path = campaign / 'accepted_matrix.json'
report = json.loads(accepted_path.read_text())
assert report['verified_complete'] == report['expected_endpoints'] == len(report['rows']) == 15
manifest = json.loads((campaign / 'manifest.json').read_text())
assert all(sha(root / name) == digest for name, digest in manifest['source_sha256'].items())
for row in report['rows']:
    assert row['status'] == 'VERIFIED_COMPLETE'
    run = Path(row['run_dir'])
    assert sha(run / 'best_map.pth') == row['checkpoint_sha256']
    assert sha(run / 'official_metrics.json') == row['receipt_sha256']
    assert max(value for path in row['full_gallery_recomputed_metric_difference_pp'].values()
               for value in path.values()) < 1e-5
prior = json.loads((root / '.git/context_identity_accepted_667_20260929.json').read_text())
rows = {(row['dataset'], row['variant']): row for row in report['rows']}
assert all(rows[(row['dataset'], row['variant'])] == row for row in prior['rows']
           if row['status'] == 'VERIFIED_COMPLETE')
output.write_bytes(accepted_path.read_bytes())
campaign_copy = root / '.git/context_identity_campaign_complete_670_20260929.json'
assert not campaign_copy.exists()
campaign_copy.write_bytes((campaign / 'campaign.json').read_bytes())
print(json.dumps({'status': 'COMPLETE15_FULL50_BEST_RELOAD_CPU_VERIFIED',
                  'collected_at': report['collected_at'], 'verified_complete': 15,
                  'previous_accepted_rows_unchanged': 13, 'matrix_sha256': sha(output)}))

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.audit_correspondence_training_losses import audit

logs = root / 'logs'
snapshot = logs / 'correspondence_refinement_accepted_seventh_m2_MSVR310_regions_634_20260929.json'
matrix = json.loads(snapshot.read_text())
assert matrix['expected_endpoints'] == 18 and matrix['verified_complete'] == 10
old = json.loads((logs / 'correspondence_refinement_accepted_sixth_m2_RGBNT100_single_633_20260929.json').read_text())
old_keys = {(row['phase'], row['dataset'], row['variant']) for row in old['rows'] if row['status'] == 'VERIFIED_COMPLETE'}
new_rows = [row for row in matrix['rows'] if row['status'] == 'VERIFIED_COMPLETE' and (row['phase'], row['dataset'], row['variant']) not in old_keys]
assert len(new_rows) == 1 and new_rows[0]['dataset'] == 'MSVR310' and new_rows[0]['variant'] == 'single_regions'
row = new_rows[0]
loss = audit(row)
loss_path = logs / 'correspondence_m2_single_regions_MSVR310_losses_634_20260929.json'
loss_path.write_text(json.dumps(loss, indent=2) + '\n')
progress_path = logs / 'correspondence_m2_progress_627_20260928.json'
progress = json.loads(progress_path.read_text())
assert progress['at'].startswith('2026-09-29T00:40:02')
progress_archive = logs / 'correspondence_m2_progress_634_004002_20260929.json'
progress_archive.write_bytes(progress_path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    'status': 'SEVENTH_COMPLETE_M2_ENDPOINT_BOUND_AND_NEW_LOG_LOSS_AUDITED',
    'at': datetime.now().astimezone().isoformat(),
    'snapshot': str(snapshot), 'snapshot_sha256': sha(snapshot), 'accepted_row': row,
    'loss_report': str(loss_path), 'loss_report_sha256': sha(loss_path),
    'progress_archive': str(progress_archive), 'progress_sha256': sha(progress_archive),
    'boundary': 'Full 50 epochs, single official mAP best, strict reload and full-gallery CPU scoring. Three of five RGBNT201 cells, three of five MSVR310 cells and one of five RGBNT100 cells are complete. At 00:40:02 the new RGBNT100 query_regions M0 is still running; no qualification pass is claimed. No seven-pair comparison or configuration selection is performed.'
}
report_path = logs / 'correspondence_m2_seventh_acceptance_634_20260929.json'
report_path.write_text(json.dumps(report, indent=2) + '\n')
driver_archive = logs / 'correspondence_m2_seventh_acceptance_driver_634.py'
driver_archive.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'row': row, 'loss_summary': {key: loss[key] for key in ('logged_steps', 'nonzero_triplet_steps', 'nonzero_prediction_steps', 'maximum_loss_reconstruction_error', 'training_sha256', 'steps_sha256', 'first', 'best', 'last')},
                  'artifacts': {str(path.relative_to(root)): sha(path) for path in (snapshot, loss_path, progress_archive, report_path, driver_archive)}}))

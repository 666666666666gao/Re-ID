from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.audit_correspondence_training_losses import audit

logs = root / 'logs'
snapshot = logs / 'correspondence_refinement_accepted_eighth_m2_RGBNT201_query_regions_635_20260929.json'
matrix = json.loads(snapshot.read_text())
assert matrix['expected_endpoints'] == 18 and matrix['verified_complete'] == 11
old = json.loads((logs / 'correspondence_refinement_accepted_seventh_m2_MSVR310_regions_634_20260929.json').read_text())
old_keys = {(row['phase'], row['dataset'], row['variant']) for row in old['rows'] if row['status'] == 'VERIFIED_COMPLETE'}
new_rows = [row for row in matrix['rows'] if row['status'] == 'VERIFIED_COMPLETE' and (row['phase'], row['dataset'], row['variant']) not in old_keys]
assert len(new_rows) == 1 and new_rows[0]['dataset'] == 'RGBNT201' and new_rows[0]['variant'] == 'query_regions'
row = new_rows[0]
loss = audit(row)
loss_path = logs / 'correspondence_m2_query_regions_RGBNT201_losses_635_20260929.json'
loss_path.write_text(json.dumps(loss, indent=2) + '\n')
progress_archive = logs / 'correspondence_m2_progress_635_012203_20260929.json'
progress = json.loads(progress_archive.read_text())
assert progress['at'].startswith('2026-09-29T01:22:03')
previous_progress = logs / 'correspondence_m2_progress_635_011722_20260929.json'
assert json.loads(previous_progress.read_text())['at'].startswith('2026-09-29T01:17:22')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    'status': 'EIGHTH_COMPLETE_M2_ENDPOINT_BOUND_AND_NEW_LOG_LOSS_AUDITED',
    'at': datetime.now().astimezone().isoformat(),
    'snapshot': str(snapshot), 'snapshot_sha256': sha(snapshot), 'accepted_row': row,
    'loss_report': str(loss_path), 'loss_report_sha256': sha(loss_path),
    'progress_archive': str(progress_archive), 'progress_sha256': sha(progress_archive),
    'previous_progress_archive': str(previous_progress), 'previous_progress_sha256': sha(previous_progress),
    'boundary': 'Full 50 epochs, single official mAP best, strict reload and full-gallery CPU scoring. Four of five RGBNT201 cells, three of five MSVR310 cells and one of five RGBNT100 cells are complete. No seven-pair comparison or configuration selection is performed.'
}
report_path = logs / 'correspondence_m2_eighth_acceptance_635_20260929.json'
report_path.write_text(json.dumps(report, indent=2) + '\n')
driver_archive = logs / 'correspondence_m2_eighth_acceptance_driver_635.py'
driver_archive.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'row': row, 'loss_summary': {key: loss[key] for key in ('logged_steps', 'nonzero_triplet_steps', 'nonzero_prediction_steps', 'maximum_loss_reconstruction_error', 'training_sha256', 'steps_sha256', 'first', 'best', 'last')},
                  'artifacts': {str(path.relative_to(root)): sha(path) for path in (snapshot, loss_path, previous_progress, progress_archive, report_path, driver_archive)}}))

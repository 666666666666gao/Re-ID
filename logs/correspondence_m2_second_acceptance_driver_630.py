from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.audit_correspondence_training_losses import audit

logs = root / 'logs'
snapshot = logs / 'correspondence_refinement_accepted_second_m2_RGBNT201_630_20260928.json'
matrix = json.loads(snapshot.read_text())
assert matrix['expected_endpoints'] == 18 and matrix['verified_complete'] == 5
old = json.loads((logs / 'correspondence_refinement_accepted_first_m2_MSVR310_629_20260928.json').read_text())
old_keys = {(row['phase'], row['dataset'], row['variant']) for row in old['rows'] if row['status'] == 'VERIFIED_COMPLETE'}
new_rows = [row for row in matrix['rows'] if row['status'] == 'VERIFIED_COMPLETE' and (row['phase'], row['dataset'], row['variant']) not in old_keys]
assert len(new_rows) == 1 and new_rows[0]['dataset'] == 'RGBNT201' and new_rows[0]['variant'] == 'single_pooled'
row = new_rows[0]
loss = audit(row)
loss_path = logs / 'correspondence_m2_single_RGBNT201_losses_630_20260928.json'
loss_path.write_text(json.dumps(loss, indent=2) + '\n')
progress_path = logs / 'correspondence_m2_progress_627_20260928.json'
progress = json.loads(progress_path.read_text())
assert progress['at'].startswith('2026-09-28T22:55:03')
progress_archive = logs / 'correspondence_m2_progress_630_225503_20260928.json'
progress_archive.write_bytes(progress_path.read_bytes())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
report = {
    'status': 'SECOND_COMPLETE_M2_ENDPOINT_BOUND_AND_NEW_LOG_LOSS_AUDITED',
    'at': datetime.now().astimezone().isoformat(),
    'snapshot': str(snapshot), 'snapshot_sha256': sha(snapshot), 'accepted_row': row,
    'loss_report': str(loss_path), 'loss_report_sha256': sha(loss_path),
    'progress_archive': str(progress_archive), 'progress_sha256': sha(progress_archive),
    'boundary': 'Full 50 epochs, single official mAP best, strict reload and full-gallery CPU scoring. One of five RGBNT201 M2 cells is complete. The parent job status can lag the complete child receipt until its 240-second refresh; this is not a restart trigger. No seven-pair comparison or configuration selection is performed.'
}
report_path = logs / 'correspondence_m2_second_acceptance_630_20260928.json'
report_path.write_text(json.dumps(report, indent=2) + '\n')
driver_archive = logs / 'correspondence_m2_second_acceptance_driver_630.py'
driver_archive.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'row': row, 'loss_summary': {key: loss[key] for key in ('logged_steps', 'nonzero_triplet_steps', 'nonzero_prediction_steps', 'maximum_loss_reconstruction_error', 'training_sha256', 'steps_sha256', 'first', 'best', 'last')},
                  'artifacts': {str(path.relative_to(root)): sha(path) for path in (snapshot, loss_path, progress_archive, report_path, driver_archive)}}))

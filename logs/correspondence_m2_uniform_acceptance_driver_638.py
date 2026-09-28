from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.audit_correspondence_training_losses import audit

logs = root / 'logs'
snapshot = logs / 'correspondence_refinement_accepted_m2_progress_638_20260929.json'
matrix = json.loads(snapshot.read_text())
assert matrix['expected_endpoints'] == 18 and matrix['verified_complete'] == 16
old = json.loads((logs / 'correspondence_refinement_accepted_m2_progress_637_20260929.json').read_text())
old_keys = {(r['phase'], r['dataset'], r['variant']) for r in old['rows'] if r['status'] == 'VERIFIED_COMPLETE'}
new_rows = [r for r in matrix['rows'] if r['status'] == 'VERIFIED_COMPLETE' and (r['phase'], r['dataset'], r['variant']) not in old_keys]
assert {(r['phase'], r['dataset'], r['variant']) for r in new_rows} == {('m2', 'RGBNT201', 'uniform_pooled'), ('m2', 'MSVR310', 'uniform_pooled')}
progress_paths = [logs / 'correspondence_m2_progress_638_023803_20260929.json', logs / 'correspondence_m2_progress_638_024800_20260929.json']
assert json.loads(progress_paths[0].read_text())['at'].startswith('2026-09-29T02:38:03.')
assert json.loads(progress_paths[1].read_text())['at'].startswith('2026-09-29T02:48:00.')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

accepted = []
artifacts = [snapshot, *progress_paths]
for row in new_rows:
    loss = audit(row)
    loss_path = logs / f"correspondence_m2_uniform_pooled_{row['dataset']}_losses_638_20260929.json"
    assert not loss_path.exists()
    loss_path.write_text(json.dumps(loss, indent=2) + '\n')
    artifacts.append(loss_path)
    accepted.append({'row': row, 'loss_report': str(loss_path), 'loss_report_sha256': sha(loss_path), 'loss_summary': {key: loss[key] for key in ('logged_steps', 'nonzero_triplet_steps', 'nonzero_prediction_steps', 'maximum_loss_reconstruction_error', 'training_sha256', 'steps_sha256', 'first', 'best', 'last')}})
report = {'status': 'TWELFTH_AND_THIRTEENTH_COMPLETE_M2_ENDPOINTS_BOUND_AND_NEW_LOG_LOSSES_AUDITED', 'at': datetime.now().astimezone().isoformat(), 'snapshot': str(snapshot), 'snapshot_sha256': sha(snapshot), 'accepted': accepted, 'progress_archives': [{'path': str(p), 'sha256': sha(p)} for p in progress_paths], 'boundary': 'Full 50 epochs, single official mAP best, strict reload and full-gallery CPU scoring. RGBNT201 and MSVR310 each have all five cells; RGBNT100 has three of five. Seven prespecified pairs can now be analyzed for the two complete datasets. No configuration selection or training change.'}
report_path = logs / 'correspondence_m2_uniform_acceptance_638_20260929.json'
assert not report_path.exists()
report_path.write_text(json.dumps(report, indent=2) + '\n')
driver_archive = logs / 'correspondence_m2_uniform_acceptance_driver_638.py'
assert not driver_archive.exists()
driver_archive.write_bytes(Path(__file__).read_bytes())
artifacts.extend([report_path, driver_archive])
print(json.dumps({'accepted': accepted, 'artifacts': {str(p.relative_to(root)): sha(p) for p in artifacts}}))

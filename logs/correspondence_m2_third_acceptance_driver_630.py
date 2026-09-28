from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.audit_correspondence_training_losses import audit
from tools.queue_correspondence_refinement import child_campaign

logs = root / 'logs'
snapshot = logs / 'correspondence_refinement_accepted_third_m2_RGBNT201_query_630_20260928.json'
matrix = json.loads(snapshot.read_text())
assert matrix['expected_endpoints'] == 18 and matrix['verified_complete'] == 6
old = json.loads((logs / 'correspondence_refinement_accepted_second_m2_RGBNT201_630_20260928.json').read_text())
old_keys = {(row['phase'], row['dataset'], row['variant']) for row in old['rows'] if row['status'] == 'VERIFIED_COMPLETE'}
new_rows = [row for row in matrix['rows'] if row['status'] == 'VERIFIED_COMPLETE' and (row['phase'], row['dataset'], row['variant']) not in old_keys]
assert len(new_rows) == 1 and new_rows[0]['dataset'] == 'RGBNT201' and new_rows[0]['variant'] == 'query_pooled'
row = new_rows[0]
loss = audit(row)
loss_path = logs / 'correspondence_m2_query_RGBNT201_losses_630_20260928.json'
loss_path.write_text(json.dumps(loss, indent=2) + '\n')
progress_path = logs / 'correspondence_m2_progress_627_20260928.json'
progress = json.loads(progress_path.read_text())
assert progress['at'].startswith('2026-09-28T23:02:03')
progress_archive = logs / 'correspondence_m2_progress_630_230203_20260928.json'
progress_archive.write_bytes(progress_path.read_bytes())
directory = child_campaign(logs / 'correspondence_refinement_20260928', 'm2', 'MSVR310', 'query_pooled')
state = json.loads((directory / 'campaign.json').read_text())
qualification = state['jobs'][0]
assert qualification['mode'] == 'm0' and qualification['status'] == 'COMPLETE' and qualification['exit_code'] == 0
m0_path = Path(qualification['output_dir']) / 'training.json'
m0 = json.loads(m0_path.read_text())
assert m0['status'] == 'M0_PASS'
assert m0['m0']['frozen_signal_unchanged']
assert m0['m0']['nonzero_gradient_parameters'] == m0['m0']['trainable_parameters']
assert m0['m0']['reload_max_abs_difference'] <= 1e-5
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
report = {
    'status': 'THIRD_COMPLETE_M2_ENDPOINT_BOUND_AND_NEW_LOG_LOSS_AUDITED',
    'at': datetime.now().astimezone().isoformat(),
    'snapshot': str(snapshot), 'snapshot_sha256': sha(snapshot), 'accepted_row': row,
    'loss_report': str(loss_path), 'loss_report_sha256': sha(loss_path),
    'progress_archive': str(progress_archive), 'progress_sha256': sha(progress_archive),
    'new_actual_m0': {'dataset': 'MSVR310', 'variant': 'query_pooled', 'receipt': str(m0_path), 'receipt_sha256': sha(m0_path), 'checks': m0['m0']},
    'boundary': 'Full 50 epochs, single official mAP best, strict reload and full-gallery CPU scoring. Two of five RGBNT201 M2 cells are complete. The parent job status can lag the complete child receipt until its 240-second refresh; this is not a restart trigger. No seven-pair comparison or configuration selection is performed.'
}
report_path = logs / 'correspondence_m2_third_acceptance_630_20260928.json'
report_path.write_text(json.dumps(report, indent=2) + '\n')
driver_archive = logs / 'correspondence_m2_third_acceptance_driver_630.py'
driver_archive.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'row': row, 'new_actual_m0': report['new_actual_m0'], 'loss_summary': {key: loss[key] for key in ('logged_steps', 'nonzero_triplet_steps', 'nonzero_prediction_steps', 'maximum_loss_reconstruction_error', 'training_sha256', 'steps_sha256', 'first', 'best', 'last')},
                  'artifacts': {str(path.relative_to(root)): sha(path) for path in (snapshot, loss_path, progress_archive, report_path, driver_archive)}}))

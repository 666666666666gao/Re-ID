from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('/data/gaob/Re-ID/Trifusion')
logs = root / 'logs'
progress_data = (logs / 'correspondence_m2_progress_627_20260928.json').read_bytes()
progress = json.loads(progress_data)
assert progress['at'].startswith('2026-09-29T02:15:04.')
assert progress['m2_status_counts'] == {'PENDING': 0, 'RUNNING': 4, 'COMPLETE': 11, 'FAILED': 0}
progress_archive = logs / 'correspondence_m2_progress_637_021504_20260929.json'
assert not progress_archive.exists()
progress_archive.write_bytes(progress_data)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = []
for dataset in ('RGBNT100', 'MSVR310'):
    state = json.loads((logs / f'correspondence_refinement_20260928/correspondence_refinement_20260928_m2_uniform_pooled_{dataset}/campaign.json').read_text())
    qualification = state['jobs'][0]
    assert qualification['mode'] == 'm0' and qualification['status'] == 'COMPLETE' and qualification['exit_code'] == 0
    path = Path(qualification['output_dir']) / 'training.json'
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'M0_PASS'
    checks = receipt['m0']
    assert checks['nonzero_gradient_parameters'] == checks['trainable_parameters'] == 118
    assert checks['frozen_signal_unchanged'] is True and checks['reload_max_abs_difference'] == 0
    rows.append({'dataset': dataset, 'variant': 'uniform_pooled', 'receipt': str(path), 'receipt_sha256': sha(path), 'checks': checks})
report = {'status': 'FOURTEENTH_AND_FIFTEENTH_ACTUAL_M2_M0_VERIFIED', 'at': datetime.now().astimezone().isoformat(), 'rows': rows, 'progress_archive': str(progress_archive), 'progress_sha256': sha(progress_archive), 'boundary': 'All fifteen actual M2 qualifications are complete; only eleven retrieval endpoints are accepted. Qualification does not establish retrieval gain. Four full training jobs continue unchanged.'}
report_path = logs / 'correspondence_m2_final_two_actual_m0_637_20260929.json'
assert not report_path.exists()
report_path.write_text(json.dumps(report, indent=2) + '\n')
driver_archive = logs / 'correspondence_m2_final_two_actual_m0_driver_637.py'
assert not driver_archive.exists()
driver_archive.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'report': report, 'artifacts': {str(p.relative_to(root)): sha(p) for p in (report_path, driver_archive, progress_archive)}}))

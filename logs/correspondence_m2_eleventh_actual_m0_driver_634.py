from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('/data/gaob/Re-ID/Trifusion')
logs = root / 'logs'
state = json.loads((logs / 'correspondence_refinement_20260928/correspondence_refinement_20260928_m2_query_regions_RGBNT100/campaign.json').read_text())
qualification = state['jobs'][0]
assert qualification['mode'] == 'm0' and qualification['status'] == 'COMPLETE' and qualification['exit_code'] == 0
path = Path(qualification['output_dir']) / 'training.json'
receipt = json.loads(path.read_text())
assert receipt['status'] == 'M0_PASS'
checks = receipt['m0']
assert checks['nonzero_gradient_parameters'] == checks['trainable_parameters'] == 118
assert checks['frozen_signal_unchanged'] is True and checks['reload_max_abs_difference'] == 0
progress_path = logs / 'correspondence_m2_progress_627_20260928.json'
progress = json.loads(progress_path.read_text())
assert progress['at'].startswith('2026-09-29T00:44:')
progress_archive = logs / 'correspondence_m2_progress_634_0044_20260929.json'
progress_archive.write_bytes(progress_path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {'status': 'ELEVENTH_ACTUAL_M2_M0_VERIFIED', 'at': datetime.now().astimezone().isoformat(),
          'dataset': 'RGBNT100', 'variant': 'query_regions', 'receipt': str(path), 'receipt_sha256': sha(path), 'checks': checks,
          'progress_archive': str(progress_archive), 'progress_sha256': sha(progress_archive),
          'boundary': 'Gradient, frozen-baseline and strict-reload qualification is not an official retrieval result; 7/15 M2 endpoints remain accepted.'}
report_path = logs / 'correspondence_m2_eleventh_actual_m0_634_20260929.json'
report_path.write_text(json.dumps(report, indent=2) + '\n')
driver_archive = logs / 'correspondence_m2_eleventh_actual_m0_driver_634.py'
driver_archive.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'report': report, 'artifacts': {str(p.relative_to(root)): sha(p) for p in (progress_archive, report_path, driver_archive)}}))

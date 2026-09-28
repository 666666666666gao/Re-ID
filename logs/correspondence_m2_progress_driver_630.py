from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools.queue_correspondence_refinement import child_campaign

logs = root / 'logs'
source = logs / 'correspondence_m2_progress_627_20260928.json'
progress = json.loads(source.read_text())
assert progress['at'].startswith('2026-09-28T23:07:02')
archive = logs / 'correspondence_m2_progress_630_230702_20260928.json'
archive.write_bytes(source.read_bytes())
directory = child_campaign(logs / 'correspondence_refinement_20260928', 'm2', 'RGBNT201', 'single_regions')
qualification = json.loads((directory / 'campaign.json').read_text())['jobs'][0]
assert qualification['mode'] == 'm0' and qualification['status'] == 'COMPLETE' and qualification['exit_code'] == 0
receipt = Path(qualification['output_dir']) / 'training.json'
m0 = json.loads(receipt.read_text())
assert m0['status'] == 'M0_PASS' and m0['m0']['frozen_signal_unchanged']
assert m0['m0']['nonzero_gradient_parameters'] == m0['m0']['trainable_parameters']
assert m0['m0']['reload_max_abs_difference'] <= 1e-5
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
report = {'at': datetime.now().astimezone().isoformat(), 'dataset': 'RGBNT201', 'variant': 'single_regions',
          'receipt': str(receipt), 'receipt_sha256': sha(receipt), 'm0': m0['m0'],
          'progress': str(archive), 'progress_sha256': sha(archive),
          'boundary': 'Read-only intake of actual M0 and actual process observation; qualification is not retrieval benefit.'}
path = logs / 'correspondence_m2_seventh_actual_m0_630_20260928.json'
path.write_text(json.dumps(report, indent=2) + '\n')
driver = logs / 'correspondence_m2_progress_driver_630.py'
driver.write_bytes(Path(__file__).read_bytes())
print(json.dumps({'report': report, 'artifacts': {str(item.relative_to(root)): sha(item) for item in (archive, path, driver)}}))

"""Complete the never-started strict evaluation/report after the recorded disk stop."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(ROOT))
from tools import queue_semantic_capacity as panel

original = ROOT / 'logs/semantic_capacity_control_v1_20261005_856'
campaign = Path(__file__).resolve().parent
report_dir = ROOT / 'results/semantic_capacity_control_v1_complete_20261005_856'
plan = json.loads((campaign / 'PLAN.json').read_text())
panel.configure()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
assert all(panel.base.sha(Path(n)) == d for n, d in plan['original_failure_sha256'].items())
assert not Path('/proc/642951').exists() and not Path('/proc/642952').exists()
assert not report_dir.exists()
old_state = json.loads((original / 'campaign.json').read_text())
assert old_state['report_invocations'] == 0
full = next(r for r in old_state['jobs'] if (r['dataset'], r['phase']) == ('RGBNT100', 'full'))
assert len(full['steps']) == 1 and full['steps'][0]['mode'] == 'train' and full['steps'][0]['exit_code'] == 0
run = panel.base.output_dir(original, 'full', 'RGBNT100', 'native')
assert not (run / 'official_metrics.json').exists() and not (original / 'RGBNT100_evaluate.log').exists()
assert panel.base.sha(run / 'training.json') == plan['training_sha256']
assert panel.base.sha(run / 'best_map.pth') == plan['best_sha256']
assert panel.base.sha(campaign / Path(__file__).name) == plan['coordinator_sha256']
panel.base.require_sources(original)
manifest = json.loads((original / 'manifest.json').read_text())
manifest['administrative_completion'] = dict(original_campaign=str(original), plan_sha256=panel.base.sha(campaign / 'PLAN.json'),
    boundary='Inherited original initializers and all three completed trainings; only first strict100 and first CPU report run here.')
panel.base.queue.write(campaign / 'manifest.json', manifest)
shutil.copytree(original / 'initialization', campaign / 'initialization')
state = json.loads(json.dumps(old_state))
state.update(status='RUNNING', controller_pid=os.getpid(), started_at=panel.base.queue.stamp(),
             original_campaign=str(original), original_parent_exit_code=1,
             inherited_training_endpoints=3, new_training_invocations=0, report_invocations=0)
panel.base.queue.write(campaign / 'campaign.json', state)
step = dict(mode='evaluate', command=panel.command('RGBNT100', 'native', 'evaluate', original, run),
            origin='FIRST_STRICT_NOT_PREVIOUSLY_STARTED')
full = next(r for r in state['jobs'] if (r['dataset'], r['phase']) == ('RGBNT100', 'full'))
full['steps'].append(step)
code = panel.previous.previous.previous.run_logged(campaign, state, step, 'RGBNT100_first_evaluate.log')
if code:
    raise SystemExit(code)
assert all(panel.base.sha(Path(n)) == d for n, d in plan['original_failure_sha256'].items())
row = panel.previous.accept_and_retire_probe(original, 'RGBNT100', 'native')
full.update(status='COMPLETE', exit_code=0, completed_at=panel.base.queue.stamp(), result=row,
            completion_origin='ORIGINAL_FULL50_PLUS_FIRST_STRICT_ADMINISTRATIVE_COMPLETION')
shutil.copytree(original / 'acceptance', campaign / 'acceptance')
shutil.copyfile(original / 'probe_retirement.jsonl', campaign / 'probe_retirement.jsonl')
rows = [panel.previous.accepted_row(campaign, dataset, 'native') for dataset in panel.DATASETS]
panel.previous.require_controls()
panel.base.queue.write(campaign / 'accepted_matrix.json', dict(schema=panel.SCHEMA, accepted=3, expected=3, rows=rows))
state.update(status='COMPLETE', completed_at=panel.base.queue.stamp(), report_invocations=1)
panel.base.queue.write(campaign / 'campaign.json', state)
assert shutil.disk_usage(ROOT).free >= 2 * 1024**3
with (campaign / 'report.log').open('x') as log:
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/report_semantic_capacity.py'),
        '--campaign', str(campaign), '--output-dir', str(report_dir)], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
state.update(report_exit_code=result.returncode, report_completed_at=panel.base.queue.stamp())
panel.base.queue.write(campaign / 'campaign.json', state)
assert all(panel.base.sha(Path(n)) == d for n, d in plan['original_failure_sha256'].items())
print(json.dumps(dict(status='FIRST_STRICT_AND_SINGLE_REPORT_FINISHED', at=datetime.now().astimezone().isoformat(),
                     first_evaluation_exit=0, report_exit=result.returncode, original_parent_exit=1,
                     new_training_invocations=0)), flush=True)
raise SystemExit(result.returncode)

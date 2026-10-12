"""Additional bounded saved-receipt links; no source execution or remote access."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
hashes, checks, failures = {}, [], []


def read(relative):
    path = BASE / relative
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def equal(name, left, right):
    checks.append(name)
    if left != right:
        failures.append(dict(check=name, actual=left, expected=right))


campaign_path = 'results911/logs/region_use_v1_20261012_906/campaign.json'
exit_path = 'results911/logs/region_use_resume906/EXIT.json'
campaign = read(campaign_path)
original_exit = read(exit_path)
snapshot = read('results908/logs/region_use_resume906/ORIGINAL_STATUS_908.json')
reports = read('results908/logs/region_use_resume906/CPU_REPORTS_908.json')
continuation = read('results912/logs/region_use_remaining_20261013_911/campaign.json')
equal('original-terminal-exit1-preserved', original_exit['exit_code'], 1)
equal('stale-original-status-retained', campaign['status'], 'RUNNING')
equal('five-complete-original-jobs', [j['status'] for j in campaign['jobs']], ['COMPLETE'] * 5 + ['PENDING'])
equal('original-sixth-unattempted', campaign['jobs'][5]['stages'], [])
equal('continuation-original-campaign-pin', continuation['original_campaign_sha256'], hashes[str(BASE / campaign_path)])
equal('continuation-original-exit-pin', continuation['original_exit_sha256'], hashes[str(BASE / exit_path)])
for index, job in enumerate(campaign['jobs'][:5]):
    label = job['dataset'] + '/' + job['comparison']
    equal(label + '/four-stages', [s['mode'] for s in job['stages']], ['prepare', 'm0', 'train', 'evaluate'])
    equal(label + '/four-real-exit-records', [(s['status'], s['exit_code']) for s in job['stages']], [('COMPLETE', 0)] * 4)
    suffix = '911' if job['dataset'] == 'RGBNT100' else '908'
    run = 'region_use_v1_20261012_906_full_' + job['comparison'] + '_' + job['dataset']
    receipt = read('results' + suffix + '/trained-model/' + run + '/official_metrics.json')
    equal(label + '/job-result-exact', job['result'], receipt)
    if index < 4:
        equal(label + '/earlier-job-unchanged', snapshot['state']['jobs'][index], job)
        equal(label + '/CPU-exit', reports['jobs'][index]['exit_code'], 0)
        equal(label + '/CPU-metrics', reports['jobs'][index]['metrics'], receipt['metrics'])
equal('four-reports-zero-NN-record', (reports['model_inference'], reports['optimizer_updates'], reports['gpu_visibility']), (0, 0, ''))
result = dict(status='PASS' if not failures else 'FAIL', checks=len(checks), check_names=checks,
              failures=failures, reviewed_hashes=hashes, interpreter=sys.executable,
              command=[sys.executable, '-B', str(Path(__file__))],
              limitation='These are historical saved records, not new observations of original five runs. Original campaign RUNNING is stale; original supervisor EXIT1 and five accepted job receipts are retained.')
(HERE / 'CHECKS_CHAIN.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], checks=len(checks), failures=failures)))
raise SystemExit(bool(failures))

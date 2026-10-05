"""Fixed-best decomposition for five accepted endpoints with mixed provenance."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(ROOT))
from tools import diagnose_native_research_best as diagnosis
from tools import run_deployment_metric_role as intervention

SCHEMA = 'trifusion-deployment-metric-role-five-fixed-best-diagnosis-v1'
EXPECTED = [('RGBNT201', 'semantic'), ('RGBNT201', 'native'), ('MSVR310', 'semantic'),
            ('RGBNT100', 'semantic'), ('RGBNT100', 'native')]
ORIGIN = ROOT / 'logs/deployment_metric_role_v1_20261005_837'
PENDING = ROOT / 'logs/deployment_metric_role_pending100_20261005_842'
CONTINUATION = ROOT / 'logs/deployment_metric_storage_continuation_20261005_848'
REPORT = ROOT / 'results/deployment_metric_storage_five_20261005_848/SUMMARY.json'

intervention.runner = diagnosis.entry.runner
diagnosis.entry = intervention
diagnosis.SCHEMA = SCHEMA
diagnosis.__file__ = str(Path(__file__).resolve())


def coordinate(args, seal):
    diagnosis.require_inputs(seal)
    report = json.loads(REPORT.read_text())
    assert report['report_work_status'] == 'COMPLETE'
    assert report['accepted_formal_endpoints'] == 5 and report['expected_formal_endpoints'] == 6
    assert report['formal_epochs'] == 250 and report['formal_steps'] == 12262
    assert len(report['pairs']) == 12 and len(report['unavailable_pairs']) == 3
    assert seal['planned_models'] == 5 and len(seal['rows']) == 8
    assert [(r['dataset'], r['variant']) for r in report['rows']] == EXPECTED
    assert {(r['dataset'], r['variant']) for r in seal['rows']} == set(EXPECTED) | {
        (dataset, 'global_only') for dataset in diagnosis.DATASETS}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    devices = subprocess.check_output(['nvidia-smi', '--id=0,1', '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    assert all(int(line.split(',')[1]) < 500 for line in devices.splitlines())
    assert shutil.disk_usage(ROOT).free > 2 * 1024**3
    assert not args.output_dir.exists()
    args.output_dir.mkdir()
    state = dict(schema=SCHEMA, status='RUNNING', started_at=diagnosis.stamp(), pid=os.getpid(),
        seal_sha256=diagnosis.panel.base.sha(args.seal), jobs=[], optimizer_updates=0,
        planned_models=5, physical_gpus=[0, 1], environment=sys.executable,
        missing_endpoint=dict(dataset='MSVR310', variant='native', status='ORIGINAL_M0_FAILED_NO_FORMAL_WEIGHT'))
    diagnosis.write(args.output_dir / 'campaign.json', state)
    for dataset, variant in EXPECTED:
        campaign = (PENDING if variant == 'semantic' else CONTINUATION) if dataset == 'RGBNT100' else ORIGIN
        row = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
        assert row['artifact_campaign'] == str(campaign)
        job = dict(dataset=dataset, variant=variant, artifact_campaign=str(campaign),
                   status='RUNNING', started_at=diagnosis.stamp())
        state['jobs'].append(job)
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--campaign', str(campaign),
            '--seal', str(args.seal), '--output-dir', str(args.output_dir), '--dataset', dataset, '--variant', variant]
        with (args.output_dir / f'{dataset}_{variant}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            job['pid'] = process.pid
            diagnosis.write(args.output_dir / 'campaign.json', state)
            code = process.wait()
        job.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=diagnosis.stamp())
        diagnosis.write(args.output_dir / 'campaign.json', state)
        if code:
            state.update(status='FAILED', completed_at=diagnosis.stamp())
            diagnosis.write(args.output_dir / 'campaign.json', state)
            return code
    diagnosis.require_inputs(seal)
    state.update(status='COMPLETE', completed_at=diagnosis.stamp())
    diagnosis.write(args.output_dir / 'campaign.json', state)
    return 0


diagnosis.coordinate = coordinate

if __name__ == '__main__':
    intervention.configure()
    raise SystemExit(diagnosis.main())

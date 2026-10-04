"""First semantic evaluation after disk interruption, then never-started native."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(ROOT))
from tools import queue_deployment_metric_role as original

ORIGIN = ROOT / 'logs/deployment_metric_role_v1_20261005_837'
PENDING = ROOT / 'logs/deployment_metric_role_pending100_20261005_842'
CAMPAIGN = ROOT / 'logs/deployment_metric_storage_continuation_20261005_848'
ORIGIN_SHA = '88e2575a7d3693afc0e9e02d0ed3352e61e0eb02447263b8a5c4ff1612db62f9'
PENDING_SHA = '4d5b9e00e2b3739a59c0ae411c89b28b4955701b5bb16780eaab84e54130223d'


def main():
    original.configure()
    base = original.base
    runner = original.previous.previous
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert base.sha(ORIGIN / 'campaign.json') == ORIGIN_SHA
    assert base.sha(PENDING / 'campaign.json') == PENDING_SHA
    pending = json.loads((PENDING / 'campaign.json').read_text())
    semantic = next(j for j in pending['jobs'] if (j['variant'], j['phase']) == ('semantic', 'full'))
    assert semantic['status'] == 'PENDING' and len(semantic['steps']) == 1
    assert semantic['steps'][0]['mode'] == 'train' and semantic['steps'][0]['exit_code'] == 0
    assert all(j['status'] == 'PENDING' for j in pending['jobs'] if j['variant'] == 'native')
    assert not any(p['variant'] == 'native' for p in pending['preparation'])
    run = base.output_dir(PENDING, 'full', 'RGBNT100', 'semantic')
    training = json.loads((run / 'training.json').read_text())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert [r['epoch'] for r in training['history']] == list(range(1, 51))
    assert not (run / 'official_metrics.json').exists()
    assert not (PENDING / 'RGBNT100_semantic_evaluate.log').exists()
    assert not CAMPAIGN.exists()
    # Admission covers first evaluation plus one native best and live probe;
    # the existing per-command 2GiB reserve remains unchanged.
    assert shutil.disk_usage(ROOT).free >= 2 * 384 * 1024**2 + 2 * 1024**3
    controls, sources = original.require_controls(), original.source_map()
    jobs = [dict(dataset='RGBNT100', variant='semantic', phase='first_evaluate', status='PENDING')]
    jobs += [dict(dataset='RGBNT100', variant='native', phase=p, status='PENDING') for p in ('m0', 'full')]
    CAMPAIGN.mkdir()
    witness = PENDING / 'initialization/RGBNT100_semantic.json'
    manifest = dict(schema=original.SCHEMA, seed=42, epochs=50, source_sha256=sources,
        initialization_sha256={str(witness): base.sha(witness)}, jobs=jobs,
        physical_gpus=[0, 1], max_parallel_jobs=1, poll_seconds=240,
        control_seal_sha256=base.sha(original.CONTROLS), role_metric_policy=original.METRIC_POLICY,
        continuation_source_sha256=base.sha(Path(__file__)),
        original_campaign_sha256=ORIGIN_SHA, pending_campaign_sha256=PENDING_SHA,
        boundary='Administrative storage continuation only: original semantic full50 already completed; first strict evaluation has never started. Original parent EXIT1 and both campaign snapshots stay unchanged. Native is originally never-started,own initializer/eight-step M0/fresh50/firststrict. Same scientific339/control187,seed/recipe/precision/thresholds. No MSVR native retry,no report846 execution,no power-temperature action.')
    state = dict(status='RUNNING', controller_pid=os.getpid(), started_at=base.queue.stamp(),
                 jobs=jobs, preparation=[], report_invocations=0)
    base.queue.write(CAMPAIGN / 'manifest.json', manifest)
    base.queue.write(CAMPAIGN / 'campaign.json', state)
    job = jobs[0]
    step = dict(mode='evaluate', command=original.command('RGBNT100', 'semantic', 'evaluate', PENDING, run))
    job['steps'] = [step]
    if runner.run_logged(CAMPAIGN, state, step, 'RGBNT100_semantic_evaluate.log'):
        return 1
    row = original.accept_and_retire_probe(PENDING, 'RGBNT100', 'semantic')
    job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(), result=row,
               artifact_campaign=str(PENDING))
    base.queue.write(CAMPAIGN / 'campaign.json', state)
    prep = dict(dataset='RGBNT100', variant='native', mode='prepare',
        command=original.command('RGBNT100', 'native', 'prepare', CAMPAIGN,
                                 ROOT / 'trained-model/storage_continuation_prepare_unused'))
    state['preparation'].append(prep)
    if runner.run_logged(CAMPAIGN, state, prep, 'prepare_RGBNT100_native.log'):
        return 1
    witness = CAMPAIGN / 'initialization/RGBNT100_native.json'
    binding = base.expected_binding(CAMPAIGN, 'RGBNT100', 'native')
    control = next(r['initializer'] for r in controls['rows'] if (r['dataset'], r['variant']) == ('RGBNT100', 'native'))
    excluded = ('architecture', 'entry_sha256', 'scope', 'role_metric_policy')
    assert {k:v for k,v in binding.items() if k not in excluded} == {k:v for k,v in control.items() if k not in excluded}
    manifest['initialization_sha256'][str(witness)] = base.sha(witness)
    base.queue.write(CAMPAIGN / 'manifest.json', manifest)
    for phase in ('m0', 'full'):
        if phase == 'full':
            runner.verify_m0(CAMPAIGN, 'RGBNT100', 'native')
        job = next(j for j in jobs if (j['variant'], j['phase']) == ('native', phase))
        job['steps'] = []
        for mode in (('m0',) if phase == 'm0' else ('train', 'evaluate')):
            step = dict(mode=mode, command=original.command('RGBNT100', 'native', mode, CAMPAIGN,
                                base.output_dir(CAMPAIGN, phase, 'RGBNT100', 'native')))
            job['steps'].append(step)
            if runner.run_logged(CAMPAIGN, state, step, f'RGBNT100_native_{mode}.log'):
                job.update(status='FAILED', exit_code=step['exit_code'])
                base.queue.write(CAMPAIGN / 'campaign.json', state)
                return 1
        row = runner.verify_m0(CAMPAIGN, 'RGBNT100', 'native') if phase == 'm0' else original.accept_and_retire_probe(CAMPAIGN, 'RGBNT100', 'native')
        job.update(status='COMPLETE', exit_code=0, completed_at=base.queue.stamp(), result=row)
        base.queue.write(CAMPAIGN / 'campaign.json', state)
    rows = [original.accepted_row(PENDING, 'RGBNT100', 'semantic'),
            original.accepted_row(CAMPAIGN, 'RGBNT100', 'native')]
    assert base.sha(ORIGIN / 'campaign.json') == ORIGIN_SHA
    assert base.sha(PENDING / 'campaign.json') == PENDING_SHA
    original.require_controls()
    base.queue.write(CAMPAIGN / 'accepted_matrix.json', dict(schema=original.SCHEMA,
        accepted=2, expected=2, rows=rows, artifact_campaigns=[str(PENDING), str(CAMPAIGN)],
        original_campaign_sha256=ORIGIN_SHA, pending_campaign_sha256=PENDING_SHA))
    state.update(status='COMPLETE', completed_at=base.queue.stamp(), report_invocations=0)
    base.queue.write(CAMPAIGN / 'campaign.json', state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

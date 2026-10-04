"""Once-only CPU report of five closed endpoints and one original M0 failure."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(ROOT))
import torch
from tools import queue_deployment_metric_role as panel
from tools.report_deployment_metric_role import paired

ORIGIN = ROOT / 'logs/deployment_metric_role_v1_20261005_837'
PENDING = ROOT / 'logs/deployment_metric_role_pending100_20261005_842'
CONTINUATION = ROOT / 'logs/deployment_metric_storage_continuation_20261005_848'
OUTPUT = ROOT / 'results/deployment_metric_storage_five_20261005_848'
PENDING_SHA = '4d5b9e00e2b3739a59c0ae411c89b28b4955701b5bb16780eaab84e54130223d'
STORAGE_COORDINATOR_SHA = 'f4e347dc0b22a8d0b3ec226d01ea21a737ec6840c13711681afc64ffaf7a6aba'
ORIGIN_SHA = '88e2575a7d3693afc0e9e02d0ed3352e61e0eb02447263b8a5c4ff1612db62f9'
COORDINATOR_SHA = '19ac57ad4172be5e60f0ff48a4b6acc41e6f5de4761701662a485075d3b88b94'
EXPECTED = [('RGBNT201', 'semantic'), ('RGBNT201', 'native'), ('MSVR310', 'semantic'),
            ('RGBNT100', 'semantic'), ('RGBNT100', 'native')]


def verify_campaigns():
    assert panel.base.sha(ORIGIN / 'campaign.json') == ORIGIN_SHA
    assert panel.base.sha(PENDING / 'campaign.json') == PENDING_SHA
    origin = json.loads((ORIGIN / 'campaign.json').read_text())
    pending = json.loads((PENDING / 'campaign.json').read_text())
    continuation = json.loads((CONTINUATION / 'campaign.json').read_text())
    assert origin['status'] == 'FAILED' and 'active_command' not in origin
    assert continuation['status'] == 'COMPLETE' and 'active_command' not in continuation
    assert origin['report_invocations'] == pending['report_invocations'] == continuation['report_invocations'] == 0
    assert [(j['dataset'],j['variant']) for j in origin['jobs'] if j['phase']=='full' and j['status']=='COMPLETE'] == EXPECTED[:3]
    failed = next(j for j in origin['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','native','m0'))
    assert failed['status']=='FAILED' and failed['exit_code']==1
    assert all(j['status']=='PENDING' for j in origin['jobs'] if j['dataset']=='RGBNT100')
    semantic = next(j for j in pending['jobs'] if (j['variant'],j['phase'])==('semantic','full'))
    assert semantic['status']=='PENDING' and len(semantic['steps'])==1
    assert semantic['steps'][0]['mode']=='train' and semantic['steps'][0]['exit_code']==0
    assert all(j['status']=='PENDING' for j in pending['jobs'] if j['variant']=='native')
    exit_path = ROOT/'logs/deployment_metric_pending100_launch_20261005_843/EXIT.json'
    assert panel.base.sha(exit_path)=='57418a8ba131304cf38feb34195a47da211e2a15ffbc023bccbb5006591a2aac'
    assert json.loads(exit_path.read_text())['exit_code']==1
    assert len(continuation['jobs'])==3 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in continuation['jobs'])
    assert [(j['variant'],j['phase']) for j in continuation['jobs']]==[('semantic','first_evaluate'),('native','m0'),('native','full')]
    matrix = json.loads((CONTINUATION/'accepted_matrix.json').read_text())
    assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==2
    assert [(r['dataset'],r['variant']) for r in matrix['rows']]==EXPECTED[3:]
    assert matrix['artifact_campaigns']==[str(PENDING),str(CONTINUATION)]
    assert matrix['original_campaign_sha256']==ORIGIN_SHA and matrix['pending_campaign_sha256']==PENDING_SHA
    manifest = json.loads((CONTINUATION/'manifest.json').read_text())
    assert manifest['continuation_source_sha256']==STORAGE_COORDINATOR_SHA
    assert manifest['original_campaign_sha256']==ORIGIN_SHA and manifest['pending_campaign_sha256']==PENDING_SHA
    assert panel.base.sha(ROOT/'refine-logs/deployment_metric_role_v1/STORAGE_CONTINUATION_848.py')==STORAGE_COORDINATOR_SHA
    assert panel.base.sha(ROOT/'refine-logs/deployment_metric_role_v1/PENDING_RGBNT100_QUEUE.py')==COORDINATOR_SHA
    for campaign in (ORIGIN,PENDING,CONTINUATION):
        panel.base.require_sources(campaign)
    assert not (ORIGIN/'acceptance/MSVR310_native.json').exists()
    assert not (ROOT/'trained-model/deployment_metric_role_v1_20261005_837_full_native_MSVR310').exists()
    assert not (ORIGIN/'accepted_matrix.json').exists() and not (PENDING/'accepted_matrix.json').exists()
    return origin, continuation, matrix


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not OUTPUT.exists()
    panel.configure()
    origin, continuation, continuation_matrix = verify_campaigns()
    controls = panel.require_controls()
    torch.set_num_threads(1)
    rows = []
    for dataset, variant in EXPECTED:
        campaign = (PENDING if variant=='semantic' else CONTINUATION) if dataset=='RGBNT100' else ORIGIN
        row = panel.accepted_row(campaign, dataset, variant)
        if dataset == 'RGBNT100':
            assert row == next(r for r in continuation_matrix['rows'] if r['variant'] == variant)
        run = Path(row['run_dir'])
        training = json.loads((run / 'training.json').read_text())
        assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
        assert [r['epoch'] for r in training['history']] == list(range(1, 51))
        assert training['best_epoch'] == row['best_epoch']
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        batches = [json.loads(line) for line in (run / 'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps) == len(batches) == sum(r['steps'] for r in training['history'])
        assert [(r['epoch'], r['batch']) for r in steps] == [(r['epoch'], r['batch']) for r in batches]
        control = next(r for r in controls['rows'] if (r['dataset'], r['variant']) == (dataset, variant))
        assert (run / 'training_batch_order.jsonl').read_bytes() == (Path(control['run_dir']) / 'training_batch_order.jsonl').read_bytes()
        rows.append({**row, 'formal_steps': len(steps), 'history': training['history'],
            'training_and_epoch_evaluation_seconds': (datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds()})
    assert sum(r['formal_steps'] for r in rows) == 12262
    combined = dict(rows=[dict(r, variant='control_' + r['variant']) for r in controls['rows']] + rows)
    pairs = []
    for row in rows:
        for control in ('control_' + row['variant'], 'control_global_only'):
            pair = paired(combined, row['dataset'], control, row['variant'])
            pair['actual_training_batch_order_equal'] = True
            pairs.append(pair)
    for dataset in ('RGBNT201', 'RGBNT100'):
        pairs.append(paired(combined, dataset, 'semantic', 'native'))
    assert len(pairs) == 12
    missing = [dict(dataset='MSVR310', control=control, candidate='native', status='UNAVAILABLE_ORIGINAL_M0_FAILED')
               for control in ('control_native', 'control_global_only', 'semantic')]
    result = dict(schema='trifusion-deployment-metric-role-storage-five-report-v1',
        status='PARTIAL_FIVE_FORMAL_ENDPOINTS_AND_ONE_ORIGINAL_M0_FAILURE', report_work_status='COMPLETE',
        created_at=datetime.now().astimezone().isoformat(), expected_formal_endpoints=6, accepted_formal_endpoints=5,
        formal_epochs=250, formal_steps=12262, rows=rows, pairs=pairs, unavailable_pairs=missing,
        original_failure=dict(dataset='MSVR310', variant='native', phase='m0', exit_code=1,
            origin_campaign_sha256=ORIGIN_SHA, original_job=next(j for j in origin['jobs']
                if (j['dataset'], j['variant'], j['phase']) == ('MSVR310', 'native', 'm0'))),
        primary_matched_role_progress_count=sum(p['phase_progress'] for p in pairs if p['control'] == 'control_' + p['variant']),
        independent_global_progress_count=sum(p['phase_progress'] for p in pairs if p['control'] == 'control_global_only'),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        storage_provenance=dict(original_pending_sha256=PENDING_SHA,continuation_campaign=str(CONTINUATION),continuation_sha256=panel.base.sha(CONTINUATION/'campaign.json')),
        boundary=('Five accepted original fresh50/first-strict endpoints only, with 12 complete saved-array comparisons and '
            'three explicitly unavailable comparisons. Original six-endpoint campaign stays FAILED and report_invocations=0; '
            'The original pending100 parent EXIT1 and snapshot remain unchanged; semantic is first-evaluated from its original full50, native is first-trained in the distinct storage continuation. This partial report does not establish six-endpoint completion, '
            'the full goal, SOTA or training-seed significance. No model inference, replay of retired M0 probes, '
            'best reselection, seed search, reranking, power/temperature action or failure rescue. '
            'Identity bootstrap resamples fixed-model identities on consumed official benchmarks.'))
    verify_campaigns()
    panel.require_controls()
    assert not OUTPUT.exists()
    OUTPUT.mkdir(parents=True)
    panel.base.queue.write(OUTPUT / 'SUMMARY.json', result)
    lines = ['# Five completed endpoints; one original M0 failure', '', result['boundary'], '',
        '| Dataset | Variant | Selected epoch | mAP | Rank-1 | Steps |', '|---|---|---:|---:|---:|---:|']
    lines.extend(f'| {r["dataset"]} | {r["variant"]} | {r["best_epoch"]} | {r["metrics"]["mAP"]:.6f} | {r["metrics"]["Rank-1"]:.6f} | {r["formal_steps"]} |' for r in rows)
    lines.extend(['', 'MSVR310 native: original M0 failed; no formal score and no retraining.', '',
        'Available all-query pairs: 12/15. Unavailable: raw-native → native, global-only → native, semantic → native on MSVR310.'])
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], accepted_formal_endpoints=5, formal_steps=12262,
        available_pairs=12, unavailable_pairs=3)), flush=True)


if __name__ == '__main__':
    main()

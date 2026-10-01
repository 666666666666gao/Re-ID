"""Archive the actual milestone, complete endpoints and six production M0s."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
DEST = ROOT / '.codex_tmp/visual_start_milestone_698_20261001'
TAR = DEST.with_suffix('.tar.gz')
assert not DEST.exists() and not TAR.exists()
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 221
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
           for name, digest in manifest['source_sha256'].items())
observer = json.loads((ROOT / 'logs/visual_start_roles_observer_1145_20261001.json').read_text())
outer = json.loads((ROOT / 'logs/visual_start_roles_observer_1145_wrapper_20261001.json').read_text())
milestone = CAMPAIGN / 'milestone_1145_snapshot.json'
assert observer['status'] == outer['status'] == 'COMPLETE'
assert observer['exit_code'] == outer['exit_code'] == 0
assert hashlib.sha256(milestone.read_bytes()).hexdigest() == observer['snapshot_sha256']
state = json.loads((CAMPAIGN / 'campaign.json').read_text())
files = [CAMPAIGN / 'manifest.json', CAMPAIGN / 'campaign.json', milestone,
         ROOT / 'logs/visual_start_roles_observer_1145_20261001.json',
         ROOT / 'logs/visual_start_roles_observer_1145_wrapper_20261001.json',
         ROOT / 'logs/visual_start_roles_wrapper_20261001_v1.json',
         ROOT / 'logs/visual_start_roles_controller_20261001_v1.log',
         ROOT / 'logs/visual_start_milestone_sync697_20261001.json',
         ROOT / 'logs/visual_start_roles_analysis_waiter_wrapper_20261001.json',
         CAMPAIGN / 'analysis_waiter_status.json',
         Path(manifest['inputs_path']), Path(manifest['initialization_witness_path']),
         ROOT / 'logs/visual_start_roles_observer_1315_launch_20261001.json',
         ROOT / 'logs/visual_start_roles_observer_1315_wrapper_20261001.json',
         ROOT / '.codex_tmp/visual_training_boundary_20261001/SOURCE_AUDIT.json',
         ROOT / '.codex_tmp/audit_visual_training_boundary_20261001.py']
for name in ('preserve_visual_start_milestone_698_20261001.py',
             'observe_visual_start_roles_1315_20261001.py',
             'launch_visual_start_observer_1315_20261001.py'):
    files.append(ROOT / '.codex_tmp' / name)
rows = []
support = []
now = datetime.now().astimezone()
for row in state['jobs']:
    item = dict(row)
    child = CAMPAIGN / (CAMPAIGN.name + '_visual_start_' + row['variant'] + '_' + row['dataset'])
    child_state = json.loads((child / 'campaign.json').read_text())
    item['child'] = child_state
    files.append(child / 'campaign.json')
    for stage in child_state['jobs']:
        files.append(child / (stage['mode'] + '.log'))
        run = Path(stage['output_dir'])
        if stage['mode'] == 'm0' and stage['status'] == 'COMPLETE':
            m0_path = run / 'training.json'
            m0 = json.loads(m0_path.read_text())
            assert m0['status'] == 'M0_PASS' and stage['exit_code'] == 0
            assert m0['m0']['nonzero_gradient_parameters'] == m0['m0']['trainable_parameters'] == 128
            assert m0['m0']['frozen_signal_unchanged'] and m0['m0']['reload_max_abs_difference'] == 0
            item['m0'] = m0
            steps_path = run / 'training_steps.jsonl'
            steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
            assert len(steps) == 8 and [s['batch'] for s in steps] == list(range(8))
            support.append({'dataset': row['dataset'], 'variant': row['variant'], 'batches': 8,
                            'nonzero_triplet_batches': sum(s['triplet'] > 0 for s in steps),
                            'mean_id': sum(s['id'] for s in steps) / 8,
                            'mean_triplet': sum(s['triplet'] for s in steps) / 8,
                            'source_sha256': hashlib.sha256(steps_path.read_bytes()).hexdigest()})
            files.extend((m0_path, steps_path))
        if stage['mode'] == 'train':
            path = run / 'training_steps.jsonl'
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 8192))
                last = json.loads(stream.read().splitlines()[-1])
            item['progress'] = {'last_epoch': last['epoch'], 'last_batch': last['batch'],
                                'train_started_at': stage['started_at'],
                                'scope': 'Epoch/batch progress only; no interim official score analysis.'}
            if stage['status'] == 'RUNNING':
                elapsed = (now - datetime.fromisoformat(stage['started_at'])).total_seconds()
                item['progress']['rough_train_end_estimate'] = datetime.fromtimestamp(
                    now.timestamp() + elapsed * (50 - last['epoch']) / last['epoch'],
                    tz=now.tzinfo).isoformat()
        if stage['status'] == 'RUNNING':
            process = Path('/proc') / str(stage['pid'])
            item['running_process_exists'] = process.exists()
            if process.exists():
                item['running_command'] = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if row['status'] == 'COMPLETE':
        assert child_state['status'] == 'COMPLETE' and row['exit_code'] == 0
        assert all(s['status'] == 'COMPLETE' and s['exit_code'] == 0 for s in child_state['jobs'])
        assert child_state['verification']['status'] == 'VERIFIED_COMPLETE'
        run = Path(child_state['jobs'][2]['output_dir'])
        for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
            files.append(run / name)
        official = json.loads((run / 'official_metrics.json').read_text())
        assert official['training_epochs'] == 50 and not official['reranking']
        for name, field in (('best_map.pth', 'checkpoint_sha256'), ('official_distances.pt', 'distance_sha256')):
            assert hashlib.sha256((run / name).read_bytes()).hexdigest() == official[field]
        item['complete_endpoint'] = official
    rows.append(item)
record = {'observed_at': datetime.now().astimezone().isoformat(), 'campaign': state, 'rows': rows,
          'source_count_verified': 221, 'disk_free_bytes': shutil.disk_usage(ROOT).free,
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits']).decode(),
          'scope': 'Actual complete full50 endpoints and stage/progress records; remaining endpoints unaccepted. No interim score selection, rerun or experiment mutation.'}
DEST.mkdir()
entries = []
for path in dict.fromkeys(files):
    content = path.read_bytes()
    target = DEST / 'raw' / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    entries.append({'path': target.relative_to(DEST).as_posix(), 'bytes': len(content),
                    'sha256': hashlib.sha256(content).hexdigest()})
(DEST / 'SNAPSHOT.json').write_text(json.dumps(record, indent=2) + '\n')
(DEST / 'M0_INITIAL_SUPPORT.json').write_text(json.dumps({
    'calculated_at': record['observed_at'], 'rows': support,
    'scope': 'Six real-label eight-batch production M0s. Source-task loss support only; not retrieval gain, true optimizer update share or a sole causal explanation.'}, indent=2) + '\n')
(DEST / 'INTAKE.json').write_text(json.dumps({'captured_at': record['observed_at'], 'raw_files': entries}, indent=2) + '\n')
with tarfile.open(TAR, 'w:gz') as archive:
    archive.add(DEST, arcname='.')
print(json.dumps({'observed_at': record['observed_at'], 'archive': str(TAR),
                  'archive_bytes': TAR.stat().st_size, 'archive_sha256': hashlib.sha256(TAR.read_bytes()).hexdigest(),
                  'raw_files': len(entries), 'source_count_verified': 221,
                  'gpu': record['gpu'], 'disk_free_bytes': record['disk_free_bytes'],
                  'complete': sum(r['status'] == 'COMPLETE' for r in rows),
                  'running': sum(r['status'] == 'RUNNING' for r in rows),
                  'pending': sum(r['status'] == 'PENDING' for r in rows)}))

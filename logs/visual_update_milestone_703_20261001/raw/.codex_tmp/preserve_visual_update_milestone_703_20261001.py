"""Archive a dated text-only milestone from the existing twelve-end queue."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_update_control_20261001_v1'
OWN = ROOT / '.codex_tmp/visual_update_milestone_intake_703_20261001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


now = datetime.now().astimezone()
assert now.isoformat() >= '2026-10-01T15:10:00+08:00'
assert not OWN.exists()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == 'e75486f07a02465f5f11e2e61afe1467e64b6b14'
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert sha(CAMPAIGN / 'manifest.json') == '947b028e835a6c722fac47f5e6333f7f02838ec9b1fbd2beda35ec1885543e5c'
assert len(manifest['source_sha256']) == 222
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
assert sha(Path(manifest['preflight_path'])) == manifest['preflight_sha256']
assert sha(ROOT / 'tools/report_visual_update_control_complete.py') == '4c21b1cc6212ef0202512ef46afc8fc714e3b2329fd68f701e4a7ac24acd4765'
state = json.loads((CAMPAIGN / 'campaign.json').read_text())
waiter = json.loads((CAMPAIGN / 'analysis_waiter_status.json').read_text())
assert len(state['jobs']) == 12 and waiter['report_invocations'] == 0
paths = list(CAMPAIGN.glob('*.json')) + list(CAMPAIGN.glob('*.log'))
paths += list((CAMPAIGN / 'milestone_snapshots').glob('*.json'))
paths += [ROOT / 'logs/visual_update_analysis_waiter_20261001.json',
          ROOT / 'logs/visual_update_milestone_sync702_20261001.json', Path(__file__)]
snapshot = {'observed_at': now.isoformat(), 'parent_status': state['status'],
            'controller_pid': state['controller_pid'], 'parent_jobs': state['jobs'],
            'bound_source_count': 222, 'manifest_sha256': sha(CAMPAIGN / 'manifest.json'),
            'free_disk_bytes': shutil.disk_usage(ROOT).free,
            'gpu_state': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                '--format=csv,noheader,nounits'], text=True),
            'analysis_waiter': waiter, 'endpoints': [], 'processes': [],
            'boundary': 'Sequential dated text snapshot, not an atomic realtime view. Existing worker verification only; no model/array replay, inference or complete report.'}
pid_list = [state['controller_pid'], waiter['pid']]
for job in state['jobs']:
    child = CAMPAIGN / f"{CAMPAIGN.name}_visual_update_{job['variant']}_{job['dataset']}"
    if not (child / 'campaign.json').exists():
        continue
    paths += list(child.glob('*.json')) + list(child.glob('*.log'))
    item = json.loads((child / 'campaign.json').read_text())
    endpoint = {'dataset': job['dataset'], 'variant': job['variant'], 'gpu': job['gpu'],
                'parent_status': job['status'], 'child_status': item['status'], 'stages': item['jobs']}
    if item['status'] == 'COMPLETE':
        endpoint['verification'] = item['verification']
    for stage in item['jobs']:
        run = Path(stage['output_dir'])
        for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
            if (run / name).exists():
                paths.append(run / name)
        if stage['mode'] == 'train' and (run / 'training.json').exists():
            receipt = json.loads((run / 'training.json').read_text())
            count = len(receipt['history'])
            endpoint.update(recorded_complete_epochs=count, training_started_at=receipt['started_at'],
                            training_status=receipt['status'])
            if count and count < 50:
                elapsed = (now - datetime.fromisoformat(receipt['started_at'])).total_seconds()
                endpoint['estimated_remaining_seconds'] = elapsed * (50 - count) / count
        if stage['status'] == 'RUNNING':
            pid_list.append(stage['pid'])
    if job['status'] == 'RUNNING':
        pid_list.append(job['pid'])
    snapshot['endpoints'].append(endpoint)
snapshot['formal_parent_complete'] = sum(row['status'] == 'COMPLETE' for row in state['jobs'])
snapshot['child_verified_complete'] = sum(row['child_status'] == 'COMPLETE' for row in snapshot['endpoints'])
for pid in pid_list:
    stat = Path('/proc') / str(pid) / 'stat'
    snapshot['processes'].append({'pid': pid, 'present': stat.exists()})
OWN.mkdir()
copied = []
for source in sorted(set(paths)):
    relative = source.relative_to(ROOT)
    destination = OWN / 'raw' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    destination.write_bytes(content)
    if source.suffix == '.json':
        json.loads(content)
    copied.append({'path': relative.as_posix(), 'bytes': len(content), 'sha256': sha(destination)})
snapshot_path = OWN / 'SNAPSHOT.json'
snapshot_path.write_text(json.dumps(snapshot, indent=2) + '\n')
archive = ROOT / '.codex_tmp/visual_update_milestone_intake_703_20261001.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as bundle:
    bundle.add(OWN / 'raw', arcname='raw')
    bundle.add(snapshot_path, arcname='SNAPSHOT.json')
receipt = {'preserved_at': datetime.now().astimezone().isoformat(), 'snapshot_at': now.isoformat(),
           'source_sha256': sha(Path(__file__)), 'files': copied,
           'snapshot_sha256': sha(snapshot_path), 'archive_sha256': sha(archive),
           'archive_bytes': archive.stat().st_size,
           'boundary': 'Existing twelve-end milestone text only; all neural weights, tensors and images remain remote. No duplicate observer/report/queue.'}
(ROOT / '.codex_tmp/visual_update_milestone_intake_703_20261001.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'preserved_at': receipt['preserved_at'], 'files': len(copied),
                  'formal_parent_complete': snapshot['formal_parent_complete'],
                  'child_verified_complete': snapshot['child_verified_complete'],
                  'archive_sha256': receipt['archive_sha256'], 'archive_bytes': receipt['archive_bytes'],
                  'progress': [{key: row[key] for key in ('dataset', 'variant', 'gpu', 'parent_status',
                                                       'child_status', 'recorded_complete_epochs',
                                                       'estimated_remaining_seconds') if key in row}
                               for row in snapshot['endpoints']]}))

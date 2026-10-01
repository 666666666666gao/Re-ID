"""Preserve one dated launch snapshot without changing running experiments."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OWN = ROOT / '.codex_tmp/visual_update_launch_intake_701_20261001'
CAMPAIGN = ROOT / 'logs/visual_update_control_20261001_v1'
PREFLIGHT = ROOT / 'logs/visual_update_preflight_20261001_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


now = datetime.now().astimezone()
assert now.isoformat() >= '2026-10-01T14:06:30+08:00', now.isoformat()
assert not OWN.exists()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == '1d25150bd05ddb10b5b2b21e75b5c710b3a07793'
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['jobs']) == 12 and manifest['schema'] == 'trifusion-visual-update-panel-v1'
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
original = ROOT / 'logs/visual_start_roles_20261001_v1/manifest.json'
old = json.loads(original.read_text())
assert sha(original) == 'c749e3089a775c5781435fe14bfb7a1b75228f59004c9a31ec26d2a9409706dc'
assert len(old['source_sha256']) == 221
assert all(sha(ROOT / path) == digest for path, digest in old['source_sha256'].items())
paths = [CAMPAIGN / 'manifest.json', CAMPAIGN / 'campaign.json', PREFLIGHT / 'PRECHECK.json',
         PREFLIGHT / 'INITIALIZATION_WITNESS.json',
         ROOT / 'logs/visual_update_control_launch_20261001_v1.json',
         ROOT / 'logs/visual_start_complete_sync700_20261001.json',
         ROOT / '.codex_tmp/visual_update_140130_snapshot_20261001.json', Path(__file__)]
paths += list((ROOT / '.codex_tmp/visual_update_launch_20261001').glob('*.json'))
paths += list((ROOT / '.codex_tmp/visual_update_launch_20261001').glob('*.log'))
paths += list(PREFLIGHT.glob('*.log'))
paths.append(ROOT / '.codex_tmp/launch_visual_update_panel_20261001.py')
state = json.loads((CAMPAIGN / 'campaign.json').read_text())
snapshot = {'observed_at': now.isoformat(), 'parent_status': state['status'],
            'controller_pid': state['controller_pid'], 'parent_jobs': state['jobs'],
            'bound_source_count': len(manifest['source_sha256']),
            'manifest_sha256': sha(CAMPAIGN / 'manifest.json'),
            'original_source_count_verified': 221,
            'free_disk_bytes': shutil.disk_usage(ROOT).free,
            'gpu_state': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                '--format=csv,noheader,nounits'], text=True),
            'endpoints': [], 'm0_results': [], 'processes': []}
pid_list = [state['controller_pid']]
for job in state['jobs']:
    child = CAMPAIGN / f"{CAMPAIGN.name}_visual_update_{job['variant']}_{job['dataset']}"
    if not (child / 'campaign.json').exists():
        continue
    paths += list(child.glob('*.json')) + list(child.glob('*.log'))
    item = json.loads((child / 'campaign.json').read_text())
    endpoint = {'dataset': job['dataset'], 'variant': job['variant'], 'gpu': job['gpu'],
                'parent_status': job['status'], 'child_status': item['status'], 'stages': item['jobs']}
    for stage in item['jobs']:
        run = Path(stage['output_dir'])
        for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
            if (run / name).exists():
                paths.append(run / name)
        if stage['mode'] == 'm0' and (run / 'training.json').exists():
            r = json.loads((run / 'training.json').read_text())
            if r['status'] == 'M0_PASS':
                assert r['history'][0]['steps'] == 8
                assert r['m0']['nonzero_gradient_parameters'] == r['m0']['trainable_parameters']
                assert r['m0']['frozen_signal_state_unchanged']
                assert r['m0']['visual_parameters_changed'] == job['variant'].startswith('low_lr_')
                snapshot['m0_results'].append({'dataset': job['dataset'], 'variant': job['variant'],
                                                'm0': r['m0'], 'completed_at': r['completed_at']})
        if stage['mode'] == 'train' and (run / 'training.json').exists():
            r = json.loads((run / 'training.json').read_text())
            endpoint['recorded_complete_epochs'] = len(r['history'])
            endpoint['training_started_at'] = r['started_at']
            endpoint['training_status'] = r['status']
            count = len(r['history'])
            if count:
                seconds = (now - datetime.fromisoformat(r['started_at'])).total_seconds()
                endpoint['estimated_remaining_seconds'] = seconds * (50 - count) / count
                endpoint['eta_boundary'] = 'Elapsed since training initialization divided by observed complete epochs; approximate, not actual completion.'
        if stage['status'] == 'RUNNING':
            pid_list.append(stage['pid'])
    if job['status'] == 'RUNNING':
        pid_list.append(job['pid'])
    snapshot['endpoints'].append(endpoint)
for pid in pid_list:
    stat = Path('/proc') / str(pid) / 'stat'
    assert stat.exists(), pid
    snapshot['processes'].append({'pid': pid, 'stat': stat.read_text().split(') ', 1)[1].split()[0]})
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
archive = ROOT / '.codex_tmp/visual_update_launch_intake_701_20261001.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as bundle:
    bundle.add(OWN / 'raw', arcname='raw')
    bundle.add(snapshot_path, arcname='SNAPSHOT.json')
receipt = {'preserved_at': datetime.now().astimezone().isoformat(), 'snapshot_at': now.isoformat(),
           'source_sha256': sha(Path(__file__)), 'files': copied,
           'snapshot_sha256': sha(snapshot_path), 'archive_sha256': sha(archive),
           'archive_bytes': archive.stat().st_size,
           'boundary': 'One dated text/JSON launch snapshot; training continues, no model/weight/array/image exported.'}
(ROOT / '.codex_tmp/visual_update_launch_intake_701_20261001.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'preserved_at': receipt['preserved_at'], 'files': len(copied),
                  'm0_pass': len(snapshot['m0_results']),
                  'running_endpoints': len(snapshot['endpoints']),
                  'gpu_state': snapshot['gpu_state'], 'archive_sha256': receipt['archive_sha256'],
                  'archive_bytes': receipt['archive_bytes'],
                  'progress': [{k: row[k] for k in ('dataset', 'variant', 'gpu', 'recorded_complete_epochs',
                                                   'estimated_remaining_seconds') if k in row}
                               for row in snapshot['endpoints']]}))

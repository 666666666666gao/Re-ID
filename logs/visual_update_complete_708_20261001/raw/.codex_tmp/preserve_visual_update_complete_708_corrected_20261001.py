"""Preserve existing complete twelve-end receipts and the once-executed CPU report."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_update_control_20261001_v1'
REPORT = ROOT / 'results/visual_update_control_complete_20261001'
OWN = ROOT / '.codex_tmp/visual_update_complete_intake_708_20261001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


now = datetime.now().astimezone()
assert now.isoformat() >= '2026-10-01T17:40:00+08:00'
assert not OWN.exists()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == 'd72b62deb7ef87aef71f690c6a55d4bd707b92ad'
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert sha(CAMPAIGN / 'manifest.json') == '947b028e835a6c722fac47f5e6333f7f02838ec9b1fbd2beda35ec1885543e5c'
assert len(manifest['source_sha256']) == 222
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
assert sha(Path(manifest['preflight_path'])) == manifest['preflight_sha256']
assert sha(ROOT / 'tools/report_visual_update_control_complete.py') == '4c21b1cc6212ef0202512ef46afc8fc714e3b2329fd68f701e4a7ac24acd4765'
state = json.loads((CAMPAIGN / 'campaign.json').read_text())
waiter = json.loads((CAMPAIGN / 'analysis_waiter_status.json').read_text())
wrapper_path = ROOT / 'logs/visual_update_analysis_waiter_20261001.json'
wrapper = json.loads(wrapper_path.read_text())
assert state['status'] == 'COMPLETE' and len(state['jobs']) == 12
assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
assert waiter['status'] == 'CPU_REPORT_COMPLETE' and waiter['report_invocations'] == 1 and waiter['report_exit_code'] == 0
assert wrapper['status'] == 'COMPLETE' and wrapper['exit_code'] == 0
accepted = json.loads((CAMPAIGN / 'accepted_matrix.json').read_text())
assert accepted['verified_complete'] == accepted['expected_endpoints'] == 12
summary = json.loads((REPORT / 'SUMMARY.json').read_text())
assert summary['accepted'] == 12 and len(summary['rows']) == 12
paths = list(CAMPAIGN.glob('*.json')) + list(CAMPAIGN.glob('*.log'))
paths += list((CAMPAIGN / 'milestone_snapshots').glob('*.json'))
paths += [wrapper_path, ROOT / 'logs/visual_update_milestone_sync707_20261001.json', Path(__file__),
          ROOT / '.codex_tmp/preserve_visual_update_complete_708_20261001.py',
          ROOT / 'tools/report_visual_update_control_complete.py',
          ROOT / '.codex_tmp/wait_visual_update_complete_analysis_20261001.py']
paths += [REPORT / name for name in ('SUMMARY.json', 'REPORT.md', 'trajectories.png', 'trajectories.svg')]
snapshot = {'observed_at': now.isoformat(), 'parent_status': state['status'],
            'controller_pid': state['controller_pid'], 'parent_jobs': state['jobs'],
            'bound_source_count': 222, 'manifest_sha256': sha(CAMPAIGN / 'manifest.json'),
            'free_disk_bytes': shutil.disk_usage(ROOT).free,
            'analysis_waiter': waiter, 'analysis_wrapper': wrapper, 'endpoints': [],
            'report_summary_sha256': sha(REPORT / 'SUMMARY.json'),
            'report_markdown_sha256': sha(REPORT / 'REPORT.md'),
            'boundary': 'Sequential preservation of existing terminal evidence only. No queue, observer, CPU report, neural or array replay.'}
for job in state['jobs']:
    child = CAMPAIGN / f"{CAMPAIGN.name}_visual_update_{job['variant']}_{job['dataset']}"
    paths += list(child.glob('*.json')) + list(child.glob('*.log'))
    item = json.loads((child / 'campaign.json').read_text())
    assert item['status'] == 'COMPLETE' and item['verification']['status'] == 'VERIFIED_COMPLETE'
    assert len(item['jobs']) == 3 and all(stage['status'] == 'COMPLETE' and stage['exit_code'] == 0 for stage in item['jobs'])
    endpoint = {'dataset': job['dataset'], 'variant': job['variant'], 'gpu': job['gpu'],
                'parent_status': job['status'], 'child_status': item['status'],
                'stages': item['jobs'], 'verification': item['verification']}
    for stage in item['jobs']:
        run = Path(stage['output_dir'])
        for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
            if (run / name).exists():
                paths.append(run / name)
        if stage['mode'] == 'train':
            receipt = json.loads((run / 'training.json').read_text())
            assert len(receipt['history']) == 50 and receipt['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
            endpoint['recorded_complete_epochs'] = len(receipt['history'])
    snapshot['endpoints'].append(endpoint)
snapshot['formal_parent_complete'] = snapshot['child_verified_complete'] = 12
snapshot['processes'] = [{'pid': pid, 'present': (Path('/proc') / str(pid)).exists()}
                         for pid in (2337321, 2342941, 2372978, 2372984, waiter['report_pid'])]
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
archive = ROOT / '.codex_tmp/visual_update_complete_intake_708_20261001.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as bundle:
    bundle.add(OWN / 'raw', arcname='raw')
    bundle.add(snapshot_path, arcname='SNAPSHOT.json')
receipt = {'preserved_at': datetime.now().astimezone().isoformat(), 'snapshot_at': now.isoformat(),
           'source_sha256': sha(Path(__file__)), 'files': copied,
           'snapshot_sha256': sha(snapshot_path), 'archive_sha256': sha(archive),
           'archive_bytes': archive.stat().st_size,
           'boundary': 'All12 existing terminals and original once-executed CPU report. Neural weights, raw images and NPY remain remote; report figures included.'}
(ROOT / '.codex_tmp/visual_update_complete_intake_708_20261001.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'preserved_at': receipt['preserved_at'], 'files': len(copied), 'accepted': 12,
                  'report_invocations': waiter['report_invocations'], 'actual_report_exit_code': waiter['report_exit_code'],
                  'report_completed_at': waiter['report_completed_at'],
                  'archive_sha256': receipt['archive_sha256'], 'archive_bytes': receipt['archive_bytes'],
                  'registered_visual_update_gate': summary['registered_visual_update_gate'],
                  'registered_role_gate': summary['registered_role_gate']}))

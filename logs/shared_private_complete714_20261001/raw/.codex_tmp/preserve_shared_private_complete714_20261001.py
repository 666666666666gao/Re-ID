"""Receive the existing complete report and the two newly completed endpoints."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/shared_private_evidence_20261001_v2'
REPORT = ROOT / 'results/shared_private_evidence_complete_20261001'
DEST = ROOT / '.codex_tmp/shared_private_complete714_intake_20261001'
PREFIX = 'logs/shared_private_complete714_20261001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    waiter = json.loads((CAMPAIGN / 'analysis_waiter_status.json').read_text())
    assert waiter['status'] == 'CPU_REPORT_COMPLETE'
    assert waiter['report_invocations'] == 1 and waiter['report_actual_exit_code'] == 0
    snapshot = Path(waiter['last_snapshot'])
    assert snapshot.parent == CAMPAIGN / 'milestone_snapshots'
    measured = json.loads(snapshot.read_text())
    assert measured['formal_accepted'] == 9 and measured['parent_status'] == 'COMPLETE'
    assert measured['wrapper_status'] == 'COMPLETE'
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    old = json.loads((ROOT / 'logs/visual_update_control_20261001_v1/manifest.json').read_text())
    for item, count in ((manifest, 229), (old, 222)):
        assert len(item['source_sha256']) == count
        assert all(sha(ROOT / name) == value for name, value in item['source_sha256'].items())
        for name in ('preflight', 'initialization_witness'):
            assert sha(Path(item[name + '_path'])) == item[name + '_sha256']
    processes = {}
    for name, pid in (('wrapper', 2840894), ('controller', 2846591), ('analysis_waiter', 2910065), ('cpu_report', waiter['report_pid'])):
        proc = Path('/proc') / str(pid)
        processes[name] = {'pid': pid, 'exists': proc.exists(),
                           'command': (proc / 'cmdline').read_bytes().replace(b'\x00', b' ').decode() if proc.exists() else None}
    disk = shutil.disk_usage(ROOT)
    storage_path = ROOT / 'logs/shared_private_storage714_20261001.json'
    assert not storage_path.exists()
    storage_path.write_text(json.dumps({'observed_at': datetime.now().astimezone().isoformat(),
                                      'processes': processes, 'disk': dict(zip(('total', 'used', 'free'), disk)),
                                      'scope': 'One terminal process/storage check. No neural or score call, no retirement.'}, indent=2) + '\n')
    paths = {snapshot, storage_path, Path(__file__),
             ROOT / 'logs/shared_private_milestone_sync713_20261001.json',
             ROOT / 'logs/shared_private_evidence_launch_20261001_v2.json',
             ROOT / 'logs/shared_private_analysis_waiter_20261001.json',
             ROOT / 'logs/shared_private_analysis_waiter_20261001.log'}
    paths.update(CAMPAIGN / name for name in ('manifest.json', 'campaign.json', 'accepted_matrix.json',
                                             'analysis_waiter_status.json', 'complete_analysis.log'))
    paths.update(Path(manifest[key]) for key in ('preflight_path', 'initialization_witness_path'))
    wanted = {('RGBNT100', 'separated_roles'), ('RGBNT100', 'global_only')}
    for dataset, variant in sorted(wanted):
        folder = CAMPAIGN / f'{CAMPAIGN.name}_shared_private_{variant}_{dataset}'
        paths.update(folder.glob('*.json'))
        paths.update(folder.glob('*.log'))
        state = json.loads((folder / 'campaign.json').read_text())
        assert state['status'] == 'COMPLETE'
        assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
        for stage in state['jobs']:
            run = Path(stage['output_dir'])
            paths.update(run / name for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json') if (run / name).exists())
    for path in sorted(paths):
        assert path.is_file()
        target = DEST / PREFIX / 'raw' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for path in sorted(REPORT.iterdir()):
        assert path.is_file() and path.suffix in ('.json', '.md', '.png', '.svg')
        target = DEST / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    record = {'archived_at': datetime.now().astimezone().isoformat(),
              'milestone_snapshot': str(snapshot), 'milestone_snapshot_sha256': sha(snapshot),
              'report_completed_at': waiter['completed_at'], 'report_invocations': 1,
              'active_sources_verified': 229, 'sealed_sources_verified': 222, 'accepted': 9,
              'scope': 'Existing full CPU report and terminal receipts, only two newly completed RGBNT100 endpoints. No replay, score recomputation or weight transfer; arrays/checkpoints stay remote.'}
    (DEST / PREFIX / 'ARCHIVE.json').write_text(json.dumps(record, indent=2) + '\n')
    files = {str(path.relative_to(DEST)): {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in DEST.rglob('*') if path.is_file()}
    archive = ROOT / '.codex_tmp/shared_private_complete714_intake_20261001.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(DEST)))
    record.update(archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, files=files)
    (ROOT / '.codex_tmp/shared_private_complete714_intake_20261001.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'files'}))


if __name__ == '__main__':
    main()

"""Archive the new completed endpoints and an existing milestone snapshot only."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/shared_private_evidence_20261001_v2'
DEST = ROOT / '.codex_tmp/shared_private_milestone713_intake_20261001'
PREFIX = 'logs/shared_private_milestone713_20261001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    request = json.loads((ROOT / '.codex_tmp/shared_private_milestone713_request_20261001.json').read_text())
    snapshot = Path(request['snapshot'])
    assert snapshot.parent == CAMPAIGN / 'milestone_snapshots'
    measured = json.loads(snapshot.read_text())
    assert measured['formal_accepted'] == 7 and measured['parent_status'] == 'RUNNING'
    wanted = {('RGBNT100', 'coupled_roles'), ('MSVR310', 'global_only')}
    complete = {(row['dataset'], row['variant']): row for row in measured['parent_jobs'] if row['status'] == 'COMPLETE'}
    assert wanted <= complete.keys()
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    old = json.loads((ROOT / 'logs/visual_update_control_20261001_v1/manifest.json').read_text())
    for item, count in ((manifest, 229), (old, 222)):
        assert len(item['source_sha256']) == count
        assert all(sha(ROOT / name) == value for name, value in item['source_sha256'].items())
        assert sha(Path(item['preflight_path'])) == item['preflight_sha256']
        assert sha(Path(item['initialization_witness_path'])) == item['initialization_witness_sha256']
    paths = {snapshot, CAMPAIGN / 'manifest.json', CAMPAIGN / 'campaign.json',
             CAMPAIGN / 'analysis_waiter_status.json',
             ROOT / 'logs/shared_private_storage713_20261001.json',
             ROOT / 'logs/shared_private_milestone_sync712_20261001.json',
             ROOT / '.codex_tmp/check_shared_private_storage713_20261001.py',
             ROOT / '.codex_tmp/preserve_shared_private_milestone713_20261001.py'}
    paths.update(Path(manifest[key]) for key in ('preflight_path', 'initialization_witness_path'))
    for dataset, variant in sorted(wanted):
        folder = CAMPAIGN / f'{CAMPAIGN.name}_shared_private_{variant}_{dataset}'
        paths.update(folder.glob('*.json'))
        paths.update(folder.glob('*.log'))
        state = json.loads((folder / 'campaign.json').read_text())
        assert state['status'] == 'COMPLETE' and all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in state['jobs'])
        for stage in state['jobs']:
            run = Path(stage['output_dir'])
            paths.update(run / name for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json') if (run / name).exists())
    for path in sorted(paths):
        assert path.is_file()
        target = DEST / PREFIX / 'raw' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    record = {'archived_at': datetime.now().astimezone().isoformat(),
              'milestone_observed_at': measured['observed_at'],
              'milestone_snapshot': str(snapshot), 'milestone_snapshot_sha256': sha(snapshot),
              'active_sources_verified': 229, 'sealed_sources_verified': 222,
              'accepted_at_snapshot': measured['formal_accepted'],
              'scope': 'Only new completed coupledRGBNT100/globalMSVR endpoints, existing snapshot and source/storage/previous-sync receipts. No model/score call or duplicate archive of all prior endpoints. Arrays and weights stay remote.'}
    (DEST / PREFIX / 'ARCHIVE.json').write_text(json.dumps(record, indent=2) + '\n')
    files = {str(path.relative_to(DEST)): {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in DEST.rglob('*') if path.is_file()}
    archive = ROOT / '.codex_tmp/shared_private_milestone713_intake_20261001.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(DEST)))
    record.update({'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size, 'files': files})
    (ROOT / '.codex_tmp/shared_private_milestone713_intake_20261001.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'files'}))


if __name__ == '__main__':
    main()

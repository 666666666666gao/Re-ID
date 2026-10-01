"""Preserve the existing observer milestone and current text receipts; no neural calls."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/shared_private_evidence_20261001_v2'
PREFIX = 'logs/shared_private_milestone711_20261001'
DEST = ROOT / '.codex_tmp/shared_private_milestone711_intake_20261001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    old = json.loads((ROOT / 'logs/visual_update_control_20261001_v1/manifest.json').read_text())
    for item, count in ((manifest, 229), (old, 222)):
        assert len(item['source_sha256']) == count
        assert all(sha(ROOT / name) == value for name, value in item['source_sha256'].items())
        assert sha(Path(item['preflight_path'])) == item['preflight_sha256']
        assert sha(Path(item['initialization_witness_path'])) == item['initialization_witness_sha256']
    snapshot = CAMPAIGN / 'milestone_snapshots/20261001_200613.json'
    measured = json.loads(snapshot.read_text())
    assert measured['formal_accepted'] == 3 and measured['parent_status'] == 'RUNNING'
    pair = [row for row in measured['parent_jobs'] if row['dataset'] == 'RGBNT201' and row['status'] == 'COMPLETE']
    assert {row['variant'] for row in pair} == {'coupled_roles', 'separated_roles'}
    assert pair[0]['result']['initial_model_state_sha256'] == pair[1]['result']['initial_model_state_sha256']
    assert pair[0]['result']['trainable_parameters'] == pair[1]['result']['trainable_parameters']
    paths = set(CAMPAIGN.rglob('*.json')) | set(CAMPAIGN.rglob('*.log'))
    paths.update(Path(manifest['preflight_path']).parent.glob('*'))
    paths.add(ROOT / 'logs/shared_private_report_ready_sync710_20261001.json')
    paths.add(ROOT / '.codex_tmp/preserve_shared_private_milestone711_20261001.py')
    for path in CAMPAIGN.rglob('campaign.json'):
        for row in json.loads(path.read_text())['jobs']:
            if 'output_dir' not in row:
                continue
            run = Path(row['output_dir'])
            paths.update(run / name for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json') if (run / name).exists())
    for path in sorted(paths):
        assert path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.py')
        target = DEST / PREFIX / 'raw' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    record = {'archived_at': datetime.now().astimezone().isoformat(),
              'milestone_observed_at': measured['observed_at'],
              'milestone_snapshot': str(snapshot), 'milestone_snapshot_sha256': sha(snapshot),
              'active_source_count_verified': 229, 'old_source_count_verified': 222,
              'manifest_sha256': sha(CAMPAIGN / 'manifest.json'),
              'scope': 'Existing 20:06 observer snapshot plus later timestamped raw text receipts. No new process/GPU query, scoring or training. Weights/arrays remain remote.'}
    (DEST / PREFIX / 'ARCHIVE.json').write_text(json.dumps(record, indent=2) + '\n')
    files = {str(path.relative_to(DEST)): {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in DEST.rglob('*') if path.is_file()}
    archive = ROOT / '.codex_tmp/shared_private_milestone711_intake_20261001.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(DEST)))
    record.update({'archive': str(archive), 'archive_sha256': sha(archive),
                   'archive_bytes': archive.stat().st_size, 'files': files})
    (ROOT / '.codex_tmp/shared_private_milestone711_intake_20261001.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'files'}))


if __name__ == '__main__':
    main()

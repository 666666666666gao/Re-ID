"""Preserve the actual 08:45 endpoint/stage update as a separate immutable intake."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/role_global_tokens_20261001_v1'
SNAPSHOT = json.loads((CAMPAIGN / 'milestone_0845_snapshot.json').read_text())
DEST = ROOT / '.codex_tmp/role_global_tokens_milestone_691_20261001'
assert not DEST.exists()
DEST.mkdir()
paths = [CAMPAIGN / 'milestone_0845_snapshot.json', CAMPAIGN / 'campaign.json']
paths.append(ROOT / '.codex_tmp/observe_role_global_tokens_milestone_0845_20261001.py')
for row in SNAPSHOT['rows']:
    folder = CAMPAIGN / (CAMPAIGN.name + '_role_global_tokens_' + row['variant'] + '_' + row['dataset'])
    if row.get('stages'):
        paths.append(folder / 'campaign.json')
    for stage in row.get('stages', []):
        paths.append(folder / (stage['mode'] + '.log'))
        output = Path(stage['output_dir'])
        paths += [output / name for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json') if (output / name).exists()]
paths += [ROOT / name for name in (
    'logs/role_global_tokens_milestone_sync690_20261001.json',
    '.codex_tmp/sync_role_global_tokens_690_20261001.py',
    '.codex_tmp/observe_role_global_tokens_milestone_0845_20261001.py',
    '.codex_tmp/launch_role_global_tokens_observer_0845_20261001.py',
    'logs/role_global_tokens_observer_0845_20261001.json')]
observer = json.loads((ROOT / 'logs/role_global_tokens_observer_0845_20261001.json').read_text())
assert observer['status'] == 'COMPLETE' and observer['exit_code'] == 0

assert observer['source_sha256'] == hashlib.sha256((ROOT / '.codex_tmp/observe_role_global_tokens_milestone_0845_20261001.py').read_bytes()).hexdigest()
paths += [ROOT / 'logs/role_global_tokens_observer_0845_20261001.log', ROOT / 'logs/role_global_tokens_observer_0845_20261001_wrapper.log']
paths = list(dict.fromkeys(paths))
entries = []
for path in paths:
    assert path.resolve().is_relative_to(ROOT.resolve())
    assert path.suffix in ('.json', '.jsonl', '.log', '.py')
    relative = path.relative_to(ROOT)
    data = path.read_bytes()
    target = DEST / 'files' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    entries.append({'source': str(relative), 'path': str(target.relative_to(DEST)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
record = {'preserved_at': datetime.now().astimezone().isoformat(), 'snapshot_observed_at': SNAPSHOT['observed_at'],
          'scope': 'Separate 08:45 text update; live receipts can be newer than snapshot. No model/image/distance export.', 'files': entries}
(DEST / 'INTAKE_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
archive = DEST.with_suffix('.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(DEST, arcname='role_global_tokens_milestone_691_20261001')
print(json.dumps({'archive': str(archive), 'files': len(entries), 'bytes': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'preserved_at': record['preserved_at']}))
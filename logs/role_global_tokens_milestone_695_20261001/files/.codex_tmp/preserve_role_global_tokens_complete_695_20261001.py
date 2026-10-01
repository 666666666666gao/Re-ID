"""Archive the completed nine-end campaign and its already executed CPU report."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
C = ROOT / 'logs/role_global_tokens_20261001_v1'
DEST = ROOT / '.codex_tmp/role_global_tokens_milestone_695_20261001'
assert not DEST.exists()
snapshot = json.loads((C / 'milestone_1030_snapshot.json').read_text())
parent = json.loads((C / 'campaign.json').read_text())
matrix = json.loads((C / 'accepted_matrix.json').read_text())
assert parent['status'] == 'COMPLETE'
assert len(parent['jobs']) == matrix['verified_complete'] == 9
assert all(j['status'] == 'COMPLETE' and j['exit_code'] == 0 for j in parent['jobs'])
report_dir = ROOT / 'results/role_global_tokens_complete_20261001'
report = json.loads((report_dir / 'SUMMARY.json').read_text())
assert report['accepted'] == 9
waiter = json.loads((ROOT / 'logs/role_global_tokens_analysis_waiter_20261001.json').read_text())
observer = json.loads((ROOT / 'logs/role_global_tokens_observer_1030_20261001.json').read_text())
assert waiter['status'] == observer['status'] == 'COMPLETE'
assert waiter['exit_code'] == observer['exit_code'] == 0
paths = [C / name for name in ('campaign.json','manifest.json','accepted_matrix.json','milestone_1030_snapshot.json','analysis_waiter_status.json')]
for row in snapshot['rows']:
    child = C / (C.name + '_role_global_tokens_' + row['variant'] + '_' + row['dataset'])
    paths.append(child / 'campaign.json')
    for stage in row['stages']:
        paths.append(child / (stage['mode'] + '.log'))
        output = Path(stage['output_dir'])
        paths += [output / name for name in ('training.json','training_steps.jsonl','official_metrics.json') if (output/name).is_file()]
paths += list(report_dir.iterdir())
paths += [ROOT / name for name in (
    'logs/role_global_tokens_milestone_sync694_20261001.json',
    '.codex_tmp/sync_role_global_tokens_694_20261001.py',
    '.codex_tmp/observe_role_global_tokens_milestone_1030_20261001.py',
    '.codex_tmp/launch_role_global_tokens_observer_1030_20261001.py',
    'logs/role_global_tokens_observer_1030_20261001.json',
    'logs/role_global_tokens_observer_1030_20261001.log',
    'logs/role_global_tokens_observer_1030_20261001_wrapper.log',
    '.codex_tmp/wait_role_global_tokens_complete_analysis_20261001.py',
    '.codex_tmp/launch_role_global_tokens_analysis_waiter_20261001.py',
    'logs/role_global_tokens_analysis_waiter_20261001.json',
    'logs/role_global_tokens_analysis_waiter_20261001.log',
    'logs/role_global_tokens_analysis_waiter_20261001_wrapper.log',
    '.codex_tmp/preserve_role_global_tokens_complete_695_20261001.py')]
paths = list(dict.fromkeys(paths))
for path in paths:
    assert path.is_file(), path
    assert path.resolve().is_relative_to(ROOT.resolve())
    assert path.suffix in ('.json','.jsonl','.log','.py','.md','.png','.svg')
DEST.mkdir()
entries = []
for path in paths:
    data = path.read_bytes()
    target = DEST / 'files' / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    entries.append({'source': str(path.relative_to(ROOT)), 'path': str(target.relative_to(DEST)),
                    'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
record = {'preserved_at': datetime.now().astimezone().isoformat(), 'snapshot_observed_at': snapshot['observed_at'],
          'scope': 'Complete nine-end terminal text evidence and executed CPU analysis; no PT, NPY or dataset images exported.', 'files': entries}
(DEST / 'INTAKE_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
archive = DEST.with_suffix('.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(DEST, arcname=DEST.name)
print(json.dumps({'archive': str(archive), 'files': len(entries), 'bytes': archive.stat().st_size,
                  'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'preserved_at': record['preserved_at']}))

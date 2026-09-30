"""Archive completed six-end receipts; leave all weights and arrays remote."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root / 'logs/patch_memory_roles_recovery_20261001'
target = root / 'logs/patch_memory_complete_intake_20261001'
observer = root / 'logs/patch_memory_recovery_final_observer_20261001_0235.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


state = json.loads((campaign / 'campaign.json').read_text())
matrix = json.loads((campaign / 'accepted_matrix.json').read_text())
observed = json.loads(observer.read_text())
assert state['status'] == observed['status'] == 'COMPLETE'
assert matrix['verified_complete'] == matrix['expected_endpoints'] == observed['accepted_count'] == 6
assert observed['accepted_matrix_sha256'] == sha(campaign / 'accepted_matrix.json')
manifest = json.loads((campaign / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 210
assert all(sha(root / path) == digest for path, digest in manifest['source_sha256'].items())
assert not target.exists()
target.mkdir()
files = {}


def copy(source, name):
    destination = target / name
    assert source.resolve().is_relative_to(root.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    shutil.copyfile(source, destination)
    assert sha(source) == sha(destination)
    files[name] = {'source': str(source), 'sha256': sha(source), 'bytes': source.stat().st_size}


for name in ('campaign.json', 'accepted_matrix.json', 'manifest.json', 'recovery_manifest.json'):
    copy(campaign / name, name)
copy(observer, 'observer_0235.json')
copy(root / '.codex_tmp/patch_memory_recovery_final_observer_20261001_0235.log', 'observer_0235.log')
copy(Path(__file__).resolve(), 'intake_source.py')
for row in matrix['rows']:
    assert row['status'] == 'VERIFIED_COMPLETE'
    folder = Path(row['campaign_dir'])
    label = row['dataset'] + '_' + row['variant']
    copy(folder / 'campaign.json', label + '/campaign.json')
    if row['dataset'] != 'RGBNT100':
        continue
    run, m0 = Path(row['run_dir']), Path(row['m0_run_dir'])
    for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
        copy(run / name, label + '/' + name)
    for name in ('m0.log', 'train.log', 'evaluate.log'):
        copy(folder / name, label + '/' + name)
    for name in ('training.json', 'training_steps.jsonl'):
        copy(m0 / name, label + '/m0/' + name)
    assert sha(run / 'best_map.pth') == row['checkpoint_sha256']
    assert sha(run / 'official_distances.pt') == row['distance_sha256']

receipt = {'copied_at': datetime.now().astimezone().isoformat(), 'files': files,
           'scope': 'Six-end accepted matrix and raw new RGBNT100 receipts. Earlier four raw runs '
                    'remain in their original archived intakes. No binary download or score change.'}
(target / 'INTAKE.json').write_text(json.dumps(receipt, indent=2) + '\n')
archive = root / '.codex_tmp/patch_memory_complete_intake_20261001.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as stream:
    stream.add(target, arcname=target.relative_to(root))
print(json.dumps({'status': 'COMPLETE_TEXT_INTAKE', 'copied_files': len(files),
                  'accepted': matrix['verified_complete'], 'archive': str(archive),
                  'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
                  'intake_sha256': sha(target / 'INTAKE.json')}, indent=2))

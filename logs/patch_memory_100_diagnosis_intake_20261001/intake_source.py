from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root = Path('/data/gaob/Re-ID/Trifusion')
source = root / 'logs/patch_memory_100_diagnosis_20261001'
target = root / 'logs/patch_memory_100_diagnosis_intake_20261001'
state = json.loads((source / 'campaign.json').read_text())
assert state['status'] == 'COMPLETE'
assert all(job['status'] == 'COMPLETE' and job['exit_code'] == 0 for job in state['jobs'])
assert not target.exists()
target.mkdir()
files = {}


def copy(path, name):
    destination = target / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    data = path.read_bytes()
    shutil.copyfile(path, destination)
    assert destination.read_bytes() == data == path.read_bytes()
    files[name] = {'source': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


for name in ('campaign.json', 'cpu.log', 'slots.log', 'cpu/RGBNT100.json',
             'slots/local_memory.json', 'slots/full_memory.json', 'slots/COMPLETE.json'):
    copy(source / name, name)
copy(root / 'logs/patch_memory_100_diagnosis_launch_20261001.json', 'LAUNCH.json')
for name in ('launch_patch_memory_100_diagnosis_20261001.py', 'start_patch_memory_100_diagnosis_20261001.py'):
    copy(root / '.codex_tmp' / name, name)
copy(Path(__file__), 'intake_source.py')
(target / 'INTAKE.json').write_text(json.dumps({'copied_at': datetime.now().astimezone().isoformat(),
    'files': files, 'scope': 'Original complete diagnostic text and exact producer sources only'}, indent=2) + '\n')
archive = root / '.codex_tmp/patch_memory_100_diagnosis_intake_20261001.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as stream:
    stream.add(target, arcname=target.relative_to(root))
print(json.dumps({'archive': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'bytes': archive.stat().st_size, 'copied_files': len(files)}))

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root = Path('/data/gaob/Re-ID/Trifusion')
output = root / 'logs/slot_competition_m0_failure_intake_20261001'
assert not output.exists()
output.mkdir()
members = []


def copy(source, relative):
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    members.append({'member': relative, 'original': str(source), 'bytes': target.stat().st_size,
                    'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})


campaign = root / 'logs/slot_competition_roles_20261001'
for name in ('campaign.json', 'manifest.json'):
    copy(campaign / name, name)
copy(root / 'logs/slot_competition_launch_20261001.json', 'LAUNCH.json')
copy(root / 'logs/slot_competition_controller_20261001.log', 'controller_snapshot.log')
for folder in sorted(campaign.glob(campaign.name + '_slot_competition_*')):
    state = json.loads((folder / 'campaign.json').read_text())
    stage = state['jobs'][0]
    assert stage['mode'] == 'm0'
    label = state['dataset'] + '_' + state['variant']
    copy(folder / 'campaign.json', label + '/campaign_snapshot.json')
    copy(folder / 'm0.log', label + '/m0.log')
    for name in ('training.json', 'training_steps.jsonl'):
        copy(Path(stage['output_dir']) / name, label + '/' + name)
for label, stem in (('gradient_capture', 'slot_competition_m0_gradient_20261001'),
                    ('counterfactual_fp32', 'slot_competition_m0_fp32_20261001')):
    for extension in ('json', 'log'):
        copy(root / ('logs/' + stem + '.' + extension), label + '/capture.' + extension)
    run = root / ('trained-model/' + stem)
    for name in ('training.json', 'training_steps.jsonl'):
        copy(run / name, label + '/' + name)
for name in ('start_slot_competition_20261001.py', 'check_slot_competition_launch_20261001.py',
             'diagnose_slot_m0_gradient_20261001.py', 'diagnose_slot_m0_fp32_20261001.py'):
    copy(root / '.codex_tmp' / name, 'sources/' + name)
copy(Path(__file__), 'intake_source.py')
report = {'captured_at': datetime.now().astimezone().isoformat(), 'members': members,
          'scope': 'Original failed competitive M0, completed control M0 receipts and mutable campaign snapshots; two explicit8-step diagnostics. Temporary FP32 instrumentation is not a new production M0 or retrieval result. No original receipt changed.'}
(output / 'INTAKE.json').write_text(json.dumps(report, indent=2) + '\n')
archive = root / '.codex_tmp/slot_competition_m0_failure_intake_20261001.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            tar.add(path, arcname=path.relative_to(output))
print(json.dumps({'members': len(members), 'archive': str(archive),
                  'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))

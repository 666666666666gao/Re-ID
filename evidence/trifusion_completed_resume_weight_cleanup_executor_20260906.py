from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess
import time

directory = Path('/root/autodl-tmp/trifusion-v2/transport')
plan_path = directory / 'completed_resume_weight_cleanup_plan_20260906.json'
receipt_path = directory / 'completed_resume_weight_cleanup_receipt_20260906.json'
assert not receipt_path.exists()
plan = json.loads(plan_path.read_text())
allowed = Path(plan['allowed_root']).resolve(strict=True)
runs = plan['runs']
assert len(runs) == 12 and sum(len(r['delete']) for r in runs) == 24


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


started = time.perf_counter()
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
assert not [line for line in processes if any(r['run'] in line for r in runs)]
before = subprocess.check_output(['df', '-B1', '/root/autodl-tmp', '/root/trifusion-storage'], text=True)
checked = []
for run in runs:
    parent = Path(run['run'])
    assert parent.resolve(strict=True).is_relative_to(allowed)
    latest_path = parent / '.resume/latest.json'
    latest = json.loads(latest_path.read_text())
    assert latest['phase'] == 'complete' and latest['epoch'] == run['epoch']
    keep = run['retained_model']
    kept = Path(keep['path'])
    assert kept.parent == parent and kept.suffix == '.pth'
    assert kept.stat().st_size == keep['bytes'] and digest(kept) == keep['sha256']
    for key, target in zip(('current', 'previous'), run['delete'], strict=True):
        path = Path(target['path'])
        assert path == parent / latest[key]['path']
        assert path.resolve(strict=True) == path and path.parent == parent / '.resume'
        assert path.suffix == '.pt' and path.name.startswith('generation-')
        assert latest[key]['bytes'] == target['bytes'] == path.stat().st_size
        assert latest[key]['sha256'] == target['sha256'] == digest(path)
        checked.append(target)
assert len(checked) == 24
receipt = {'started_at': datetime.now().astimezone().isoformat(),
           'status': 'ALL24_TARGETS_AND12_RETAINED_MODELS_VERIFIED_BEFORE_DELETE',
           'authorization': plan['authorization'], 'plan_sha256': digest(plan_path),
           'df_before': before, 'planned_files': checked, 'deleted_files': [],
           'retained_models': [r['retained_model'] for r in runs],
           'scope': plan['scope'], 'limits': plan['limits'],
           'model_tensor_image_calls': 0, 'training_updates': 0}
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
for target in checked:
    path = Path(target['path'])
    assert path.stat().st_size == target['bytes']
    path.unlink()
    assert not path.exists()
    receipt['deleted_files'].append(target)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
for kept in receipt['retained_models']:
    path = Path(kept['path'])
    assert path.stat().st_size == kept['bytes'] and digest(path) == kept['sha256']
assert all(not Path(row['path']).exists() for row in checked)
receipt.update({'status': 'COMPLETE_UNUSED24_RESUME_WEIGHTS_DELETED_RETAINED12_MODELS_BYTE_VERIFIED',
                'completed_at': datetime.now().astimezone().isoformat(),
                'deleted_file_count': len(checked), 'deleted_logical_bytes': sum(row['bytes'] for row in checked),
                'retained_model_checks_after_delete': 12,
                'df_after': subprocess.check_output(['df', '-B1', '/root/autodl-tmp', '/root/trifusion-storage'], text=True),
                'active_training_processes_after': subprocess.run(
                    ['ps', '-o', 'pid,ppid,stat,etime,args', '-p', '93313', '--ppid', '93313'],
                    capture_output=True, text=True).stdout,
                'elapsed_seconds': time.perf_counter() - started})
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('planned_files','deleted_files','retained_models')}))


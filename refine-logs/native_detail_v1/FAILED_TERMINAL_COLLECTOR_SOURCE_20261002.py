"""Archive the original failed N1 campaign after all six training arms end."""
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

import paramiko

TASK = Path('C:/Users/gb')
PRIVATE = TASK / '.codex_tmp'
ROOT = TASK / '.trifusion_github_publish_22c3bee'
REMOTE = PurePosixPath('/data/gaob/Re-ID/Trifusion')
DEST = ROOT / 'logs/native_detail_failed_terminal_20261002'

remote_code = r'''
from datetime import datetime
import hashlib, json, shutil
from pathlib import Path

root = Path('/data/gaob/Re-ID/Trifusion')
campaign = root / 'logs/native_detail_20261002_v1'
state = json.loads((campaign / 'campaign.json').read_text())
manifest = json.loads((campaign / 'manifest.json').read_text())
waiter_path = root / 'logs/native_detail_analysis_waiter_20261002.json'
waiter = json.loads(waiter_path.read_text())
assert state['status'] == 'FAILED' and state['completed_at']
assert len(state['jobs']) == 6
assert {(j['dataset'], j['variant']) for j in state['jobs']} == {
    (d, v) for d in ('RGBNT201', 'RGBNT100', 'MSVR310') for v in ('high', 'low')}
assert {(j['dataset'], j['variant']) for j in state['jobs'] if j['status'] == 'FAILED'} == {('MSVR310', 'low')}
assert all(j['status'] in ('COMPLETE', 'FAILED') for j in state['jobs'])
assert not Path(f"/proc/{state['controller_pid']}/cmdline").exists()
assert waiter['status'] == 'CAMPAIGN_FAILED_NO_REPORT' and waiter['invocations'] == 0
assert not (campaign / 'accepted_matrix.json').exists()
assert not (root / 'results/native_detail_complete_20261002').exists()
assert len(manifest['source_sha256']) == 238
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
           for name, digest in manifest['source_sha256'].items())

paths = {campaign / 'campaign.json', campaign / 'manifest.json', waiter_path,
         root / 'logs/native_detail_analysis_launch724_20261002.json',
         root / 'logs/native_detail_analysis_waiter_20261002.stdout.log',
         Path(manifest['preflight_path'])}
paths.update(Path(name) for name in manifest['initialization_sha256'])
rows = []
for job in state['jobs']:
    child_dir = campaign / f"{campaign.name}_clean_clip_{job['variant']}_{job['dataset']}"
    child = json.loads((child_dir / 'campaign.json').read_text())
    assert child['status'] == job['status']
    assert [s['mode'] for s in child['jobs']] == ['m0', 'train', 'evaluate']
    m0, train, evaluation = child['jobs']
    assert m0['status'] == train['status'] == 'COMPLETE'
    assert m0['exit_code'] == train['exit_code'] == 0
    run = Path(train['output_dir'])
    training = json.loads((run / 'training.json').read_text())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(training['history']) == 50
    assert [h['epoch'] for h in training['history']] == list(range(1, 51))
    assert run == Path(evaluation['output_dir'])
    formal_steps = sum(1 for _ in (run / 'training_steps.jsonl').open())
    assert formal_steps == sum(h['steps'] for h in training['history'])
    official_path = run / 'official_metrics.json'
    row = {'dataset': job['dataset'], 'variant': job['variant'],
           'status': job['status'], 'training_epochs': 50,
           'formal_steps': formal_steps, 'run_dir': str(run),
           'm0_dir': m0['output_dir'], 'evaluation_exit_code': evaluation['exit_code']}
    if job['status'] == 'COMPLETE':
        assert job['exit_code'] == evaluation['exit_code'] == 0
        assert evaluation['status'] == 'COMPLETE'
        official = json.loads(official_path.read_text())
        assert official['status'] == 'COMPLETE' and official['training_epochs'] == 50
        assert official['selected_epoch'] == training['best_epoch']
        row.update(verification=child['verification'], official_receipt=official)
    else:
        assert job['exit_code'] == evaluation['exit_code'] == 1
        assert evaluation['status'] == 'FAILED' and not official_path.exists()
        row['official_receipt'] = None
        row['boundary'] = 'Original full50 training completed; strict reload evaluation failed. Formal metrics unavailable; diagnostics cannot substitute.'
    assert not Path(f"/proc/{job['pid']}/cmdline").exists()
    assert all(not Path(f"/proc/{stage['pid']}/cmdline").exists()
               for stage in child['jobs'] if 'pid' in stage)
    paths.update(p for p in child_dir.rglob('*') if p.is_file() and p.suffix in ('.json', '.log'))
    for output in (run, Path(m0['output_dir'])):
        paths.update(p for p in output.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl'))
    rows.append(row)
assert sum(row['status'] == 'COMPLETE' for row in rows) == 5
entries = []
for path in sorted(paths):
    assert root in path.parents and path.suffix in ('.json', '.jsonl', '.log')
    data = path.read_bytes()
    entries.append({'path': path.relative_to(root).as_posix(), 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest()})
print(json.dumps({'observed_at': datetime.now().astimezone().isoformat(),
    'campaign_status': state['status'], 'controller_ended': True,
    'accepted': 5, 'failed_evaluation': 1, 'formal_epochs': 300,
    'formal_steps': sum(row['formal_steps'] for row in rows),
    'report_invocations': 0, 'accepted_matrix_absent': True,
    'source_count': 238, 'sources_unchanged': True, 'rows': rows,
    'files': entries, 'disk_free_bytes': shutil.disk_usage(root).free,
    'scope': 'Read-only original failure terminal archive. No model/scorer/report execution, retry, tolerance change, acceptance rewrite or tensor/image transfer.'}))
'''

assert not DEST.exists()
client = paramiko.SSHClient()
client.load_host_keys(str(TASK / '.ssh/known_hosts'))
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename=str(TASK / '.ssh/id_ed25519'), timeout=20)
_, out, err = client.exec_command(shlex.join([
    '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', '-c', remote_code]), timeout=60)
stdout, stderr = out.read(), err.read()
(PRIVATE / 'native_detail_failed_terminal_collection_20261002.stderr').write_bytes(stderr)
assert out.channel.recv_exit_status() == 0, stderr.decode()
intake = json.loads(stdout)
DEST.mkdir(parents=True)
sftp = client.open_sftp()
for entry in intake['files']:
    data = sftp.open(str(REMOTE / entry['path'])).read()
    assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
    path = DEST / 'raw' / entry['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
sftp.close()
client.close()
(DEST / 'INTAKE.json').write_bytes(stdout)
(PRIVATE / 'native_detail_failed_terminal_intake_20261002.json').write_bytes(stdout)
print(json.dumps({'accepted': intake['accepted'], 'failed_evaluation': intake['failed_evaluation'],
                  'formal_epochs': intake['formal_epochs'], 'files': len(intake['files']),
                  'bytes': sum(item['bytes'] for item in intake['files']), 'report_invocations': 0}))

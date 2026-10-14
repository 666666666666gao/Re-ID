import base64
import hashlib
import json
from pathlib import Path
import subprocess

w = Path(__file__).resolve().parent
assert not (w / 'SEED43_MSVR_TERMINAL.json').exists()
pins = json.loads((w / 'seed43_m0/logs/near_signal_seed43_launch970/SOURCE_SEAL.json').read_bytes())
code = 'pins=' + repr(pins) + '\n' + r'''
import base64, hashlib, json, os, subprocess, time
from datetime import datetime
from pathlib import Path
root = Path('/data/gaob/Re-ID/Trifusion')
stem = 'official_extra_seed43_MSVR310_R2_near_signal_followup_20261014_970_bestmap_e50_20260926'
directory = root / 'trained-model' / stem / 'MSVR310_R2_seed43'
campaign_path = root / 'logs' / stem / 'campaign.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root / name) == digest for name, digest in pins.items())
campaign = json.loads(campaign_path.read_text())
assert campaign['status'] == 'COMPLETE' and campaign['seed'] == 43
training = json.loads((directory / 'training.json').read_text())
receipt_path = directory / 'official_metrics.json'
receipt = json.loads(receipt_path.read_text())
epochs = [json.loads(line) for line in (directory / 'epoch_official_metrics.jsonl').read_text().splitlines()]
assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
assert receipt['status'] == 'COMPLETE' and receipt['method'] == 'R2' and receipt['seed'] == 43
assert training['seed'] == 43 and training['training']['epochs'] == receipt['training_epochs'] == 50
assert [row['epoch'] for row in epochs] == list(range(1, 51))
selected = max(epochs, key=lambda row: (row['metrics']['mAP'], row['epoch']))
assert selected['epoch'] == training['selected_epoch'] == receipt['selected_epoch']
assert all(abs(selected['metrics'][name] - value) < 1e-5 for name, value in receipt['outputs']['fused']['metrics'].items())
assert training['training']['frozen_state_unchanged']
assert training['training']['missing_nonzero_gradients'] == [] and training['training']['overflow_events'] == 0
artifacts = {}
for name, path, expected in (
    ('role_best', Path(training['checkpoint']), receipt['role_checkpoint_sha256']),
    ('author_signal', Path(training['initializer']['author_checkpoint']), receipt['author_checkpoint_sha256']),
    ('official_distance', Path(receipt['distance_arrays']), receipt['distance_arrays_sha256']),
    ('protocol', Path(training['protocol']), receipt['protocol_sha256'])):
    assert path.is_file(), str(path)
    actual = sha(path)
    assert actual == expected, (name, actual, expected)
    artifacts[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=actual)
exit_path = root / 'logs/near_signal_seed43_launch970/MSVR310_EXIT.json'
exit_record = json.loads(exit_path.read_text())
assert exit_record['exit_code'] == 0
out = root / 'results/seed43_msvr_terminal975'
assert not out.exists()
out.mkdir()
diagnosis_path = out / 'MSVR310_diagnosis.json'
cmd = ['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B', str(root / 'tools/diagnose_official_retrieval.py'), '--receipt', str(receipt_path), '--output', str(diagnosis_path)]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
start = time.perf_counter()
with (out / 'stdout').open('wb') as stdout, (out / 'stderr').open('wb') as stderr:
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr)
    status = proc.wait()
assert status == 0, (out / 'stderr').read_text()
diagnosis = json.loads(diagnosis_path.read_text())
assert diagnosis['status'] == 'COMPLETE_OFFICIAL_POSTHOC_DIAGNOSIS'
assert all(sha(root / name) == digest for name, digest in pins.items())
signal = receipt['outputs']['baseline_only']['metrics']
fused = receipt['outputs']['fused']['metrics']
record = dict(status='VERIFIED_SEED43_MSVR_FULL50_TERMINAL', observed_at=datetime.now().astimezone().isoformat(),
    dataset='MSVR310', seed=43, selected_epoch=selected['epoch'], signal_metrics=signal, fused_metrics=fused,
    delta_pp={name: fused[name] - value for name, value in signal.items()},
    all_four_strictly_above_signal=all(fused[name] > value for name, value in signal.items()),
    campaign_path=str(campaign_path), campaign_sha256=sha(campaign_path), training_sha256=sha(directory / 'training.json'),
    epoch_history_sha256=sha(directory / 'epoch_official_metrics.jsonl'), receipt_sha256=sha(receipt_path), artifacts=artifacts,
    source_count=training['source_count'], source_seal_verified=len(pins), training_updates=training['training']['optimizer_steps'],
    exit_record=exit_record, diagnostic_execution=dict(command=cmd, pid=proc.pid, exit_code=status, elapsed_seconds=time.perf_counter()-start),
    nn_calls=0, optimizer_updates_in_collection=0, gpu_queries=0, other_running_dataset_queries=0,
    boundary='Original ratio1 full50 seed43 one complete mAP-best checkpoint. Cold artifact/history checks and one existing saved-distance CPU diagnosis. No seed mean, metric-wise epoch mixing, new inference or training restart.')
(out / 'VERIFIED.json').write_text(json.dumps(record, indent=2) + '\n')
paths = [campaign_path, exit_path, directory / 'training.json', directory / 'epoch_official_metrics.jsonl', receipt_path, diagnosis_path, out / 'VERIFIED.json']
print(json.dumps(dict(record=record, files={str(p.relative_to(root)): dict(sha256=sha(p), data=base64.b64encode(p.read_bytes()).decode()) for p in paths})))
'''
run = subprocess.run(['ssh', '-i', 'C:/Users/gb/.ssh/id_ed25519', '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none', '-o', 'BatchMode=yes', '-p', '2026', 'gaob@172.19.12.138', '/usr/bin/python3', '-'], input=code.encode(), capture_output=True)
(w / 'SEED43_MSVR_TERMINAL.stdout').write_bytes(run.stdout)
(w / 'SEED43_MSVR_TERMINAL.stderr').write_bytes(run.stderr)
assert run.returncode == 0, run.stderr.decode()
value = json.loads(run.stdout)
dest = w / 'seed43_msvr_terminal'
dest.mkdir()
for name, item in value.pop('files').items():
    data = base64.b64decode(item['data'])
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    path = dest / name
    assert path.resolve().is_relative_to(dest.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
(w / 'SEED43_MSVR_TERMINAL.json').write_text(json.dumps(value['record'], indent=2) + '\n', encoding='utf-8')
print(json.dumps(value['record']))

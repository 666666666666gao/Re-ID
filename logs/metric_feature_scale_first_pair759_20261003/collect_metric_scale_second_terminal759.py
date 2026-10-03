"""Read one completed full50 child and its exact terminal text, without replay."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import shlex
import paramiko

private = Path('C:/Users/gb/.codex_tmp')
campaign_private = private/'training_feature_scale_split755'
observer = json.loads((campaign_private/'second_full_observer758/STATUS.json').read_bytes())
assert observer['status'] == 'SECOND_FORMAL_COMPLETE_OBSERVED'
snapshot_path = Path(observer['last_receipt'])
snapshot = json.loads(snapshot_path.read_bytes())
assert snapshot['controller_pid'] == 2691826 and snapshot['controller_live']
parent = snapshot['files']['campaign.json']
assert parent['phase'] == 'full' and parent['report_invocations'] == 0
accepted = [row for row in parent['jobs'] if row['phase'] == 'full' and row['status'] == 'COMPLETE' and row['exit_code'] == 0]
assert len(accepted) == 2
dataset, variant = 'RGBNT201', 'normalized'
child = next(item for item in snapshot['children']
             if item['phase'] == 'full' and item['dataset'] == dataset and item['variant'] == variant)
terminal = child['files']['campaign.json']
assert terminal['status'] == 'COMPLETE'
assert len(terminal['jobs']) == 2 and all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in terminal['jobs'])
verification = terminal['verification']
assert verification['status'] == 'VERIFIED_COMPLETE'
run_dir = verification['run_dir']
output = campaign_private/'second_full_intake759'
assert not output.exists()
output.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob',
               key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
sftp = client.open_sftp()
hashes = {}
for root, names in ((run_dir, ('training.json', 'training_steps.jsonl', 'training_batch_order.jsonl', 'official_metrics.json')),
                    (child['path'], ('campaign.json', 'train.log', 'evaluate.log'))):
    for name in names:
        folder = 'trained_model' if root == run_dir else 'child_campaign'
        destination = output / folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with sftp.open(root + '/' + name, 'rb') as stream:
            data = stream.read()
        destination.write_bytes(data)
        hashes[folder + '/' + name] = hashlib.sha256(data).hexdigest()
sftp.close()
code = f"""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path({run_dir!r})
names=('best_map.pth','official_distances.pt','best_epoch_distances.pt','official_metrics.json')
rows={{}}
for name in names:
 path=root/name
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 rows[name]={{'bytes':path.stat().st_size,'sha256':digest.hexdigest()}}
print(json.dumps({{'observed_at':datetime.now().astimezone().isoformat(),'artifacts':rows,
 'disk_free_bytes':shutil.disk_usage('/data/gaob/Re-ID/Trifusion').free}}))
"""
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
client.close()
(output / 'remote_hash_stdout.txt').write_bytes(data)
(output / 'remote_hash_stderr.txt').write_bytes(error)
assert exit_code == 0, error.decode()
actual_artifacts = json.loads(data)
training = json.loads((output / 'trained_model/training.json').read_bytes())
metrics = json.loads((output / 'trained_model/official_metrics.json').read_bytes())
child_copy = json.loads((output / 'child_campaign/campaign.json').read_bytes())
assert child_copy == terminal
assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
assert [row['epoch'] for row in training['history']] == list(range(1, 51))
assert training['initializer'] == verification['initializer']
assert training['seed'] == metrics['seed'] == 42 and metrics['training_epochs'] == 50
assert metrics['status'] == 'COMPLETE' and metrics['independent_upstream_metrics_equal']
assert not metrics['reranking']
assert metrics['metrics'] == verification['metrics']
selected = max(training['history'], key=lambda row: (row['official_fused']['mAP'], row['epoch']))
assert training['best_epoch'] == metrics['selected_epoch'] == verification['best_epoch'] == selected['epoch']
assert all(abs(metrics['metrics'][name] - selected['official_fused'][name]) < 1e-5 for name in metrics['metrics'])
for name, field in (('best_map.pth', 'checkpoint_sha256'), ('official_distances.pt', 'distance_sha256'),
                    ('best_epoch_distances.pt', 'training_best_distance_sha256')):
    assert actual_artifacts['artifacts'][name]['sha256'] == metrics[field]
assert actual_artifacts['artifacts']['official_metrics.json']['sha256'] == verification['receipt_sha256']
steps = [json.loads(line) for line in (output / 'trained_model/training_steps.jsonl').read_bytes().splitlines()]
batches = [json.loads(line) for line in (output / 'trained_model/training_batch_order.jsonl').read_bytes().splitlines()]
assert len(steps) == len(batches) == sum(row['steps'] for row in training['history'])
assert [(row['epoch'], row['batch']) for row in steps] == [(row['epoch'], row['batch']) for row in batches]
assert all(math.isfinite(row[field]) for row in steps for field in ('loss', 'id', 'triplet', 'training_feature_norm_mean', 'training_feature_norm_min', 'training_feature_norm_max', 'ce_feature_norm_mean', 'ce_feature_norm_min', 'ce_feature_norm_max'))
summary = {'status': 'EXECUTOR_SECOND_FULL_TERMINAL_TEXT_CHECKED',
           'checked_at': datetime.now().astimezone().isoformat(), 'port': 2026,
           'source_snapshot_observed_at': snapshot['observed_at'],
           'source_snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
           'controller_pid': snapshot['controller_pid'], 'parent_report_invocations': 0,
           'dataset': dataset, 'variant': variant, 'training_epochs': 50,
           'formal_steps': len(steps), 'selected_epoch': metrics['selected_epoch'], 'metrics': metrics['metrics'],
           'raw_text_sha256': hashes, 'remote_artifacts': actual_artifacts,
           'training_started_at': training['started_at'], 'training_completed_at': training['completed_at'],
           'strict_evaluation_completed_at': metrics['completed_at'],
           'boundary': 'One completed child only, original strict full-gallery scoring; CPU exact text/hash checks without replay. Original parent accepted two completed controls; four formal children remain. Paired other condition/final report unavailable; no Triplet-scale efficacy, module or SOTA claim. Not independent integrity audit.'}
(output / 'INTAKE.json').write_bytes((json.dumps(summary, indent=2) + '\n').encode())
print(json.dumps({k: v for k, v in summary.items() if k != 'raw_text_sha256'}, indent=2))

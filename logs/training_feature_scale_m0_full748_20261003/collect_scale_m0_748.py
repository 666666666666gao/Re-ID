from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import paramiko

snapshot_path = Path('C:/Users/gb/.codex_tmp/training_feature_scale_observe746/m0_gate747/STATUS.json')
snapshot = json.loads(snapshot_path.read_bytes())
campaign = snapshot['files']['campaign.json']
jobs = [row for row in campaign['jobs'] if row['phase'] == 'm0']
assert len(jobs) == 6 and all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in jobs)
output = Path('C:/Users/gb/.codex_tmp/training_feature_scale_m0_748')
assert not output.exists()
output.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
sftp = client.open_sftp()
rows = []
orders = {}
hashes = {}
for job in jobs:
    key = job['dataset'] + '_' + job['variant']
    remote = '/data/gaob/Re-ID/Trifusion/trained-model/training_feature_scale_20261003_v2_m0_' + job['variant'] + '_' + job['dataset']
    target = output / key
    target.mkdir()
    for name in ('training.json', 'training_steps.jsonl', 'training_batch_order.jsonl'):
        with sftp.open(remote + '/' + name, 'rb') as stream:
            data = stream.read()
        (target / name).write_bytes(data)
        hashes[key + '/' + name] = hashlib.sha256(data).hexdigest()
    training = json.loads((target / 'training.json').read_bytes())
    assert training == job['result']
    assert training['status'] == 'M0_PASS' and training['seed'] == 42
    binding = snapshot['initialization_files'][key + '.json']['binding']
    assert training['initializer'] == binding
    assert len(training['history']) == 1 and training['history'][0]['steps'] == 8
    steps = [json.loads(line) for line in (target / 'training_steps.jsonl').read_bytes().splitlines()]
    order = [json.loads(line) for line in (target / 'training_batch_order.jsonl').read_bytes().splitlines()]
    assert len(steps) == len(order) == 8
    assert [row['batch'] for row in steps] == [row['batch'] for row in order] == list(range(8))
    assert all(row['epoch'] == 1 for row in steps + order)
    fields = ('loss', 'id', 'triplet', 'training_feature_norm_mean', 'training_feature_norm_min', 'training_feature_norm_max')
    assert all(math.isfinite(row[field]) for row in steps for field in fields)
    assert all(training[field] for field in ('frozen_parameters_unchanged', 'visual_parameters_changed', 'fresh_camera_parameters_changed'))
    assert training['m0']['nonzero_gradient_parameters'] == training['m0']['trainable_parameters'] == 155
    assert training['m0']['reload_max_abs_difference'] == 0.0
    orders[key] = order
    rows.append({'dataset': job['dataset'], 'variant': job['variant'], 'batches': 8,
                 'initial_model_state_sha256': binding['initial_model_state_sha256'],
                 'm0': training['m0'], 'trainable_parameter_tensors': binding['trainable_parameter_tensors'],
                 'peak_allocated_bytes': training['peak_allocated_bytes'],
                 'feature_norm_mean_range': [min(row['training_feature_norm_mean'] for row in steps), max(row['training_feature_norm_mean'] for row in steps)],
                 'positive_triplet_batches': sum(row['triplet'] > 0 for row in steps)})
sftp.close()
client.close()
pairs = []
for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
    normalized = snapshot['initialization_files'][dataset + '_normalized.json']['binding']
    raw = snapshot['initialization_files'][dataset + '_raw.json']['binding']
    allowed = {'recipe', 'training_feature_scaling'}
    assert {k: v for k, v in normalized.items() if k not in allowed} == {k: v for k, v in raw.items() if k not in allowed}
    assert orders[dataset + '_normalized'] == orders[dataset + '_raw']
    pairs.append({'dataset': dataset, 'initial_state_sha256': normalized['initial_model_state_sha256'],
                  'm0_batch_order_equal': True, 'm0_batches_each': 8})
summary = {'status': 'EXECUTOR_M0_RECORDS_CHECKED', 'checked_at': datetime.now().astimezone().isoformat(),
           'source_snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
           'source_snapshot_observed_at': snapshot['observed_at'], 'port': 2026,
           'controller_pid': snapshot['controller_pid'], 'total_m0_batches': 48,
           'm0_rows': rows, 'pairs': pairs, 'raw_text_sha256': hashes,
           'report_invocations': campaign['report_invocations'],
           'boundary': 'CPU text intake/aggregation only, no model, scorer, optimizer or replay. M0 engineering evidence and eight-batch order; full50 exposure/accuracy acceptance still pending. Not an independent audit.'}
(output / 'M0_GATE_SUMMARY.json').write_bytes((json.dumps(summary, indent=2) + '\n').encode())
print(json.dumps({k: v for k, v in summary.items() if k != 'raw_text_sha256'}, indent=2))

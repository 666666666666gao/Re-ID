"""One live registered-campaign observation near the first predicted endpoint."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/data/gaob/Re-ID/Trifusion')
launch = json.loads((ROOT / 'logs/role_global_tokens_launch_20261001_v1.json').read_text())
remaining = (datetime.fromisoformat('2026-10-01T09:53:00+08:00') - datetime.now().astimezone()).total_seconds()
if remaining > 0:
    time.sleep(remaining)
campaign = Path(launch['campaign'])
manifest = json.loads((campaign / 'manifest.json').read_text())
assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest['source_sha256'].items())
state = json.loads((campaign / 'campaign.json').read_text())

def process(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return {'pid': pid, 'exists': False, 'live': False}
    status = (path / 'stat').read_text().split(') ', 1)[1].split()[0]
    return {'pid': pid, 'exists': True, 'state': status, 'live': status != 'Z',
            'command': (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}

rows = []
for item in state['jobs']:
    folder = campaign / (campaign.name + '_role_global_tokens_' + item['variant'] + '_' + item['dataset'])
    row = {k: item[k] for k in ('dataset', 'variant', 'status')}
    if 'pid' in item:
        row['worker'] = process(item['pid'])
        row['gpu'] = item['gpu']
    if (folder / 'campaign.json').exists():
        child = json.loads((folder / 'campaign.json').read_text())
        row['child_status'] = child['status']
        row['stages'] = []
        for stage in child['jobs']:
            observed = {k: stage[k] for k in ('mode', 'status', 'started_at', 'pid', 'output_dir')}
            observed.update({k: stage[k] for k in ('exit_code', 'completed_at') if k in stage})
            observed['process'] = process(stage['pid'])
            output = Path(stage['output_dir'])
            receipt = output / ('official_metrics.json' if stage['mode'] == 'evaluate' else 'training.json')
            if receipt.exists():
                value = json.loads(receipt.read_text())
                observed['receipt_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
                observed['receipt_status'] = value['status']
                if stage['mode'] in ('m0', 'train'):
                    observed['history'] = value['history']
                    binding = value['initializer']
                    observed['initializer'] = {k: binding[k] for k in ('initial_model_state_sha256', 'trainable_parameters', 'architecture', 'token_mode', 'transformer_sequence_length', 'attention_subgraph_dtype')}
                if stage['mode'] == 'm0' and value['status'] == 'M0_PASS':
                    observed['m0'] = value['m0']
                    assert len(value['history']) == 1 and value['history'][0]['steps'] == 8
                    assert value['m0']['frozen_signal_unchanged']
                    assert value['m0']['nonzero_gradient_parameters'] == value['m0']['trainable_parameters']
                    assert value['m0']['reload_max_abs_difference'] <= 1e-5
                    probe = output / 'm0_reload_probe.pth'
                    assert hashlib.sha256(probe.read_bytes()).hexdigest() == value['m0']['reload_probe_sha256']
            if stage['status'] == 'FAILED':
                observed['failure_log_tail'] = (folder / (stage['mode'] + '.log')).read_text().splitlines()[-30:]
            row['stages'].append(observed)
    rows.append(row)
for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
    initializers = [stage['initializer'] for row in rows if row['dataset'] == dataset for stage in row.get('stages', []) if stage['mode'] == 'm0' and 'initializer' in stage]
    assert len({item['initial_model_state_sha256'] for item in initializers}) <= 1
    assert len({item['trainable_parameters'] for item in initializers}) <= 1
record = {'observed_at': datetime.now().astimezone().isoformat(), 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'launch': launch, 'controller': process(launch['pid']), 'campaign_status': state['status'],
          'source_count_verified': len(manifest['source_sha256']), 'rows': rows,
          'gpu_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True),
          'data_free_bytes': shutil.disk_usage(ROOT).free, 'verified_child_endpoints': sum(row.get('child_status') == 'COMPLETE' for row in rows),
          'scope': 'Live registered-campaign observation; only COMPLETE child records have finished their full endpoint verification.'}
path = campaign / 'milestone_0953_snapshot.json'
assert not path.exists()
path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'snapshot': str(path), 'observed_at': record['observed_at'], 'campaign_status': state['status'], 'controller': record['controller'], 'source_count_verified': record['source_count_verified'], 'gpu_inventory': record['gpu_inventory'], 'data_free_bytes': record['data_free_bytes'],
                  'jobs': [{'dataset': row['dataset'], 'variant': row['variant'], 'status': row['status'], 'gpu': row.get('gpu'), 'worker_live': row.get('worker', {}).get('live'), 'stages': [{'mode': s['mode'], 'status': s['status'], 'exit_code': s.get('exit_code'), 'receipt_status': s.get('receipt_status'), 'process_live': s['process']['live'], 'm0': s.get('m0'), 'completed_epochs': len(s.get('history', [])), 'initializer': s.get('initializer')} for s in row.get('stages', [])]} for row in rows]}))
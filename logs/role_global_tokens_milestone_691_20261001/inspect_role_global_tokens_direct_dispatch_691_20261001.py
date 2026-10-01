"""Record actual post-endpoint queue dispatch and the eighth real M0."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/role_global_tokens_20261001_v1'
DEST = ROOT / '.codex_tmp/role_global_tokens_direct_dispatch_691_20261001'
assert not DEST.exists()
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 215
assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == s for p, s in manifest['source_sha256'].items())
state = json.loads((CAMPAIGN / 'campaign.json').read_text())
assert state['status'] == 'RUNNING'
active = []
paths = [CAMPAIGN / 'campaign.json']
for row in state['jobs']:
    if row['status'] != 'RUNNING':
        continue
    child_path = CAMPAIGN / (CAMPAIGN.name + '_role_global_tokens_' + row['variant'] + '_' + row['dataset']) / 'campaign.json'
    child = json.loads(child_path.read_text())
    stage = child['jobs'][-1]
    assert stage['mode'] == 'train' and stage['status'] == 'RUNNING'
    proc = Path('/proc') / str(stage['pid'])
    assert proc.exists() and (proc / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    active.append({'dataset': row['dataset'], 'variant': row['variant'], 'gpu': row['gpu'], 'train_pid': stage['pid']})
    paths.append(child_path)
    if row['dataset'] == 'RGBNT201' and row['variant'] == 'direct':
        m0 = child['jobs'][0]
        assert m0['mode'] == 'm0' and m0['status'] == 'COMPLETE' and m0['exit_code'] == 0
        run = Path(m0['output_dir'])
        receipt = json.loads((run / 'training.json').read_text())
        assert receipt['status'] == 'M0_PASS'
        assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == 128
        assert receipt['m0']['frozen_signal_unchanged'] and receipt['m0']['reload_max_abs_difference'] == 0
        assert hashlib.sha256((run / 'm0_reload_probe.pth').read_bytes()).hexdigest() == receipt['m0']['reload_probe_sha256']
        assert receipt['initializer']['initial_model_state_sha256'] == '0296d6a9b8ba817d50e880c1e533bccace1e0ba4d32ff6c27a5fc6cf8730adaf'
        paths += [run / 'training.json', run / 'training_steps.jsonl', child_path.parent / 'm0.log']
        direct_m0 = receipt['m0']
assert len(active) == 4 and len({item['gpu'] for item in active}) == 4
assert any(item['dataset'] == 'RGBNT201' and item['variant'] == 'direct' for item in active)
DEST.mkdir()
entries = []
for path in paths:
    relative = path.relative_to(ROOT)
    target = DEST / 'files' / relative
    data = path.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    entries.append({'source': str(relative), 'path': str(target.relative_to(DEST)), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
record = {'observed_at': datetime.now().astimezone().isoformat(), 'active_train_processes': active,
          'direct_RGBNT201_m0': direct_m0, 'source_count_verified': 215,
          'files': entries, 'scope': 'Actual post-harvest process and M0 read, no model forward or restart.'}
(DEST / 'DISPATCH.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))

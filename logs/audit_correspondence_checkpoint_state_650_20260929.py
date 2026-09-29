"""Inspect inference-checkpoint retention; do not reconstruct a training teacher."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from datetime import datetime
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
root = Path('/data/gaob/Re-ID/Trifusion')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
matrix_path = root / '.git/correspondence_m3_accepted_649_20260929.json'
assert sha(matrix_path) == '2d6db04a1e36658ca54b75a5c4b4a43310e5e071b81f5b6784c3a0cfbb77bdf2'
sources = {'tools/run_correspondence_roles.py': 'e50865fb923297cd61cf38b33ec2bc95154f8c503b5dfe5de9823cad3f03d7ef',
           'tools/run_correspondence_role_prediction.py': '6aac5bde7c65f91954298a749782a86fa344c6cdd063fa830e899d1350671b9b',
           'modeling/trifusion/correspondence_role_prediction.py': '7c80d7aa86ac5bd0d07ab7f6df7b2e93f708a2407bfee03fb078ff2bb182787d'}
assert all(sha(root / name) == digest for name, digest in sources.items())
matrix = json.loads(matrix_path.read_bytes())
selected = [r for r in matrix['rows'] if r['status'] == 'VERIFIED_COMPLETE']
assert len(selected) == matrix['verified_complete'] == 7
rows = []
for row in selected:
    checkpoint = Path(row['run_dir']) / 'best_map.pth'
    assert sha(checkpoint) == row['checkpoint_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'trifusion-correspondence-role-prediction-v1'
    assert payload['dataset'] == row['dataset'] and payload['epoch'] == row['best_epoch']
    state = payload['state']
    counts = {prefix: sum(name.startswith(prefix) for name in state) for prefix in
              ('roles.', 'teacher.', 'backbone.adapters.', 'teacher_adapters.', 'predictors.', 'backbone.signal.')}
    assert counts['roles.'] > 0 and counts['teacher.'] == 0
    assert counts['teacher_adapters.'] > 0 and counts['backbone.signal.'] == 0
    rows.append({'dataset': row['dataset'], 'variant': row['variant'],
                 'best_epoch': row['best_epoch'], 'checkpoint_sha256': row['checkpoint_sha256'],
                 'stored_state_entries': len(state), 'retained_prefix_counts': counts,
                 'historical_role_ema_teacher_restorable_from_this_checkpoint': False})
assert not torch.cuda.is_initialized()
report = {'status': 'SEALED_INFERENCE_CHECKPOINT_STATE_RETENTION_AUDIT_COMPLETE',
          'at': datetime.now().astimezone().isoformat(), 'matrix_sha256': sha(matrix_path),
          'driver_sha256': sha(Path(__file__)), 'source_sha256': sources, 'rows': rows,
          'boundary': 'All seven sealed checkpoints retain inference student state and teacher adapters but omit teacher role state by the existing compact checkpoint contract. Normal retrieval does not use the teacher; accepted retrieval scores remain valid. A newly constructed/reloaded model cannot reproduce the historical M3 teacher or M3 gradient from this checkpoint alone. No teacher reset, substitution, model construction, images, inference, gradients, metrics or training executed.'}
out = root / '.git/correspondence_checkpoint_state_audit_650_20260929.json'
assert not out.exists()
out.write_bytes((json.dumps(report, indent=2) + '\n').encode())
print(json.dumps({'status': report['status'], 'at': report['at'], 'driver_sha256': report['driver_sha256'], 'rows': rows}))

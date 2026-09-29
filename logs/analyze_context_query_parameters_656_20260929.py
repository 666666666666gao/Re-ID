"""Inspect two accepted checkpoints on CPU; no model or activation forward."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch

root = Path('/data/gaob/Re-ID/Trifusion')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
matrix_path = root / '.git/context_identity_accepted_658_20260929.json'
assert sha(matrix_path) == '409985b75c0b2b8043a67ee84a48bca7cf306e5e8db0a312711a2282832bb22f'
matrix = json.loads(matrix_path.read_text())
manifest = json.loads((root / 'logs/correspondence_context_identity_20260929/manifest.json').read_text())
assert all(sha(root / name) == digest for name, digest in manifest['source_sha256'].items())
rows = {r['variant']: r for r in matrix['rows'] if r['dataset'] == 'RGBNT201' and r['status'] == 'VERIFIED_COMPLETE'}
assert set(rows) == {'static_none', 'context_none'}
assert rows['static_none']['initial_model_state_sha256'] == rows['context_none']['initial_model_state_sha256']
torch.set_num_threads(2)
cells = {}
for variant in ('static_none', 'context_none'):
    row = rows[variant]
    checkpoint = Path(row['run_dir']) / 'best_map.pth'
    assert sha(checkpoint) == row['checkpoint_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'trifusion-correspondence-context-identity-v1'
    assert payload['condition'] == row['condition'] and payload['epoch'] == row['best_epoch'] == 2
    state = payload['state']
    weights = state['roles.context_queries.weight'].double()
    anchors = state['roles.anchor_queries'].double()
    assert weights.shape == (384, 512) and bool(torch.isfinite(weights).all())
    singular = torch.linalg.svdvals(weights)
    cells[variant] = {
        'best_epoch': row['best_epoch'], 'checkpoint_sha256': row['checkpoint_sha256'],
        'query_projection_shape': list(weights.shape),
        'query_projection_frobenius_norm': float(weights.norm()),
        'query_projection_spectral_norm': float(singular[0]),
        'query_projection_max_abs': float(weights.abs().max()),
        'query_projection_nonzero_elements': int(torch.count_nonzero(weights)),
        'query_projection_column_deviation_frobenius': float((weights - weights.mean(dim=1, keepdim=True)).norm()),
        'query_projection_numeric_rank_float64': int(torch.linalg.matrix_rank(weights)),
        'role_projection_frobenius_norms': [float(block.norm()) for block in weights.split(128)],
        'role_projection_spectral_norms': [float(torch.linalg.svdvals(block)[0]) for block in weights.split(128)],
        'anchor_query_frobenius_norm': float(anchors.norm()),
        'anchor_query_mean_row_norm': float(anchors.norm(dim=1).mean()),
        'readout_gain': float(state['readout_gain']),
    }
report = {'status': 'ACCEPTED_CHECKPOINT_QUERY_PARAMETERS_INSPECTED_ON_CPU',
          'at': datetime.now().astimezone().isoformat(), 'dataset': 'RGBNT201',
          'source_sha256': sha(Path(__file__)), 'matrix_sha256': sha(matrix_path),
          'model_source_sha256': manifest['source_sha256']['modeling/trifusion/correspondence_context_identity.py'],
          'initialization': 'ContextSelectionRoles initializes this projection to zero; both conditions share the accepted full initial state.',
          'cells': cells,
          'boundary': 'Only saved same-best parameter tensors, not activations or sample-dependent candidate weights. '
                      'A nonzero projection shows an actual update from zero, not useful identity information or causal benefit. '
                      'Static unit context exposes one constant direction; equal stored parameter counts do not imply equal effective capacity. '
                      'Numeric rank is tolerance/dtype dependent and is not semantic diversity. Spectral norm bounds the output '
                      'for unit-norm context but does not measure its realized magnitude. No new forward, gradient, GPU, training, '
                      'reweighting or configuration selection.'}
target = root / '.git/context_identity_query_parameters_RGBNT201_656_20260929.json'
assert not target.exists()
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

"""CPU-only parameter bounds for the seven sealed, reloaded M3 checkpoints."""
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
assert sha(root / 'modeling/trifusion/correspondence_roles.py') == 'e2a51154b5b327341f35e453dd32a821b091a49a669b49a7c70e79e6e2f2591b'
assert sha(root / 'modeling/trifusion/correspondence_role_prediction.py') == '7c80d7aa86ac5bd0d07ab7f6df7b2e93f708a2407bfee03fb078ff2bb182787d'
matrix = json.loads(matrix_path.read_bytes())
selected = [r for r in matrix['rows'] if r['status'] == 'VERIFIED_COMPLETE']
assert matrix['verified_complete'] == len(selected) == 7
rows = []
for row in selected:
    checkpoint = Path(row['run_dir']) / 'best_map.pth'
    assert sha(checkpoint) == row['checkpoint_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'trifusion-correspondence-role-prediction-v1'
    assert payload['dataset'] == row['dataset'] and payload['epoch'] == row['best_epoch']
    state = payload['state']
    weight = state['readout.weight'].double()
    gain = float(state['readout_gain'])
    assert weight.shape == (1536, 384) and state['roles.anchor_queries'].shape == (16, 128)
    assert torch.isfinite(weight).all() and torch.isfinite(state['readout_gain']).all()
    blocks = []
    feature_bounds = []
    for index, role in enumerate(('cnn', 'transformer', 'mamba')):
        block = weight[:, index * 128:(index + 1) * 128]
        sigma = float(torch.linalg.svdvals(block)[0])
        norm_weight = state[f'roles.output_norms.{index}.weight'].double()
        norm_bias = state[f'roles.output_norms.{index}.bias'].double()
        assert norm_weight.shape == norm_bias.shape == (128,)
        bound = float(norm_weight.abs().max()) * 128 ** 0.5 + float(norm_bias.norm())
        feature_bounds.append(bound)
        blocks.append({'role': role, 'weight_frobenius_norm': float(block.norm()),
                      'weight_spectral_norm': sigma,
                      'pooled_feature_to_pre_normalized_output_operator_norm': abs(gain) * sigma,
                      'one_anchor_feature_to_pre_normalized_output_operator_norm': abs(gain) * sigma / 48,
                      'post_layernorm_feature_norm_upper_bound': bound})
    sigma_all = float(torch.linalg.svdvals(weight)[0])
    rows.append({'dataset': row['dataset'], 'variant': row['variant'],
                 'best_epoch': row['best_epoch'], 'checkpoint_sha256': row['checkpoint_sha256'],
                 'readout_gain': gain, 'weight_rms': float(weight.square().mean().sqrt()),
                 'full_readout_weight_spectral_norm': sigma_all,
                 'pooled_role_vector_to_pre_normalized_output_operator_norm': abs(gain) * sigma_all,
                 'role_correction_norm_upper_bound': abs(gain) * sigma_all * sum(b * b for b in feature_bounds) ** 0.5,
                 'per_role': blocks})
assert not torch.cuda.is_initialized()
record = {'status': 'SEALED_CHECKPOINT_READOUT_PARAMETER_BOUNDS_COMPLETE',
          'at': datetime.now().astimezone().isoformat(), 'driver_sha256': sha(Path(__file__)),
          'matrix_sha256': sha(matrix_path), 'selected_checkpoints': len(rows), 'cpu_threads': 1,
          'rows': rows,
          'formula': 'u=g+gamma*W*[mean_48(H_C),mean_48(H_A),mean_48(H_M)]; d_u/d_H[e,m,k]=gamma*W_e/48; d_u/d_g=I. Any loss using only fused readout gives the same direct readout cotangent to all 48 positions within one role. Subsequent normalization/head/loss are contained in that cotangent.',
          'bound': 'LayerNorm output h has norm <= sqrt(128)*max(abs(scale))+norm(bias). This yields a triangle/operator upper bound on the additive role correction, not its observed norm.',
          'boundary': 'CPU parameter reading only: no model construction, images, forward, gradient measurement, new retrieval scoring or training. Official-mAP-selected single-seed checkpoints remain exploratory. No gradient-share, optimizer-update-share, causal explanation or full M3 factor claim. Pre-normalization operator bounds do not compare actual global/evidence amplitudes or parameter gradients.'}
out = root / '.git/correspondence_readout_parameter_audit_649_20260929.json'
assert not out.exists()
out.write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({'status': record['status'], 'at': record['at'], 'driver_sha256': record['driver_sha256'],
                  'summary': [{key: r[key] for key in ('dataset', 'variant', 'best_epoch', 'readout_gain', 'pooled_role_vector_to_pre_normalized_output_operator_norm', 'role_correction_norm_upper_bound')} for r in rows]}))

"""Compare original completed text records; no training, inference or scoring."""
from pathlib import Path
from datetime import datetime
import hashlib
import json

base = Path('C:/Users/gb')
a = base / '.trifusion_github_publish_22c3bee/logs/foundation_complete739_20261002/raw/trained-model/foundation_recipe_20261002_v1_full_current_RGBNT201'
b = base / '.codex_tmp/training_feature_scale_first_full749/trained_model'
output = base / '.codex_tmp/training_feature_scale_control_repeat750'
assert not output.exists()
output.mkdir()
sources = {}
for label, root in (('F1_current', a), ('F2_normalized', b)):
    sources[label] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json')}
ta, tb = (json.loads((root / 'training.json').read_bytes()) for root in (a, b))
sa, sb = ([json.loads(line) for line in (root / 'training_steps.jsonl').read_bytes().splitlines()] for root in (a, b))
assert len(ta['history']) == len(tb['history']) == 50
assert len(sa) == len(sb) == 2649
step_fields = ('epoch', 'batch', 'loss', 'id', 'triplet', 'lr')
step_mismatches = [i for i, (x, y) in enumerate(zip(sa, sb))
                   if {key: x[key] for key in step_fields} != {key: y[key] for key in step_fields}]
epoch_fields = ('epoch', 'steps', 'mean_loss', 'official_fused')
epoch_mismatches = [i + 1 for i, (x, y) in enumerate(zip(ta['history'], tb['history']))
                    if {key: x[key] for key in epoch_fields} != {key: y[key] for key in epoch_fields}]
same_initial = {key: ta['initializer'][key] == tb['initializer'][key]
                for key in ('dataset', 'seed', 'public_clip_sha256', 'visual_initial_sha256',
                            'camera_initial_sha256', 'current_head_initial_sha256', 'initial_model_state_sha256',
                            'trainable_parameters', 'trainable_parameter_tensors', 'batch_size', 'num_instances')}
metrics_a, metrics_b = (json.loads((root / 'official_metrics.json').read_bytes()) for root in (a, b))
record = {'status': 'COMPLETED_CONTROL_TEXT_COMPARISON',
          'observed_at': datetime.now().astimezone().isoformat(), 'dataset': 'RGBNT201',
          'source_text_sha256': sources, 'initial_fields_equal': same_initial,
          'step_compared_fields': step_fields, 'steps_compared': len(sa),
          'step_mismatch_indices': step_mismatches,
          'epoch_compared_fields': epoch_fields, 'epochs_compared': 50,
          'epoch_mismatch_numbers': epoch_mismatches,
          'strict_metrics_equal': metrics_a['metrics'] == metrics_b['metrics'],
          'selected_epoch_equal': metrics_a['selected_epoch'] == metrics_b['selected_epoch'],
          'nonzero_triplet_steps': {'F1_current': sum(row['triplet'] > 0 for row in sa),
                                   'F2_normalized': sum(row['triplet'] > 0 for row in sb)},
          'boundary': 'CPU comparison of selected original logged fields on one completed same-seed control. Excludes wall timing and does not prove image/augmentation byte equality, cross-seed robustness, feature-scale efficacy, whole F1 causality or SOTA. Not independent integrity audit.'}
(output / 'CONTROL_REPRODUCTION.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record, indent=2))

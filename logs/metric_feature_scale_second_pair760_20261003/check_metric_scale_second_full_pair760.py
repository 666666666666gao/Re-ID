"""Check fixed initialization and full batch-order equality, without scoring."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math

private = Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
control = private/'first_full_intake758/trained_model'
hybrid = private/'fourth_full_intake760/trained_model'
a = json.loads((control/'training.json').read_bytes())
b = json.loads((hybrid/'training.json').read_bytes())
assert a['dataset'] == b['dataset'] == 'MSVR310' and a['seed'] == b['seed'] == 42
assert a['recipe'] == 'normalized' and b['recipe'] == 'metric_raw'
assert a['status'] == b['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
assert len(a['history']) == len(b['history']) == 50
init_keys = ('public_clip_sha256','protocol_sha256','author_source_commit','visual_initial_sha256',
             'camera_initial_sha256','current_head_initial_sha256','initial_model_state_sha256',
             'trainable_parameters','trainable_parameter_tensors','batch_size','num_instances',
             'base_recipe','ce_training_feature_scaling','deployment_feature_scaling')
assert all(a['initializer'][name] == b['initializer'][name] for name in init_keys)
assert a['initializer']['ce_training_feature_scaling'] == b['initializer']['ce_training_feature_scaling'] == 'normalized'
assert a['initializer']['metric_training_feature_scaling'] == 'normalized'
assert b['initializer']['metric_training_feature_scaling'] == 'raw'
order_a = (control/'training_batch_order.jsonl').read_bytes()
order_b = (hybrid/'training_batch_order.jsonl').read_bytes()
assert order_a == order_b and len(order_a.splitlines()) == 1000
norms = []
for folder in (control, hybrid):
    steps = [json.loads(line) for line in (folder/'training_steps.jsonl').read_bytes().splitlines()]
    assert len(steps) == 1000
    assert all(math.isfinite(row[key]) for row in steps for key in ('loss','id','triplet','ce_feature_norm_min','ce_feature_norm_max','training_feature_norm_min','training_feature_norm_max'))
    norms.append({'variant':'normalized' if folder == control else 'metric_raw',
                  'formal_steps':len(steps),'ce_norm_range':[min(r['ce_feature_norm_min'] for r in steps),max(r['ce_feature_norm_max'] for r in steps)],
                  'metric_norm_range':[min(r['training_feature_norm_min'] for r in steps),max(r['training_feature_norm_max'] for r in steps)]})
record = {'status':'SECOND_FRESH_FULL_PAIR_INITIALIZATION_AND_ORDER_CHECKED','recorded_at':datetime.now().astimezone().isoformat(),
          'dataset':'MSVR310','seed':42,'formal_epochs':100,'formal_steps':2000,
          'compared_initialization_fields':init_keys,'initial_model_state_sha256':a['initializer']['initial_model_state_sha256'],
          'batch_order_sha256':hashlib.sha256(order_a).hexdigest(),'batch_order_bytes_equal':True,'norms':norms,
          'boundary':'Text-only complete-pair integrity/norm check. No score comparison, primary-gate decision, model/replay/report, augmentation-byte equivalence, training multiseed or overall/module/SOTA conclusion. The RGBNT100 pair and sole final report remain incomplete.'}
output = private/'SECOND_PAIR_ACCOUNTING760.json'
assert not output.exists()
output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))

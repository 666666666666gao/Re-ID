"""Compare the fresh normalized control with the immutable prior control texts."""
from pathlib import Path
from datetime import datetime
import hashlib
import json

base = Path('C:/Users/gb/.codex_tmp')
current = base/'training_feature_scale_split755/first_full_intake758'
intake = json.loads((current/'INTAKE.json').read_bytes())
assert intake['variant'] == 'normalized' and intake['training_epochs'] == 50
dataset = intake['dataset']
old = base/'training_feature_scale_complete754/raw/trained-model'/f'training_feature_scale_20261003_v2_full_normalized_{dataset}'
new = current/'trained_model'
old_training = json.loads((old/'training.json').read_bytes())
new_training = json.loads((new/'training.json').read_bytes())
old_steps = [json.loads(line) for line in (old/'training_steps.jsonl').read_bytes().splitlines()]
new_steps = [json.loads(line) for line in (new/'training_steps.jsonl').read_bytes().splitlines()]
assert len(old_steps) == len(new_steps)
step_keys = ('epoch','batch','loss','id','triplet','training_feature_norm_mean',
             'training_feature_norm_min','training_feature_norm_max','lr')
epoch_keys = ('epoch','steps','mean_loss','official_fused')
init_keys = ('dataset','recipe','seed','public_clip_sha256','protocol_sha256','author_source_commit',
             'visual_initial_sha256','camera_initial_sha256','current_head_initial_sha256',
             'initial_model_state_sha256','trainable_parameters','trainable_parameter_tensors',
             'batch_size','num_instances','base_recipe','deployment_feature_scaling')
step_differences = [index for index, (a,b) in enumerate(zip(old_steps,new_steps))
                    if any(a[key] != b[key] for key in step_keys)]
epoch_differences = [index+1 for index, (a,b) in enumerate(zip(old_training['history'],new_training['history']))
                     if any(a[key] != b[key] for key in epoch_keys)]
init_differences = [key for key in init_keys if old_training['initializer'][key] != new_training['initializer'][key]]
old_order = (old/'training_batch_order.jsonl').read_bytes()
new_order = (new/'training_batch_order.jsonl').read_bytes()
assert old_order == new_order
record = {'status':'COMPLETED_CONTROL_RECORDED_FIELDS_COMPARED','recorded_at':datetime.now().astimezone().isoformat(),
          'dataset':dataset,'variant':'normalized','formal_epochs':50,'formal_steps':len(new_steps),
          'compared_step_fields':step_keys,'compared_epoch_fields':epoch_keys,'compared_initialization_fields':init_keys,
          'different_step_indices':step_differences,'different_epochs':epoch_differences,
          'different_initialization_fields':init_differences,'actual_batch_order_bytes_equal':True,
          'batch_order_sha256':hashlib.sha256(new_order).hexdigest(),
          'all_compared_recorded_fields_equal':not(step_differences or epoch_differences or init_differences),
          'current_terminal_intake_sha256':hashlib.sha256((current/'INTAKE.json').read_bytes()).hexdigest(),
          'boundary':'Recorded same-seed control reproducibility only; timing and F3 additional CE norm diagnostics excluded. Not complete augmentation-byte parity, functional equality on all inputs, a second seed, Triplet-scale efficacy, module gain or independent integrity audit. No model, scoring or reporting executed.'}
path = current/'CONTROL_COMPARISON.json'
assert not path.exists()
path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))

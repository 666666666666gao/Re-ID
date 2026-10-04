from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
source=Path('/tmp/trifusion_global_task_role_cpu_20261004/run_global_task_role.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='151a3350a9da7d86ce6b4b7d5160537b182792ab08727ccb5e4ab0c64bb56d5f'
sys.path.insert(0,'/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,'/data/gaob/Re-ID/Trifusion/modeling')
import trifusion
trifusion.__path__.insert(0,'/tmp/trifusion_global_task_role_cpu_20261004/trifusion')
spec=importlib.util.spec_from_file_location('tools.run_global_task_role',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
module.configure()
inner=module.previous.base.entry.entry
assert inner.AuthorHeadEvidence is module.GlobalTaskRoleHeads
assert inner.foundation.build_core is module.build_core
assert inner.foundation.loss_values is module.loss_values
assert inner.foundation.condition is module.condition
assert inner.SCHEMA==inner.foundation.SCHEMA==module.SCHEMA
args=argparse.Namespace(variant='semantic',baseline_sha256='cpu-placeholder',initialization=source)
condition=inner.foundation.condition(args)
assert condition['objective_gradient_policy']==module.POLICY
assert condition['visual_placement']=='single_process_original_full_batch_first6_cuda1_last6_and_heads_cuda0'
assert condition['author_training_objectives']==['shared_global','role_corrected_fused']
print(json.dumps({'status':'CPU_CONFIGURATION_CHAIN_PASS','condition':condition,'schema':module.SCHEMA,'boundary':'Imports and explicit configuration wiring only; no model build, production M0, GPU or training.'}))

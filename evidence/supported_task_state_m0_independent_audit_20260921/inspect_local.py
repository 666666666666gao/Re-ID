import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE = Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_task_state_m0_complete_20260921')
OUT = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
inventory = json.loads((INTAKE/'intake_inventory.json').read_text())
input_hashes = {}
for entry in inventory['files']:
    p = INTAKE / entry['path']
    assert p.stat().st_size == entry['bytes']
    assert sha(p) == entry['sha256']
    input_hashes[str(p)] = sha(p)
snapshot_files = [
    'tools/run_msvr_supported_task_state.py', 'tools/train_msvr_supported_task_state.py',
    'tools/msvr_task_state_optimizer.py', 'tools/msvr_task_state_records.py',
    'tools/check_msvr_supported_task_state.py', 'tools/check_msvr_task_state_math.py',
    'tools/verify_msvr_supported_task_state.py', 'tools/verify_msvr_task_state_records.py',
    'tools/msvr_supported_gradient_balance.py', 'tools/verify_msvr_supported_gradient_balance_stats.py',
    'tools/train_msvr310_source_style.py', 'tools/train_msvr310_trifusion_oof.py',
    'tools/train_msvr310_signal_oof.py', 'tools/train_msvr_instance_memory.py',
    'tools/train_msvr_cross_scene_smooth_ap.py', 'tools/train_msvr_smooth_ap.py',
    'tools/train_msvr_role_set.py', 'tools/train_msvr_history_gradient.py',
    'tools/train_msvr_fresh_coordinate.py', 'tools/msvr_cross_scene_smooth_ap.py',
    'tools/msvr_smooth_ap.py', 'tools/msvr_instance_memory.py', 'tools/msvr_freshness_probe.py',
    'tools/probe_msvr_role_set_gradients.py', 'tools/probe_msvr_history_candidate_gradients.py',
    'tools/msvr_role_set_relations.py', 'tools/run_signal_preserving_v5.py',
    'tools/verify_msvr310_source_style.py', 'modeling/trifusion/signal_preserving_v8.py',
    'refine-logs/msvr310_supported_task_state_v1/EXPERIMENT_PLAN.md',
    'refine-logs/msvr310_supported_task_state_v1/EXPERIMENT_TRACKER.md',
]
configs = {}; todo = ['configs/MSVR310/TriFusion-supported-task-state-paired-v1.json']
def strings(x):
    if isinstance(x, dict):
        for value in x.values(): yield from strings(value)
    elif isinstance(x, list):
        for value in x: yield from strings(value)
    elif isinstance(x, str): yield x
while todo:
    rel = todo.pop()
    if rel in configs: continue
    p = REPO/rel
    cfg = json.loads(p.read_text())
    configs[rel] = cfg
    for value in strings(cfg):
        if value.startswith('configs/') and value.endswith('.json') and (REPO/value).is_file(): todo.append(value)
snapshot_files += list(configs)
bindings = []
for name,cfg in configs.items():
    for key in ('project_file_sha256','project_source_file_sha256'):
        for rel, digest in cfg.get(key,{}).items():
            path = REPO/rel
            actual = sha(path)
            lf = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            bindings.append(dict(config=name,path=rel,expected=digest,actual=actual,passed=actual==digest,lf_normalized=lf,lf_matches=lf==digest))
for rel in sorted(set(snapshot_files)):
    p=REPO/rel; dest=OUT/'snapshots'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest); input_hashes[str(p)]=sha(p)
    if p.suffix == '.py': ast.parse(p.read_text(encoding='utf-8-sig'))
summary=json.loads((INTAKE/'m0/summary.json').read_text())
cpu=json.loads((INTAKE/'m0_cpu.json').read_text())
t0=json.loads((INTAKE/'t0.json').read_text())
overview={
    'intake_files_hash_verified':len(inventory['files']),
    'config_bindings_verified':len(bindings),
    'configs':{k:dict(keys=list(v), values={key:value for key,value in v.items() if key not in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256')}) for k,v in configs.items()},
    'summary_keys':list(summary),
    'summary_header':{k:v for k,v in summary.items() if k not in ('folds','overfit')},
    't0_header':{k:v for k,v in t0.items() if k not in ('folds',)},
    'cpu_training_checks':cpu['training_checks'],
    'endpoint_examples':{},
}
for fold in summary['folds']:
    for arm,row in fold['endpoints'].items():
        overview['endpoint_examples'][f"fold_{fold['fold']}_{arm}"]={k:v for k,v in row.items() if k!='training'}
overview['overfit']={end:{k:v for k,v in row.items() if k!='training'} for end,row in summary['overfit'].items()}
write('local_input_hashes.json',input_hashes)
write('config_binding_checks.json',bindings)
write('local_overview.json',overview)
print(json.dumps({k:v for k,v in overview.items() if k not in ('endpoint_examples','t0_header')},indent=2))
print(json.dumps({'t0_header':overview['t0_header'],'example_endpoint':next(iter(overview['endpoint_examples'].values()))},indent=2))

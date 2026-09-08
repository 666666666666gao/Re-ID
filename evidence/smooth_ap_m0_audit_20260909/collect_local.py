"""Read-only local inputs; all emitted files remain in this audit directory."""
import ast, hashlib, json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE = Path('C:/Users/gb/.codex_tmp/smooth_ap_m0_complete_20260908')
hashes = {}
bindings = []
configs = {}
seen = set()

def capture(path, label):
    data = path.read_bytes()
    target = OUT/'snapshots'/label
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    hashes[str(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'snapshot': str(target.relative_to(OUT))}
    return data

def visit_config(rel):
    if rel in configs:
        return
    path = REPO/rel
    if path.suffix != '.json' or not path.is_file():
        return
    obj = json.loads(capture(path, Path('repo')/rel))
    configs[rel] = obj
    for key in ('project_file_sha256', 'fixed_file_sha256', 'project_files', 'fixed_files', 'project_source_file_sha256'):
        for name, digest in obj.get(key, {}).items():
            bindings.append({'config': rel, 'key': key, 'path': name, 'expected_sha256': digest})
            if not name.startswith('/') and (REPO/name).is_file():
                capture(REPO/name, Path('repo')/name)
                if name.startswith('configs/'):
                    visit_config(name)
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():
                walk(k);walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
        elif isinstance(value,str) and value.startswith('configs/'):
            visit_config(value)
    walk(obj)

def code_tree(rel):
    if rel in seen or not (REPO/rel).is_file():
        return
    seen.add(rel)
    data = capture(REPO/rel, Path('repo')/rel)
    if not rel.endswith('.py'):
        return
    tree = ast.parse(data)
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, ast.Import):
            candidates = [x.name for x in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            candidates = [node.module] + [node.module+'.'+x.name for x in node.names]
        for mod in candidates:
            for prefix in ('', 'modeling/'):
                for suffix in ('.py', '/__init__.py'):
                    name = prefix+mod.replace('.', '/')+suffix
                    if (REPO/name).is_file():
                        code_tree(name)

visit_config('configs/MSVR310/TriFusion-smooth-ap-paired-v1.json')
for rel in ['tools/train_msvr_smooth_ap.py','tools/verify_msvr_smooth_ap.py','tools/check_msvr_smooth_ap.py','tools/run_msvr_smooth_ap.py','tools/msvr_smooth_ap.py','tools/check_msvr_smooth_ap_math.py','tools/msvr_freshness_probe.py','tools/probe_msvr_history_candidate_gradients.py','tools/msvr_instance_memory.py','tools/train_msvr_history_gradient.py']:
    code_tree(rel)
for rel in ['refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md','refine-logs/msvr310_smooth_ap_v1/EXPERIMENT_TRACKER.md','results/MSVR310_SMOOTH_AP_V1_M0_2026-09-08.md']:
    capture(REPO/rel, Path('repo')/rel)
for path in INTAKE.rglob('*'):
    if path.is_file():
        capture(path, Path('intake')/path.relative_to(INTAKE))
capture(Path('C:/Users/gb/.codex_tmp/smooth_ap_m0_independent_audit_request_20260908.md'),Path('reviewer_request.md'))
(OUT/'local_input_manifest.json').write_text(json.dumps(hashes, indent=2),encoding='utf-8')
(OUT/'recursive_bindings.json').write_text(json.dumps(bindings, indent=2),encoding='utf-8')
(OUT/'recursive_configs.json').write_text(json.dumps(configs, indent=2),encoding='utf-8')
(OUT/'code_dependencies.json').write_text(json.dumps(sorted(seen), indent=2),encoding='utf-8')
print(json.dumps({'local_inputs':len(hashes),'configs':list(configs),'bindings':len(bindings),'code_files':len(seen)},indent=2))

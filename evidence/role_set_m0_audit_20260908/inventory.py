"""Independent read-only local inventory; writes only to this audit directory."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
EVIDENCE = ROOT / 'evidence/msvr310_role_set_m0_complete_20260908'
INTAKE = Path('C:/Users/gb/.codex_tmp/role_set_m0_complete_20260908')
hashes = {}

def capture(path, prefix='project'):
    data = path.read_bytes()
    relative = path.relative_to(ROOT) if prefix == 'project' else path.relative_to(INTAKE)
    target = OUT / 'snapshots' / prefix / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    hashes[str(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'snapshot': str(target.relative_to(OUT))}
    return data

configs, bindings, missing = {}, [], []
pending = ['configs/MSVR310/TriFusion-role-set-paired-v1.json']
while pending:
    name = pending.pop(0)
    if name in configs:
        continue
    path = ROOT / name
    value = json.loads(capture(path))
    configs[name] = value
    for kind in ('project_file_sha256', 'fixed_file_sha256', 'project_source_file_sha256'):
        for bound, digest in value.get(kind, {}).items():
            item = {'config': name, 'kind': kind, 'path': bound, 'expected_sha256': digest}
            if not bound.startswith('/'):
                candidate = ROOT / bound
                if candidate.is_file():
                    capture(candidate)
                    item['actual_sha256'] = hashes[str(candidate)]['sha256']
                    item['pass'] = item['actual_sha256'] == digest
                    if bound.startswith('configs/') and bound.endswith('.json'):
                        pending.append(bound)
                else:
                    missing.append(bound)
            bindings.append(item)
    def config_strings(item):
        if isinstance(item, dict):
            for nested in item.values():
                config_strings(nested)
        elif isinstance(item, list):
            for nested in item:
                config_strings(nested)
        elif isinstance(item, str) and item.startswith('configs/') and item.endswith('.json'):
            pending.append(item)
    config_strings(value)

requested = ['train_msvr_role_set', 'verify_msvr_role_set', 'check_msvr_role_set', 'run_msvr_role_set', 'msvr_role_set_relations', 'check_msvr_role_set_relations', 'probe_msvr_role_set_gradients', 'msvr_freshness_probe', 'probe_msvr_history_candidate_gradients', 'msvr_instance_memory', 'train_msvr_history_gradient']
pending = [ROOT / 'tools' / (name + '.py') for name in requested]
code = {}
while pending:
    path = pending.pop(0)
    name = str(path.relative_to(ROOT)).replace('\\', '/')
    if name in code:
        continue
    source = capture(path).decode('utf-8-sig')
    tree = ast.parse(source, filename=name)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                parent = list(path.relative_to(ROOT).with_suffix('').parts[:-node.level])
                module = '.'.join(parent + ([module] if module else []))
            imports.append(module)
            imports.extend(module + '.' + a.name for a in node.names)
    local = []
    for module in imports:
        candidate = (ROOT / 'modeling' if module.startswith('trifusion.') else ROOT).joinpath(*module.split('.')).with_suffix('.py')
        if candidate.is_file():
            pending.append(candidate)
            local.append(str(candidate.relative_to(ROOT)).replace('\\', '/'))
    code[name] = {'lines': len(source.splitlines()), 'local_imports': sorted(set(local))}

for name in ('AGENTS.md','refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md','refine-logs/msvr310_role_set_v1/EXPERIMENT_TRACKER.md','results/MSVR310_ROLE_SET_V1_M0_2026-09-08.md'):
    capture(ROOT / name)
for folder, prefix in ((EVIDENCE, 'project'), (INTAKE, 'intake')):
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            capture(path, prefix)

manifest = json.loads((EVIDENCE/'intake_manifest.json').read_bytes())
intake_checks = []
for row in manifest['files']:
    check = dict(row)
    check['local_sha_match'] = hashes[str(INTAKE/row['path'])]['sha256'] == row['sha256']
    check['repo_sha_match'] = hashes[str(EVIDENCE/row['path'])]['sha256'] == row['sha256']
    check['both_bytes_match'] = all(hashes[str(p/row['path'])]['bytes'] == row['bytes'] for p in (INTAKE,EVIDENCE))
    intake_checks.append(check)

result = {'generated_at': datetime.now(timezone.utc).isoformat(), 'configs': configs, 'bindings': bindings, 'missing_local': missing, 'code_dependencies': code, 'intake_checks': intake_checks, 'hashes': hashes}
(OUT/'local_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'configs': list(configs), 'project_bindings': sum(x['kind']=='project_file_sha256' for x in bindings), 'fixed_bindings': sum(x['kind']=='fixed_file_sha256' for x in bindings), 'binding_failures': [x for x in bindings if x.get('pass') is False], 'code_files': len(code), 'captured_files': len(hashes), 'missing': missing, 'intake_files': len(intake_checks), 'intake_bytes': sum(x['bytes'] for x in intake_checks), 'intake_all_match': all(x['local_sha_match'] and x['repo_sha_match'] and x['both_bytes_match'] for x in intake_checks)}, indent=2))

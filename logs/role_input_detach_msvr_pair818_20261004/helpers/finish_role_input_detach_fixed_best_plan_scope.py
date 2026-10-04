from datetime import datetime
from pathlib import Path
import hashlib
import json

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
family = repo / 'refine-logs/role_input_detach_fixed_best_diagnosis_v1'
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
failure = json.loads((proof / 'role_input_detach_fixed_best_plan_local_preparation_failure.json').read_bytes())
assert failure['status'] == 'LOCAL_FULL_SOURCE_ASSERTION_FAILED_NO_MODEL_EXECUTION'
assert failure['missing_count'] == 107 and failure['line_endings_only_count'] == 27
output = proof / 'role_input_detach_fixed_best_plan_owned.json'
assert not output.exists()
now = datetime.now().astimezone()
versioned = list(family.glob('EXPERIMENT_PLAN_*.md'))
assert len(versioned) == 1
assert versioned[0].read_bytes() == (family / 'EXPERIMENT_PLAN.md').read_bytes()
entry = 'tools/diagnose_role_input_detach_best.py'
plan = 'refine-logs/role_input_detach_fixed_best_diagnosis_v1/EXPERIMENT_PLAN.md'
review_path = family / 'SOURCE_REVIEW.json'
review = json.loads(review_path.read_bytes())
assert review['status'] == 'FOCUSED_PRIMARY_SOURCE_REVIEW_COMPLETE_NO_EXECUTION'
assert all(hashlib.sha256((repo / n).read_bytes()).hexdigest() == h for n, h in review['source_sha256'].items())
old_scope = json.loads((repo / 'refine-logs/role_input_detach_v1/SOURCE_SCOPE.json').read_bytes())
sources = dict(old_scope['source_sha256'])
assert len(sources) == 322
observed = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_input_detach817_msvr_native_observations/115756_664086')
assert json.loads((observed / 'EXIT.json').read_bytes())['exit_code'] == 0
observation = json.loads((observed / 'stdout.json').read_bytes())
assert observation['source_sha_unchanged']
assert observation['campaign']['status'] == 'RUNNING'
for name in (entry, plan):
    assert name not in sources
    sources[name] = hashlib.sha256((repo / name).read_bytes()).hexdigest()
scope = {'schema': 'trifusion-role-input-detach-fixed-best-diagnosis-v1', 'registered_at': now.isoformat(),
    'status': 'SOURCE_SCOPE_REGISTERED_INPUT_ARTIFACT_SEAL_PENDING_FULL_SIX',
    'source_sha256': sources, 'planned_models': 6, 'physical_gpus': [0, 1], 'optimizer_updates': 0,
    'original322_verification': {'at': observation['at'], 'source': str(observed / 'stdout.json'), 'remote_sha_unchanged': True},
    'local_scope_boundary': 'Original322 inherited as exact sealed remote hashes. Sparse local tree misses107 bound sources;27 local inherited files differ only by CRLF/LF. No file normalized or replaced. Full server source verification still required before input artifact sealing/execution.',
    'boundary': 'Original322 plus new diagnosis wrapper and plan2. Artifact seal may be created only after all six formal endpoints and original once-only CPU report complete; no GPU execution yet.'}
scope_path = family / 'SOURCE_SCOPE.json'
assert not scope_path.exists()
scope_path.write_text(json.dumps(scope, indent=2) + '\n', encoding='utf-8')
failure_path = family / 'LOCAL_PREPARATION_FAILURE.json'
assert not failure_path.exists()
failure_path.write_bytes((proof / 'role_input_detach_fixed_best_plan_local_preparation_failure.json').read_bytes())
owned = [entry, plan, versioned[0].relative_to(repo).as_posix(), review_path.relative_to(repo).as_posix(),
    scope_path.relative_to(repo).as_posix(), failure_path.relative_to(repo).as_posix()]
with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
    for name in owned:
        stream.write(f'\n| {now.isoformat()} | /experiment-plan | {name} | implementation | 读取detach六端结束后的固定best分解；PREPARED_NOT_EXECUTED，324源码范围已登记，原322以远端精确SHA为准，输入封存及GPU执行待全六端和原一次CPU报告完成 |\n')
owned.append('MANIFEST.md')
output.write_text(json.dumps({'status': 'NEW_POST_FULL_SIX_DIAGNOSTIC_PLAN_PREPARED_NOT_EXECUTED', 'at': now.isoformat(),
    'files': owned, 'source_count': len(sources), 'source_scope_sha256': hashlib.sha256(scope_path.read_bytes()).hexdigest()}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'NEW_POST_FULL_SIX_DIAGNOSTIC_PLAN_PREPARED_NOT_EXECUTED', 'files': len(owned), 'source_count': len(sources), 'at': now.isoformat()}))

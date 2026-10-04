from datetime import datetime
from pathlib import Path
import ast
import hashlib
import json

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
family = repo / 'refine-logs/role_input_detach_fixed_best_diagnosis_v1'
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
output = proof / 'role_input_detach_fixed_best_plan_owned.json'
assert not output.exists()
now = datetime.now().astimezone()
entry = 'tools/diagnose_role_input_detach_best.py'
plan = 'refine-logs/role_input_detach_fixed_best_diagnosis_v1/EXPERIMENT_PLAN.md'
source = (repo / entry).read_text(encoding='utf-8')
ast.parse(source)
assert "diagnosis.__file__ = str(Path(__file__).resolve())" in source
assert "intervention.configure()" in source
assert "raise SystemExit(diagnosis.main())" in source
native_entry = (repo / 'tools/run_native_research.py').read_text(encoding='utf-8')
intervention_entry = (repo / 'tools/run_role_input_detach.py').read_text(encoding='utf-8')
diagnostic = (repo / 'tools/diagnose_native_research_best.py').read_text(encoding='utf-8')
assert "entry.build_core = build_core" in native_entry
assert "base.build_core = build_core" in intervention_entry and "base.configure()" in intervention_entry
assert "entry.configure()" in diagnostic and "str(Path(__file__))" in diagnostic
assert "assert binding == row['initializer']" in diagnostic
assert "assert before == after" in diagnostic and "panel.base.sha(Path(name))" in diagnostic
versioned = family / ('EXPERIMENT_PLAN_' + now.strftime('%Y%m%d_%H%M%S') + '.md')
assert not versioned.exists()
versioned.write_bytes((repo / plan).read_bytes())
owned = [entry, plan, versioned.relative_to(repo).as_posix()]
review = {
    'status': 'FOCUSED_PRIMARY_SOURCE_REVIEW_COMPLETE_NO_EXECUTION', 'at': now.isoformat(),
    'reviewer': 'primary agent; not independent review, runtime verification or scientific efficacy',
    'checks': [
        'Current role-input-detach configure aliases actual detached classes and updates original native schema/builder; original diagnostic calls the same configured native builder',
        'The reused diagnostic initializer equality and checkpoint metadata/epoch checks remain active for new rows',
        'Child command uses the new wrapper file after explicit filename binding; child re-enters role-input-detach configuration',
        'Reused extraction runs one original forward_features per eval record and keeps the actual native detail hook location',
        'Same full query/gallery, author camera/scene scoring, all per-query AP and first-match ranks; no new selection or gain/loss changes',
        'Seal checks and model/buffer state invariance remain active; original retired M0 verifier is not invoked',
        'All six own accepted full50 and once-only report required before sealing/execution; current live queue unchanged'
    ],
    'source_sha256': {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in (
        entry, 'tools/diagnose_native_research_best.py', 'tools/run_role_input_detach.py', 'tools/run_native_research.py')},
    'boundary': 'Static call-chain review and AST only. No GPU import/run, completed input seal, new retrieval evidence, independent reproduction, global preservation, deterministic-kernel repair or SOTA claim.'
}
review_path = family / 'SOURCE_REVIEW.json'
assert not review_path.exists()
review_path.write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
owned.append(review_path.relative_to(repo).as_posix())
old_scope = json.loads((repo / 'refine-logs/role_input_detach_v1/SOURCE_SCOPE.json').read_bytes())
assert len(old_scope['source_sha256']) == 322
assert all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest for name, digest in old_scope['source_sha256'].items())
sources = dict(old_scope['source_sha256'])
for name in (entry, plan):
    assert name not in sources
    sources[name] = hashlib.sha256((repo / name).read_bytes()).hexdigest()
scope = {'schema': 'trifusion-role-input-detach-fixed-best-diagnosis-v1', 'registered_at': now.isoformat(),
    'status': 'SOURCE_SCOPE_REGISTERED_INPUT_ARTIFACT_SEAL_PENDING_FULL_SIX',
    'source_sha256': sources, 'planned_models': 6, 'physical_gpus': [0, 1], 'optimizer_updates': 0,
    'boundary': 'Original322 plus new diagnosis wrapper and plan2. Artifact seal may be created only after all six formal endpoints and original once-only CPU report complete; no GPU execution yet.'}
scope_path = family / 'SOURCE_SCOPE.json'
assert not scope_path.exists()
scope_path.write_text(json.dumps(scope, indent=2) + '\n', encoding='utf-8')
owned.append(scope_path.relative_to(repo).as_posix())
with (repo / 'MANIFEST.md').open('a', encoding='utf-8') as stream:
    for name in owned:
        stream.write(f'\n| {now.isoformat()} | /experiment-plan | {name} | implementation | 读取detach六端结束后的固定best分解；PREPARED_NOT_EXECUTED，324源码范围已登记，实际输入封存及GPU执行待全六端和原一次CPU报告完成 |\n')
owned.append('MANIFEST.md')
output.write_text(json.dumps({'status': 'NEW_POST_FULL_SIX_DIAGNOSTIC_PLAN_PREPARED_NOT_EXECUTED', 'at': now.isoformat(), 'files': owned,
    'source_count': len(sources), 'source_scope_sha256': hashlib.sha256(scope_path.read_bytes()).hexdigest()}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'NEW_POST_FULL_SIX_DIAGNOSTIC_PLAN_PREPARED_NOT_EXECUTED', 'files': len(owned), 'source_count': len(sources), 'at': now.isoformat()}))

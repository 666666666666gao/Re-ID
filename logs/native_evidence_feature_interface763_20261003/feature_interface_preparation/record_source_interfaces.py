"""Record actual source interfaces; do not import or execute model modules."""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json

draft = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
archive = Path('C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source')
out = draft/'feature_interface_preparation763'/'SOURCE_INTERFACE_STATUS.json'
assert not out.exists()
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
previous = json.loads(Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002/publication762_local.json').read_bytes())
assert all(sha256(repo/name) == value for name, value in previous['protected_files'].items())
assert sha256(draft/'ImageNativeEvidenceReader.py') == '14c7ffef0d3e540a24542c5b6f8d9ed68c5d2410dcaed40b2225fe547d547db8'
assert sha256(draft/'feature_interface_preparation763/independent_native_roles_before_feature_interface.py') == '3bb63ff185ab52495a56df138d49f182b40c31d7e1e0919bd4df2bb79ffcca00'

relative_sources = (
    'tools/run_foundation_recipe.py', 'tools/run_clean_clip_joint.py',
    'modeling/trifusion/role_global_tokens.py',
    'modeling/trifusion/correspondence_roles.py',
    'comparators/Signal-cd1b0a6/modeling/make_model.py',
    'comparators/Signal-cd1b0a6/layers/make_loss.py',
    'comparators/Signal-cd1b0a6/solver/make_optimizer.py',
    'comparators/Signal-cd1b0a6/configs/RGBNT201/Signal.yml',
    'comparators/Signal-cd1b0a6/configs/RGBNT100/Signal.yml',
    'comparators/Signal-cd1b0a6/configs/MSVR310/Signal.yml',
)
sources = {}
for relative in relative_sources:
    path = archive/relative
    sources[relative] = {'actual_read_path': str(path), 'sha256': sha256(path)}
    if not relative.startswith('comparators/'):
        assert sha256(repo/relative) == sources[relative]['sha256'], relative

syntax = {}
for name in ('ImageNativeEvidenceReader.py', 'independent_native_roles.py'):
    path = draft/name
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    syntax[name] = {'sha256': sha256(path),
                    'classes': [node.name for node in tree.body if isinstance(node, ast.ClassDef)]}
record = {
    'status': 'UNREGISTERED_RAW_FEATURE_INTERFACE_SYNTAX_PASS_ONLY',
    'recorded_at': datetime.now().astimezone().isoformat(),
    'source_facts': sources,
    'draft_syntax': syntax,
    'forward_features': ['raw_fused', 'fused', 'shared_global', 'correction'],
    'feature_only_path_runs_neck_or_classifier': False,
    'deployment_contract': 'L2-normalized1536; no new dimension or pairwise query-gallery inference',
    'training_foundation_selected': False,
    'training_plan_registered': False,
    'new_model_forwards': 0,
    'full_integration_imported_or_executed': False,
    'canonical_runtime_source_changed': False,
    'protected_local_f3_sources_unchanged': previous['protected_files'],
    'training_ports': [2026], 'physical_gpus': [0, 1, 2, 3], 'max_parallel': 4,
    'boundary': 'Actual source inspection and AST parsing only. The old component CPU witness is unchanged and does not accept the revised full integration or selected trainer. Original F3 is not modified, restarted, scored or reported by this helper.'
}
out.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: value for key, value in record.items() if key not in ('source_facts', 'protected_local_f3_sources_unchanged')}, indent=2))

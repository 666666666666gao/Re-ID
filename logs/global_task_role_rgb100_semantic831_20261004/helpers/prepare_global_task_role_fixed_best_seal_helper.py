"""Prepare, but do not invoke, the once-only post-completion input collector."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('C:/Users/gb/.codex_tmp')
source = (root/'seal_role_input_detach_fixed_best_after_complete.py').read_text(encoding='utf-8')
replacements = (
    ('role_input_detach_fixed_best','global_task_role_fixed_best'),
    ('role_input_detach_v1_20261004_813','global_task_role_v1_20261004_824'),
    ('role_input_detach_v1_complete_20261004_813','global_task_role_v1_complete_20261004_824'),
    ('role_input_detach_launch_20261004_813','global_task_role_launch_20261004_824'),
    ('trifusion-role-input-detach-v1','trifusion-global-task-role-v1'),
    ('trifusion-role-input-detach-fixed-best-diagnosis-v1','trifusion-global-task-role-fixed-best-diagnosis-v1'),
    ('tools/diagnose_role_input_detach_best.py','tools/diagnose_global_task_role_best.py'),
    ("len(summary['pairs'])==12","len(summary['pairs'])==18"),
    ('len(source)==324','len(source)==332'),
    ("len(manifest['source_sha256'])==322","len(manifest['source_sha256'])==330"),
    ("refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json","refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json"),
    ("len(controls['artifact_sha256'])==61","len(controls['artifact_sha256'])==124"),
    ("'original_control_artifacts':61","'original_control_artifacts':124"),
    ("record['source_count'] == 324","record['source_count'] == 332"),
    ('Actual role-input-detach models','Actual task-ownership models'),
)
for old,new in replacements:
    assert old in source,old
    source = source.replace(old,new)
needle = 'assert not packet.exists() and not destination.exists()'
assert source.count(needle) == 1
source = source.replace(needle, '''complete_intake = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_complete_report')
assert json.loads((complete_intake/'EXIT.json').read_bytes())['exit_code'] == 0
'''+needle,1)
module = ast.parse(source)
remote = next(node.value.value for node in module.body if isinstance(node,ast.Assign) and
              any(isinstance(target,ast.Name) and target.id=='code' for target in node.targets))
ast.parse(remote)
assert 'probe_checkpoint.pth' not in remote and 'import torch' not in remote
assert "len(summary['pairs'])==18" in remote
assert "refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json" in remote
name = 'seal_global_task_role_fixed_best_after_complete.py'
target = root/name
assert not target.exists()
target.write_text(source,encoding='utf-8')
record = dict(status='POST_COMPLETE_SEAL_HELPER_PREPARED_NOT_EXECUTED',at=datetime.now().astimezone().isoformat(),
              helper=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              conditions='Original six full50/firststrict,300epochs/12968steps,once18pair report and local completed intake exit0. Actual332source/124control hashes required. No retired M0 binary dependency.',
              boundary='Local source and AST only; no SSH/GPU/model/seal creation or current330scientific-source change.')
packet = root/'global_task_role_fixed_best_candidate_v2'
(packet/'SEAL_HELPER_PREPARATION.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))

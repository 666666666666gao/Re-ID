"""Prepare local execution/intake helpers; never launch or connect here."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

private = Path('C:/Users/gb/.codex_tmp')
candidate = private / 'global_task_role_fixed_best_candidate_v2'
receipt = candidate / 'EXECUTION_HELPERS_PREPARATION.json'
assert not receipt.exists()
launch = private / 'launch_global_task_role_fixed_best_execution_only.py'
intake = private / 'collect_global_task_role_fixed_best.py'
assert not launch.exists() and not intake.exists()

source = (private / 'launch_role_input_detach_fixed_best_execution_only.py').read_text(encoding='utf-8')
for old, new in (
    ('role_input_detach_complete_report', 'global_task_role_complete_report'),
    ('role_input_detach_fixed_best', 'global_task_role_fixed_best'),
    ('role_input_detach_v1_20261004_813', 'global_task_role_v1_20261004_824'),
    ('role_input_detach_launch_20261004_813', 'global_task_role_launch_20261004_824'),
    ('trifusion-role-input-detach-fixed-best-diagnosis-v1', 'trifusion-global-task-role-fixed-best-diagnosis-v1'),
    ("len(seal['source_sha256']) == 324", "len(seal['source_sha256']) == 332"),
    ('tools/diagnose_role_input_detach_best.py', 'tools/diagnose_global_task_role_best.py')):
    assert old in source, old
    source = source.replace(old, new)
ast.parse(source)
assert 'role_input_detach' not in source
assert 'temperature' in source and '--query-gpu=index,memory.used,memory.total,utilization.gpu' in source
launch.write_text(source, encoding='utf-8')

source = (private / 'collect_role_input_detach_fixed_best.py').read_text(encoding='utf-8')
assert 'role_input_detach_fixed_best' in source
source = source.replace('role_input_detach_fixed_best', 'global_task_role_fixed_best')
ast.parse(source)
assert 'role_input_detach' not in source
intake.write_text(source, encoding='utf-8')
receipt.write_text(json.dumps(dict(
    status='EXECUTION_AND_INTAKE_HELPERS_PREPARED_NOT_EXECUTED',
    at=datetime.now().astimezone().isoformat(),
    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (launch, intake)},
    boundary='Pure local text adaptation and AST parsing. Requires original six full50/firststrict, once18-pair report and complete intake, deployed332 source and new input seal/publication before one execution. No SSH, model construction, GPU call, training or source330 modification.'
), indent=2) + '\n', encoding='utf-8')
print(receipt.read_text(encoding='utf-8'))

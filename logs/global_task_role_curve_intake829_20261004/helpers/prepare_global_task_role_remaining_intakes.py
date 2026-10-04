"""Prepare unused text/saved-distance intakes; do not contact a server or run training."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('C:/Users/gb/.codex_tmp')
files = {}
for variant in ('semantic', 'native'):
    source = (root / f'collect_global_task_role_msvr_{variant}828.py').read_text(encoding='utf-8')
    source = source.replace('MSVR310', 'RGBNT100')
    source = source.replace(f'global_task_role_msvr_{variant}_full828', f'global_task_role_rgb100_{variant}_full829')
    source = source.replace('RGBNT201 and RGBNT100 pairs closed; six-endpoint campaign and once-only report pending.',
                            'This original RGBNT100 endpoint closed; full campaign and original once-only CPU report must be separately verified.')
    target = root / f'collect_global_task_role_rgb100_{variant}829.py'
    assert not target.exists()
    ast.parse(source)
    assert f"('RGBNT100','{variant}','full')" in source
    assert 'global_task_role_msvr_' not in source and 'MSVR310' not in source
    assert '==3 for r in steps' in source
    target.write_text(source, encoding='utf-8')
    files[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()

source = (root / 'collect_role_input_detach_complete_report.py').read_text(encoding='utf-8')
for old, new in (
    ('role_input_detach_complete_report', 'global_task_role_complete_report'),
    ('role_input_detach_v1_20261004_813', 'global_task_role_v1_20261004_824'),
    ('role_input_detach_launch_20261004_813', 'global_task_role_launch_20261004_824'),
    ('role_input_detach_v1_complete_20261004_813', 'global_task_role_v1_complete_20261004_824'),
    ("==322", "==330"), ('len(report[\'pairs\'])==12', 'len(report[\'pairs\'])==18'),
    ('refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json', 'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'),
    ("==61", "==124"), ('trifusion-role-input-detach-v1', 'trifusion-global-task-role-v1'),
    ("'source_files_verified_unchanged':322", "'source_files_verified_unchanged':330"),
    ("'original61control_artifacts_unchanged'", "'previous124control_artifacts_unchanged'"),
    ("if k not in ('architecture','entry_sha256','role_input_gradient_policy')", "if k not in ('architecture','entry_sha256','scope','objective_gradient_policy')"),
    ("if k not in ('architecture','entry_sha256')", "if k not in ('architecture','entry_sha256','scope','objective_gradient_policy')"),
):
    assert old in source, old
    source = source.replace(old, new)
needle = "assert all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())"
assert needle in source
source = source.replace(needle, needle + "\nhistorical_path=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'\nhistorical=json.loads(historical_path.read_text())\nassert len(historical['rows'])==9 and len(historical['artifact_sha256'])==61\nassert all(sha(Path(name))==digest for name,digest in historical['artifact_sha256'].items())", 1)
needle = "'previous124control_artifacts_unchanged':True,"
assert needle in source
source = source.replace(needle, needle + "'historical61control_artifacts_unchanged':True,'historical_control_seal_sha256':sha(historical_path),", 1)
target = root / 'collect_global_task_role_complete_report.py'
assert not target.exists()
ast.parse(source)
assert "len(report['pairs'])==18" in source and "'formal_steps':12968" in source
assert "'source_files_verified_unchanged':330" in source
assert "torch" not in source
target.write_text(source, encoding='utf-8')
files[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
packet = root / 'independent_evidence_draft/global_task_role_remaining_intake_preparation'
assert not packet.exists()
packet.mkdir()
record = dict(status='PREPARED_NOT_EXECUTED', at=datetime.now().astimezone().isoformat(), files=files,
    campaign='logs/global_task_role_v1_20261004_824', original_once_report='results/global_task_role_v1_complete_20261004_824',
    formal_steps_expected=12968, formal_epochs_expected=300, saved_distance_pairs_expected=18,
    boundary='Local AST and explicit schema checks only; no server connection, model call, report replay, new training, or scientific-source mutation. RGBNT100 endpoint intake requires original full completion; final intake requires all12 original jobs and single report exit0. Existing M0 probes remain dependencies until final intake is received.')
(packet / 'PREPARATION.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))

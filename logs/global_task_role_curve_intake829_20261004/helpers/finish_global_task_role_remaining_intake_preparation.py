"""Finish only the uncreated final intake after a local overbroad text assertion."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('C:/Users/gb/.codex_tmp')
files = {}
for variant in ('semantic', 'native'):
    target = root / f'collect_global_task_role_rgb100_{variant}829.py'
    ast.parse(target.read_text(encoding='utf-8'))
    files[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
original = (root / 'prepare_global_task_role_remaining_intakes.py').read_text(encoding='utf-8')
marker = "\nsource = (root / 'collect_role_input_detach_complete_report.py')"
assert original.count(marker) == 1
remaining = marker[1:] + original.split(marker, 1)[1]
assert remaining.count('assert "torch" not in source') == 1
remaining = remaining.replace('assert "torch" not in source', 'assert "import torch" not in source and "from torch" not in source')
ast.parse(remaining)
failure = root / 'independent_evidence_draft/global_task_role_remaining_intake_preparation_failure.json'
assert not failure.exists()
failure.write_text(json.dumps(dict(status='LOCAL_GENERATOR_ASSERTION_FAILURE_PRESERVED', exit_code=1,
    at=datetime.now().astimezone().isoformat(), script='prepare_global_task_role_remaining_intakes.py',
    reason='Overbroad substring check matched the literal no-torch boundary comment. Both RGBNT100 helpers were written; final collector and preparation packet were not written.',
    repair='Finish only remaining local generation, checking import syntax strings. No server call, scientific-source edit, or replay of already written helpers.'), indent=2) + '\n', encoding='utf-8')
exec(compile(remaining, '<finish-only-final-intake>', 'exec'))

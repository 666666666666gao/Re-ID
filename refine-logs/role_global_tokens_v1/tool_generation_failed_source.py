"""Create independent queue/collector forks; leave predecessor source unchanged."""

import ast
import hashlib
from pathlib import Path

ROOT = Path('/data/gaob/Re-ID/Trifusion')


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


queue = (ROOT / 'tools/queue_slot_competition_fp32_roles.py').read_text()
queue = queue.replace('Run the registered independent/competitive allocation panel on four GPUs.',
                      'Run the registered static/token/direct interaction panel on free GPUs.')
queue = replace_once(queue, 'CONDITIONS = {"competitive": "competitive", "independent": "independent"}',
                     'CONDITIONS = {"token": "token", "static": "static", "direct": "direct"}')
for before, after in (
    ('tools/run_slot_competition_fp32_roles.py', 'tools/run_role_global_tokens.py'),
    ('tools/queue_slot_competition_fp32_roles.py', 'tools/queue_role_global_tokens.py'),
    ('tools/collect_slot_competition_fp32_roles.py', 'tools/collect_role_global_tokens.py'),
    ('tools/check_slot_competition_fp32_roles_cpu.py', 'tools/check_role_global_tokens_cpu.py'),
    ('tools.collect_slot_competition_fp32_roles', 'tools.collect_role_global_tokens'),
    ('"slot_competition_fp32"', '"role_global_tokens"'),
    ('"--attention-normalization"', '"--token-mode"'),
    ('"trifusion-slot-competition-fp32-roles-panel-v2"', '"trifusion-role-global-tokens-panel-v1"'),
    ('"trifusion-slot-competition-fp32-roles-verification-v2"', '"trifusion-role-global-tokens-verification-v1"'),
):
    assert before in queue, before
    queue = queue.replace(before, after)
queue = replace_once(queue, '    "modeling/trifusion/slot_competition_fp32_roles.py",',
                     '    "modeling/trifusion/role_global_tokens.py",\n'
                     '    "refine-logs/role_global_tokens_v1/EXPERIMENT_PLAN.md",\n'
                     '    "modeling/trifusion/slot_competition_fp32_roles.py",')
queue = replace_once(queue, 'assert len(previous_sources) == 210', 'assert len(previous_sources) == 213')
queue = replace_once(queue, '"attention_normalizations": list(CONDITIONS)',
                     '"token_modes": list(CONDITIONS), "transformer_sequence_length": 17, "attention_normalization": "independent"')
queue = replace_once(queue,
    '"boundary": "Independent versus competitive allocation uses the same trainable tensors, seed, full128 support, data, budget, "\n'
    '                    "role operators, fixed positions, attention projections and readout. Six real eight-batch M0 checks precede each full50 run; "',
    '"boundary": "Static versus token versus direct uses the same trainable tensors, seed, full128 support, data, budget, "\n'
    '                    "17-token Transformer, fixed positions and readout. Nine real eight-batch M0 checks precede each full50 run; "')
queue = replace_once(queue,
    'assert accepted["expected_endpoints"] == accepted["verified_complete"] == len(accepted["rows"]) == 6',
    'assert accepted["expected_endpoints"] == accepted["verified_complete"] == len(accepted["rows"]) == 9')

collector = (ROOT / 'tools/collect_slot_competition_fp32_roles.py').read_text()
collector = collector.replace('complete Patch-memory runs', 'complete global-token runs')
for before, after in (
    ('tools.queue_slot_competition_fp32_roles', 'tools.queue_role_global_tokens'),
    ('tools/run_slot_competition_fp32_roles.py', 'tools/run_role_global_tokens.py'),
    ('"slot_competition_fp32"', '"role_global_tokens"'),
    ('"slot_competition_fp32_roles_v2"', '"role_global_tokens_v1"'),
    ('"trifusion-slot-competition-fp32-roles-v2"', '"trifusion-role-global-tokens-v1"'),
    ('"trifusion-slot-competition-fp32-roles-panel-v2"', '"trifusion-role-global-tokens-panel-v1"'),
):
    assert before in collector, before
    collector = collector.replace(before, after)
collector = replace_once(collector, '    normalization = CONDITIONS[variant]',
                         '    token_mode = CONDITIONS[variant]\n    normalization = "independent"')
collector = replace_once(collector, '    assert binding["architecture"] == "role_global_tokens_v1"',
                         '    assert binding["architecture"] == "role_global_tokens_v1"\n'
                         '    assert binding["token_mode"] == token_mode\n'
                         '    assert binding["transformer_sequence_length"] == 17')
collector = replace_once(collector,
    '    for field, path in (("model_source_sha256", "modeling/trifusion/correspondence_roles.py"),',
    '    for field, path in (("role_global_source_sha256", "modeling/trifusion/role_global_tokens.py"),\n'
    '                        ("model_source_sha256", "modeling/trifusion/correspondence_roles.py"),')
collector = replace_once(collector,
    '    assert probe_payload["schema"] == "trifusion-role-global-tokens-v1"',
    '    assert probe_payload["schema"] == "trifusion-role-global-tokens-v1"\n'
    '    assert probe_payload["token_mode"] == token_mode')
collector = replace_once(collector,
    '    assert payload["schema"] == "trifusion-role-global-tokens-v1"',
    '    assert payload["schema"] == "trifusion-role-global-tokens-v1"\n'
    '    assert payload["token_mode"] == official["token_mode"] == token_mode')
collector = replace_once(collector,
    '            "condition": condition, "memory_mode": memory_mode, "attention_normalization": normalization,',
    '            "condition": condition, "memory_mode": memory_mode, "attention_normalization": normalization,\n'
    '            "token_mode": token_mode,')
collector = replace_once(collector, '"expected_endpoints": 6,', '"expected_endpoints": 9,')

for name, source in (('queue_role_global_tokens.py', queue), ('collect_role_global_tokens.py', collector)):
    target = ROOT / 'tools' / name
    assert not target.exists(), target
    ast.parse(source)
    target.write_text(source)
    print(name, hashlib.sha256(target.read_bytes()).hexdigest())

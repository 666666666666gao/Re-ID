"""Independent read-only review of derived publication text and all pooled values."""
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT.parent / 'trifusion_gradient_complete_interpretation_20260920'
INTAKE = ROOT.parent / 'smooth_ap_objective_gradient_source_complete_20260920'
SOURCE = ROOT / 'snapshots/terminal_texts/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920'
SNAPSHOT = ROOT / 'snapshots/interpretation_review_round2'
SNAPSHOT.mkdir(parents=True, exist_ok=False)

def seal(src, dst):
    blob = src.read_bytes()
    dst.write_bytes(blob)
    return {'path': str(src), 'snapshot': str(dst.relative_to(ROOT)), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}

inputs = [seal(DERIVED / name, SNAPSHOT / name) for name in ['MSVR310_SMOOTH_AP_SOURCE_OBJECTIVE_GRADIENTS_2026-09-20.md', 'pooled_descriptive_statistics.json']]
inputs += [seal(INTAKE / name, SNAPSHOT / name) for name in ['inventory.json', 'intake.json']]
pooled = json.loads((SNAPSHOT / 'pooled_descriptive_statistics.json').read_text(encoding='utf-8'))
report = (SNAPSHOT / 'MSVR310_SMOOTH_AP_SOURCE_OBJECTIVE_GRADIENTS_2026-09-20.md').read_text(encoding='utf-8')
assert '真实学习信号' not in report
assert '所有分支梯度都相互冲突' not in report
assert '在这些固定终态的各角色参数块内，F与O并非普遍反向' in report
assert '本诊断不能评价不同角色之间的梯度冲突或原训练全程' in report
assert '固定终态和来源视图上具有非零encoder导数' in report
assert 'WARN / CLOSED_WITH_LIMITS' in report and '同族审计 provisional' in report
rows = list(csv.DictReader((SOURCE / 'analysis/all_role_steps.csv').open(encoding='utf-8', newline='')))
for row in rows:
    for key in row:
        if key in ('fold', 'step', 'epoch'):
            row[key] = int(row[key])
        elif key not in ('endpoint', 'role', 'active_fused_metric'):
            row[key] = float(row[key]) if row[key] else None
assert len(rows) == 4680
for field, path in [('analysis_sha256', SOURCE / 'analysis/analysis.json'), ('csv_sha256', SOURCE / 'analysis/all_role_steps.csv')]:
    assert pooled[field] == hashlib.sha256(path.read_bytes()).hexdigest(), field

def distribution(values):
    defined = [x for x in values if x is not None]
    return {'defined': len(defined), 'undefined': len(values) - len(defined), 'median': statistics.median(defined), 'minimum': min(defined), 'maximum': max(defined)}

windows = []
for endpoint in ('control', 'smooth_ap'):
    for window, lo, hi in [('warmup', 1, 65), ('active', 66, 260), ('last65', 196, 260)]:
        selected = [r for r in rows if r['endpoint'] == endpoint and lo <= r['step'] <= hi]
        result = {
            'endpoint': endpoint, 'window': window, 'role_rows': len(selected),
            'negative_fused_other': sum(r['fused_other_cosine'] is not None and r['fused_other_cosine'] < 0 for r in selected),
            'negative_fused_full': sum(r['fused_full_cosine'] is not None and r['fused_full_cosine'] < 0 for r in selected),
            'zero_fused_norm': sum(r['fused_norm'] == 0 for r in selected),
            'fused_not_above_sum_repeat_differences': sum(r['fused_norm'] <= r['current_repeat_difference'] + r['history_repeat_difference'] for r in selected),
        }
        for metric in ('fused_other_ratio', 'fused_full_ratio', 'fused_other_cosine', 'fused_full_cosine'):
            result[metric] = distribution([r[metric] for r in selected])
        windows.append(result)
assert windows == pooled['windows'], 'pooled values differ from exact raw-CSV recomputation'
assert pooled['max_current_identity_error'] == max(r['current_identity_error'] for r in rows)
assert pooled['max_full_identity_error'] == max(r['full_identity_error'] for r in rows)

table = []
for line in report.splitlines():
    if re.match(r'^\| [012] \|', line):
        cells = [c.strip() for c in line.strip('|').split('|')]
        fold, role = int(cells[0]), cells[1]
        control = [r for r in rows if r['fold'] == fold and r['role'] == role and r['endpoint'] == 'control' and r['step'] >= 66]
        smooth = [r for r in rows if r['fold'] == fold and r['role'] == role and r['endpoint'] == 'smooth_ap' and r['step'] >= 66]
        assert len(control) == len(smooth) == 195
        expected = [str(fold), role,
                    format(statistics.median(r['fused_other_ratio'] for r in control), '.6f'),
                    format(statistics.median(r['fused_other_ratio'] for r in smooth), '.6f'),
                    format(statistics.median(r['fused_other_cosine'] for r in smooth), '.6f'),
                    str(sum(r['fused_other_cosine'] < 0 for r in smooth)),
                    str(sum(r['fused_full_cosine'] < 0 for r in smooth))]
        assert cells == expected, (cells, expected)
        table.append(expected)
assert len(table) == 9
last65_group_medians = [statistics.median(r['fused_other_ratio'] for r in rows if r['endpoint'] == 'smooth_ap' and r['step'] >= 196 and r['fold'] == fold and r['role'] == role) for fold in range(3) for role in ('cnn', 'transformer', 'mamba')]
assert format(min(last65_group_medians), '.6f') == '0.126566'
assert format(max(last65_group_medians), '.6f') == '0.192601'
repeat_by_role = {}
for role in ('cnn', 'transformer', 'mamba'):
    selected = [r for r in rows if r['role'] == role]
    repeat_by_role[role] = {'rows': len(selected), 'maximum_current_difference': max(r['current_repeat_difference'] for r in selected), 'maximum_history_difference': max(r['history_repeat_difference'] for r in selected)}
    if role != 'mamba':
        assert repeat_by_role[role]['maximum_current_difference'] == repeat_by_role[role]['maximum_history_difference'] == 0
assert repeat_by_role['mamba']['maximum_current_difference'] > 0
assert repeat_by_role['mamba']['maximum_history_difference'] > 0

inventory = json.loads((SNAPSHOT / 'inventory.json').read_text(encoding='utf-8'))
sealed_remote = {r['path']: r for r in json.loads((ROOT / 'AUDITED_INPUTS.json').read_text(encoding='utf-8'))['remote_inputs']}
intake_matches = []
for entry in inventory['files']:
    p = INTAKE / entry['name']
    blob = p.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    assert len(blob) == entry['bytes'] and digest == entry['sha256'], entry['name']
    assert entry['remote'] in sealed_remote, entry['remote']
    scientific = sealed_remote[entry['remote']]
    assert digest == scientific['sha256'] and len(blob) == scientific['bytes'], entry['remote']
    intake_matches.append({'name': entry['name'], 'bytes': len(blob), 'sha256': digest, 'matches_auditor_independent_remote_input': True})
assert len(intake_matches) == 33
assert sum(r['bytes'] for r in intake_matches) == 37743954

result = {
    'status': 'PASS_WITH_STATED_DIAGNOSTIC_LIMITS',
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'method': 'Independent stdlib CSV arithmetic against independently checked original source rows; exact equality of all derived pooled numerical fields; complete intake byte/SHA comparison against auditor independent remote scope.',
    'inputs': inputs,
    'raw_role_rows': len(rows), 'pooled_windows': windows,
    'max_current_identity_error': pooled['max_current_identity_error'],
    'max_full_identity_error': pooled['max_full_identity_error'],
    'all_numerical_fields_exact_match': True,
    'report_table_rows': table,
    'smooth_last65_group_median_range': [min(last65_group_medians), max(last65_group_medians)],
    'repeat_by_role': repeat_by_role,
    'intake_file_count': len(intake_matches), 'intake_bytes': sum(r['bytes'] for r in intake_matches), 'intake_files': intake_matches,
    'initial_report_sha256': '93ac261af5e670648dbd03c0c4cf7add1fb1497401b1ca5c2a22531fb1bb3e8e',
    'initial_report_capture': 'Full text was read in the auditor tool trace before root revised it; original numeric JSON remains unchanged. Exact initial report bytes were not locally snapshotted before root revision.',
    'semantic_issues': [
        {'id': 'INTERP1', 'status': 'RESOLVED', 'severity': 'required_qualification', 'line': 35, 'original': '所有分支梯度都相互冲突', 'reason': 'Observed cosines compare F and O within each role parameter block. Pairwise gradients between different role blocks are not a defined or measured quantity here.', 'resolution': 'Revised text explicitly restricts the conclusion to F/O within each fixed-terminal role block and excludes cross-role or original-training claims.'},
        {'id': 'INTERP2', 'status': 'RESOLVED', 'severity': 'recommended_precision', 'line': 29, 'original': '真实学习信号', 'reason': 'The observed nonzero local derivative above repeated-computation differences does not establish useful learning or generalization.', 'resolution': 'Revised text now reports nonzero encoder derivatives in these fixed terminal/source views above saved repeat differences.'},
        {'id': 'INTERP3', 'status': 'RESOLVED_PENDING_ROOT_ARCHIVAL_ACTION', 'severity': 'publication_update', 'lines': [3, 41], 'reason': 'Revised text now reports the actual WARN/CLOSED_WITH_LIMITS result and intended final audit link; physical publication is a root action.', 'prior_source_ranking_claim': 'Historical Q1 result only; not a new ranking evaluation or causal outcome from this gradient diagnostic.'},
    ],
}
(ROOT / 'DERIVED_INTERPRETATION_VERIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'all_numerical_fields_exact_match': True, 'raw_role_rows': len(rows), 'windows': len(windows), 'table_rows': len(table), 'repeat_by_role': repeat_by_role, 'intake_file_count': 33, 'intake_bytes': 37743954}, ensure_ascii=False, indent=2))

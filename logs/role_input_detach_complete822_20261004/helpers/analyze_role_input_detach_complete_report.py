"""Derive all six result accounts from the collected original CPU report only."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_input_detach_complete_report')
assert json.loads((packet / 'EXIT.json').read_bytes())['exit_code'] == 0
receipt = json.loads((packet / 'stdout.json').read_bytes())
assert receipt['status'] == 'ORIGINAL_ONCE_ONLY_FINAL_CPU_REPORT_AND_ARTIFACTS_VERIFIED'
assert receipt['formal_completed'] == 6 and receipt['formal_epochs'] == 300 and receipt['formal_steps'] == 12968
assert receipt['report_invocations'] == 1 and receipt['report_exit_code'] == 0
summary_path = packet / 'texts/role_input_detach_v1_complete_20261004_813_SUMMARY.json'
summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
remote_name = 'results/role_input_detach_v1_complete_20261004_813/SUMMARY.json'
assert summary_sha == receipt['files'][remote_name]['sha256']
summary = json.loads(summary_path.read_bytes())
assert summary['schema'] == 'trifusion-role-input-detach-v1' and summary['status'] == 'COMPLETE'
assert len(summary['rows']) == 6 and len(summary['pairs']) == 12
rows = {(row['dataset'], row['variant']): row for row in summary['rows']}
datasets = ('RGBNT201', 'MSVR310', 'RGBNT100')
assert set(rows) == {(dataset, variant) for dataset in datasets for variant in ('semantic', 'native')}
assert not (packet / 'ANALYSIS.json').exists() and not (packet / 'ANALYSIS.md').exists()
interventions = []
training_accounts = []
for dataset in datasets:
    for variant in ('semantic', 'native'):
        row = rows[(dataset, variant)]
        history = row['history']
        assert len(history) == 50 and [item['epoch'] for item in history] == list(range(1, 51))
        assert history[row['best_epoch'] - 1]['official_fused'] == row['metrics']
        comparison = next(item for item in summary['pairs'] if (item['dataset'], item['variant'], item['control']) == (dataset, variant, 'original_' + variant))
        assert comparison['actual_training_batch_order_equal']
        diagnosis = comparison['paired_diagnosis']
        assert diagnosis['status'] == 'CPU_ARRAY_PARITY_AND_PAIRED_DIAGNOSIS_COMPLETE'
        assert abs(diagnosis['metrics'][variant]['mAP'] - row['metrics']['mAP']) < 1e-5
        interventions.append({'dataset': dataset, 'variant': variant, 'phase_progress': comparison['phase_progress'],
            'delta_metrics': diagnosis['delta_metrics'], 'rank1_repairs': diagnosis['rank1_repairs'],
            'rank1_new_errors': diagnosis['rank1_new_errors'],
            'identity_macro_mean_delta_ap_points': diagnosis['identity_macro_mean_delta_ap_points'],
            'identity_ap_improved': diagnosis['identity_ap_improved'], 'identity_ap_worsened': diagnosis['identity_ap_worsened']})
        training_accounts.append({'dataset': dataset, 'variant': variant, 'best_epoch': row['best_epoch'],
            'best_metrics': row['metrics'], 'last_metrics': history[-1]['official_fused'],
            'best_minus_last_mAP': row['metrics']['mAP'] - history[-1]['official_fused']['mAP'],
            'training_loop_seconds': sum(item['seconds'] for item in history),
            'training_and_epoch_evaluation_seconds': row['training_and_epoch_evaluation_seconds']})

detail_pairs = []
for dataset in datasets:
    semantic = rows[(dataset, 'semantic')]
    native = rows[(dataset, 'native')]
    common = {}
    for variant in ('semantic', 'native'):
        pair = next(item for item in summary['pairs'] if (item['dataset'], item['variant'], item['control']) == (dataset, variant, 'original_global_only'))
        common[variant] = pair['paired_diagnosis']['query_changes']
    assert len(common['semantic']) == len(common['native'])
    identity_deltas = defaultdict(list)
    delta_ap = []
    repairs = new_errors = 0
    for index, (left, right) in enumerate(zip(common['semantic'], common['native'], strict=True)):
        assert left['query_index'] == right['query_index'] == index
        assert left['identity'] == right['identity']
        assert left['control_ap'] == right['control_ap'] and left['control_first_rank'] == right['control_first_rank']
        value = right['candidate_ap'] - left['candidate_ap']
        assert math.isfinite(value)
        delta_ap.append(value)
        identity_deltas[left['identity']].append(value)
        repairs += left['candidate_first_rank'] != 1 and right['candidate_first_rank'] == 1
        new_errors += left['candidate_first_rank'] == 1 and right['candidate_first_rank'] != 1
    metrics = {name: native['metrics'][name] - semantic['metrics'][name] for name in semantic['metrics']}
    assert abs(metrics['mAP'] - sum(delta_ap) * 100 / len(delta_ap)) < 1e-5
    assert abs(metrics['Rank-1'] - (repairs - new_errors) * 100 / len(delta_ap)) < 1e-5
    identities = [{'identity': identity, 'queries': len(values), 'mean_delta_ap_points': sum(values) * 100 / len(values)} for identity, values in sorted(identity_deltas.items())]
    detail_pairs.append({'dataset': dataset, 'comparison': 'new_native_minus_new_semantic', 'query_count': len(delta_ap),
        'delta_metrics': metrics, 'rank1_repairs': repairs, 'rank1_new_errors': new_errors,
        'query_ap_improved': sum(value > 1e-8 for value in delta_ap), 'query_ap_worsened': sum(value < -1e-8 for value in delta_ap),
        'identity_macro_mean_delta_ap_points': sum(item['mean_delta_ap_points'] for item in identities) / len(identities),
        'identity_ap_improved': sum(item['mean_delta_ap_points'] > 1e-6 for item in identities),
        'identity_ap_worsened': sum(item['mean_delta_ap_points'] < -1e-6 for item in identities),
        'identity_changes': identities, 'boundary': 'Descriptive same-intervention detail comparison derived from existing all-query scores; not a new gradient-intervention acceptance gate or new model execution.'})

analysis = {'status': 'ALL_SIX_RESULT_ACCOUNTS_DERIVED_FROM_ORIGINAL_COMPLETE_REPORT',
    'at': datetime.now().astimezone().isoformat(), 'summary_sha256': summary_sha,
    'interventions': interventions, 'phase_progress_count': sum(item['phase_progress'] for item in interventions),
    'training_accounts': training_accounts, 'new_detail_pairs': detail_pairs,
    'boundary': 'Original once-only CPU report retained unchanged. All six endpoints and all queries included; no new checkpoint selection, model/scorer call, seed, training, gain search or acceptance rule. Fixed-model identity statistics do not establish training-seed stability. New fixed-best g/c/h/f diagnosis remains separate; scientific Goal active/unmet.'}
(packet / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n', encoding='utf-8')
lines = ['# 读取梯度边界：完整六端结果账', '', '| 数据集 | 条件 | best轮 | mAP | R1 | R5 | R10 | best−末轮mAP |', '|---|---|---:|---:|---:|---:|---:|---:|']
for row in training_accounts:
    metric = row['best_metrics']
    lines.append(f'| {row["dataset"]} | {row["variant"]} | {row["best_epoch"]} | {metric["mAP"]:.4f} | {metric["Rank-1"]:.4f} | {metric["Rank-5"]:.4f} | {metric["Rank-10"]:.4f} | {row["best_minus_last_mAP"]:.4f} |')
lines += ['', '| 干预：新−对应原变体 | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP | 事前推进条件 |', '|---|---:|---:|---:|---:|---|']
for row in interventions:
    lines.append(f'| {row["dataset"]} {row["variant"]} | {row["delta_metrics"]["mAP"]:+.4f} | {row["delta_metrics"]["Rank-1"]:+.4f} | {row["rank1_repairs"]}/{row["rank1_new_errors"]} | {row["identity_macro_mean_delta_ap_points"]:+.4f} | {"满足" if row["phase_progress"] else "未满足"} |')
lines += ['', '| 同干预native−semantic（描述性） | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP |', '|---|---:|---:|---:|---:|']
for row in detail_pairs:
    lines.append(f'| {row["dataset"]} | {row["delta_metrics"]["mAP"]:+.4f} | {row["delta_metrics"]["Rank-1"]:+.4f} | {row["rank1_repairs"]}/{row["rank1_new_errors"]} | {row["identity_macro_mean_delta_ap_points"]:+.4f} |')
lines += ['', analysis['boundary'], '']
(packet / 'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({key: value for key, value in analysis.items() if key not in ('training_accounts', 'new_detail_pairs')}, indent=2))

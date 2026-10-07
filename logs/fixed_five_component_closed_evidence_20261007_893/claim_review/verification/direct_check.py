"""Private reviewer: stdlib verification of supplied bytes and serialized scores only."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
PACKET = ROOT / 'fixed_five_complete893'
SOURCES = ROOT / 'qualified_five_audit_sources891'
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
TRACE = Path(__file__).parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

data = read(PACKET / 'REMOTE.json')
mapping = read(PACKET / 'FILE_MAP.json')
source_map = read(SOURCES / 'FILE_MAP.json')
old_map = read(ROOT / 'qualified_five_full_complete889/FILE_MAP.json')
seal = read(PACKET / 'primary_text/0018_FIXED_FIVE_DIAGNOSIS_INPUT_SEAL.json')
cpu_summary = read(ROOT / 'fixed_five_component_analysis893/SUMMARY.json')
old_summary = read(ROOT / 'qualified_five_full_complete889/primary_text/0022_SUMMARY.json')
local_receipts = read(ROOT / 'qualified_five_full_complete889/LOCAL_RECEIPT_SUMMARY.json')
assert len(mapping) == len(data['files']) == 19
for name, item in mapping.items():
    payload = Path(item['local_file']).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == item['sha256'] == data['files'][name]['sha256']
    assert len(payload) == item['bytes'] == data['files'][name]['bytes']
    assert payload.decode('utf-8') == data['files'][name]['text']
for name, item in source_map.items():
    p = SOURCES / item['local_file']
    assert sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], name
for name, item in old_map.items():
    p = Path(item['local_file'])
    assert sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], name
new_names = {
    'tools/diagnose_incremental_fixed_best.py',
    'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_PLAN_20261007_892.md',
    'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_SOURCE_REVIEW_20261007_892.md',
    'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_SOURCE_REVIEW_20261007_892.json',
}
assert set(seal['source_sha256']) - set(source_map) == new_names
sealed_local = {name: SOURCES / source_map[name]['local_file'] for name in set(seal['source_sha256']) - new_names}
sealed_local.update({name: REPO / name for name in new_names})
assert len(sealed_local) == len(seal['source_sha256']) == data['source_count'] == 424
assert len(seal['artifact_sha256']) == data['artifact_count'] == 75
assert all(sha(sealed_local[name]) == digest for name, digest in seal['source_sha256'].items())
assert sha(REPO / 'tools/diagnose_incremental_fixed_best.py') == '3d8ca39b2418a3379224768be927ea28a96cd2d0b5253b054b2defb09b4785c3'
assert read(PACKET / 'EXIT.json')['exit_code'] == data['original_exit']['exit_code'] == 0
assert data['original_exit'] == read(PACKET / 'primary_text/0001_EXIT.json')
assert data['campaign'] == read(PACKET / 'primary_text/0017_campaign.json')
assert data['campaign']['status'] == 'COMPLETE' and data['campaign']['optimizer_updates'] == 0
assert all(j['status'] == 'COMPLETE' and j['exit_code'] == 0 for j in data['campaign']['jobs'])
jobs = {('RGBNT201', 'md_batch_ratio'), ('RGBNT201', 'repair_keep'),
        ('MSVR310', 'md_batch_ratio'), ('MSVR310', 'repair_keep'), ('RGBNT100', 'md_batch_ratio')}
assert {(r['dataset'], r['objective']) for r in data['rows']} == jobs
assert len(data['rows']) == len(data['campaign']['jobs']) == len(seal['rows']) == 5
expected = {'RGBNT201': (836, 836, 30), 'MSVR310': (591, 1055, 52), 'RGBNT100': (1715, 8575, 50)}
pairs = {'same_model_global_to_fused': ('global', 'fused'),
         'independent_global_only_to_same_model_global': ('independent_global_only', 'global'),
         'independent_global_only_to_fused': ('independent_global_only', 'fused')}
result_rows = []
counts = defaultdict(int)
for row in data['rows']:
    dataset, objective = row['dataset'], row['objective']
    nq, ng, ni = expected[dataset]
    original_name = f'logs/fixed_five_incremental_best_diagnosis_20261007_892/{dataset}_{objective}/DIAGNOSIS.json'
    assert row == read(Path(mapping[original_name]['local_file']))
    sealed = next(r for r in seal['rows'] if (r['dataset'], r['variant']) == (dataset, objective))
    old = next(r for r in old_summary['rows'] if (r['dataset'], r['variant']) == (dataset, objective))
    global_row = next(r for r in seal['global_rows'] if r['dataset'] == dataset)
    assert row['input_checkpoint_sha256'] == sealed['checkpoint_sha256'] == old['checkpoint_sha256']
    assert row['original_receipt_sha256'] == sealed['receipt_sha256'] == old['receipt_sha256']
    assert row['selected_epoch'] == sealed['best_epoch'] == old['best_epoch']
    assert row['binding'] == sealed['initializer'] == old['initializer']
    assert row['binding']['seed'] == 42
    assert row['model_state_before_sha256'] == row['model_state_after_sha256']
    assert row['fused_distance_max_absolute_difference_from_original'] == 0.0
    assert len(row['artifacts']) == 3
    assert all(v['bytes'] > 0 and len(v['sha256']) == 64 for v in row['artifacts'].values())
    for name, score in row['scores'].items():
        ap, ranks = score['average_precision'], score['first_match_rank']
        assert len(ap) == len(ranks) == nq
        assert all(math.isfinite(a) and 0 <= a <= 1 for a in ap)
        assert all(type(r) is int and 1 <= r <= ng for r in ranks)
        values = {'mAP': statistics.mean(ap) * 100}
        values.update({f'Rank-{k}': sum(r <= k for r in ranks) * 100 / nq for k in (1, 5, 10)})
        assert all(abs(v - score['metrics'][k]) < 1e-10 for k, v in values.items())
        counts['score_tables'] += 1
        counts['score_query_rows'] += nq
        counts['score_ap_and_rank_values'] += 2 * nq
    assert all(abs(row['scores']['fused']['metrics'][k] - v) < 1e-5 for k, v in sealed['metrics'].items())
    assert all(abs(row['scores']['independent_global_only']['metrics'][k] - v) < 1e-5 for k, v in global_row['metrics'].items())
    identity_order = [q['identity'] for q in row['comparisons']['same_model_global_to_fused']['query_changes']]
    for pair_name, (a, b) in pairs.items():
        pair = row['comparisons'][pair_name]
        first, second = row['scores'][a], row['scores'][b]
        changes = pair['query_changes']
        assert len(changes) == nq and [q['identity'] for q in changes] == identity_order
        grouped = defaultdict(list)
        for i, q in enumerate(changes):
            assert q == dict(query_index=i, identity=identity_order[i],
                control_ap=first['average_precision'][i], candidate_ap=second['average_precision'][i],
                control_first_rank=first['first_match_rank'][i], candidate_first_rank=second['first_match_rank'][i])
            grouped[q['identity']].append(q['candidate_ap'] - q['control_ap'])
        repairs = sum(a != 1 and b == 1 for a, b in zip(first['first_match_rank'], second['first_match_rank']))
        errors = sum(a == 1 and b != 1 for a, b in zip(first['first_match_rank'], second['first_match_rank']))
        assert repairs == pair['rank1_repairs'] and errors == pair['rank1_new_errors']
        for k, delta in pair['delta_metrics'].items():
            assert abs(delta - (second['metrics'][k] - first['metrics'][k])) < 1e-10
        assert abs(pair['delta_metrics']['Rank-1'] - 100 * (repairs - errors) / nq) < 1e-10
        assert len(grouped) == len(pair['identity_changes']) == ni
        assert {x['identity'] for x in pair['identity_changes']} == set(grouped)
        for identity in pair['identity_changes']:
            values = grouped[identity['identity']]
            assert len(values) == identity['queries']
            assert abs(statistics.mean(values) * 100 - identity['mean_delta_ap_points']) < 1e-10
        assert abs(statistics.mean(statistics.mean(v) * 100 for v in grouped.values()) - pair['identity_macro_delta_ap_points']) < 1e-10
        counts['paired_comparisons'] += 1
        counts['paired_query_rows'] += nq
        counts['paired_identity_rows'] += ni
    prior = next(p['paired_diagnosis'] for p in old_summary['pairs']
                 if p['dataset'] == dataset and p['objective'] == objective and p['control'] == 'raw_global_only')
    assert prior['query_changes'] == row['comparisons']['independent_global_only_to_fused']['query_changes']
    counts['prior_paired_query_rows_exact'] += nq
    own = row['comparisons']['same_model_global_to_fused']
    up = sum(q['candidate_ap'] > q['control_ap'] for q in own['query_changes'])
    down = sum(q['candidate_ap'] < q['control_ap'] for q in own['query_changes'])
    cpu = next(r for r in cpu_summary['rows'] if (r['dataset'], r['objective']) == (dataset, objective))
    assert (up, down, nq-up-down) == (cpu['query_ap_improved'], cpu['query_ap_worsened'], cpu['query_ap_unchanged'])
    assert cpu['metrics'] == {k:v['metrics'] for k,v in row['scores'].items()}
    assert cpu['same_model_delta'] == own['delta_metrics']
    assert cpu['repairs'] == own['rank1_repairs'] and cpu['new_errors'] == own['rank1_new_errors']
    assert cpu['independent_to_own_global_delta'] == row['comparisons']['independent_global_only_to_same_model_global']['delta_metrics']
    for split, n in [('query', nq), ('gallery', ng)]:
        assert all(d['count'] == n for d in row['diagnostic'][split].values())
        counts['evaluation_record_instances'] += n
    g, ig = row['scores']['global'], row['scores']['independent_global_only']
    equal_arrays = g['average_precision'] == ig['average_precision'] and g['first_match_rank'] == ig['first_match_rank']
    assert equal_arrays == (dataset != 'RGBNT100')
    result_rows.append(dict(dataset=dataset, objective=objective, selected_epoch=row['selected_epoch'],
        queries=nq, gallery=ng, query_identities=ni, own_global_score_arrays_equal_independent=equal_arrays,
        metrics={k:v['metrics'] for k,v in row['scores'].items()},
        own_global_to_fused={k:v for k,v in own.items() if k not in ['query_changes','identity_changes']},
        independent_to_own_global=row['comparisons']['independent_global_only_to_same_model_global']['delta_metrics'],
        query_ap_improved=up, query_ap_worsened=down, query_ap_unchanged=nq-up-down,
        original_distance_max_difference=row['fused_distance_max_absolute_difference_from_original'],
        diagnostic_means={s:{k:v['mean'] for k,v in vs.items()} for s,vs in row['diagnostic'].items()},
        checkpoint_sha256=row['input_checkpoint_sha256'], original_receipt_sha256=row['original_receipt_sha256'],
        report_path=mapping[original_name]['local_file'], report_sha256=mapping[original_name]['sha256'],
        remote_binary_artifacts=row['artifacts']))
assert counts['evaluation_record_instances'] == 16926
assert old_summary['accepted'] == 5 and old_summary['formal_epochs'] == 250 and old_summary['formal_steps'] == 9839
assert sum(len(r['history']) for r in old_summary['rows']) == 250
assert sum(r['formal_steps'] for r in old_summary['rows']) == 9839
assert all([e['epoch'] for e in r['history']] == list(range(1, 51)) for r in old_summary['rows'])
assert len(old_summary['pairs']) == local_receipts['all_comparisons'] == 12
assert sum(p['phase_progress'] for p in old_summary['pairs']) == local_receipts['all_advances'] == 0
assert local_receipts['primary_comparisons'] == 5 and local_receipts['primary_advances'] == 0
assert len(old_summary['missing']) == 1
helper_paths = [Path('C:/Users/gb/.aris/installed-skills-codex.txt'),
    REPO / '.aris/installed-skills-codex.txt', Path('C:/Users/gb/tools/evidence_check.py'), REPO / 'tools/evidence_check.py']
assert not os.environ.get('ARIS_REPO') and not any(p.exists() for p in helper_paths)
precheck = dict(status='UNRESOLVED', canonical_helper='evidence_check.py', ARIS_REPO=os.environ.get('ARIS_REPO'),
    checked_paths={str(p):p.exists() for p in helper_paths},
    direct_checks_are_separate=True, consequence='Canonical verifier was not run; no canonical PASS claimed.')
(TRACE / 'evidence_precheck.json').write_text(json.dumps(precheck, indent=2)+'\n', encoding='utf-8')
result = dict(status='PASS', verification_type='stdlib_local_bytes_and_serialized_arithmetic_only',
    generated_at=datetime.now().astimezone().isoformat(), source_snapshot_entries_verified=len(source_map),
    sealed_source_entries_verified=424, primary_files_verified=19, primary_payloads_exact=19,
    previous_formal_primary_files_verified=len(old_map), sealed_physical_inputs=75,
    physical_inputs_and_tensor_sha_assurance='Original collector source plus terminal receipt only; no fresh remote or binary replay.',
    counts=dict(counts), rows=result_rows, original_controller_exit=data['original_exit'],
    historical_formal_epochs=250, historical_formal_updates=9839, primary_advances='0/5', all_advances='0/12',
    canonical_evidence_check_status='UNRESOLVED',
    scope='No SSH, packages, NN, fitting, tensor loading, feature-cache experiment, source mutation or Git operation.')
(TRACE / 'direct_checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))

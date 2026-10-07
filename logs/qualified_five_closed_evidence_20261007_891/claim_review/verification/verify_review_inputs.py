"""Read-only CPU verification of received text; writes only this review trace."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

BASE = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
OUT = BASE / 'qualified_five_result_to_claim891'
TRACE = OUT / '.aris/traces/result-to-claim/2026-10-07_five891'
PRIMARY = BASE / 'qualified_five_full_complete889'
SOURCE = BASE / 'qualified_five_audit_sources891'
HASHES = {}


def load(path):
    raw = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def lines(path):
    raw = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]


claim_inputs = load(OUT / 'CLAIMS_AND_INPUTS.json')
precheck = load(OUT / 'EVIDENCE_PRECHECK.json')
file_map = load(PRIMARY / 'FILE_MAP.json')
source_map = load(SOURCE / 'FILE_MAP.json')
receipt = load(PRIMARY / 'LOCAL_RECEIPT_SUMMARY.json')
summary = load(PRIMARY / 'primary_text/0022_SUMMARY.json')
derived = load(BASE / 'qualified_five_training_text_analysis891/SUMMARY.json')
m0 = load(BASE / 'checked_m0_support_analysis889/SUMMARY.json')
direct_byte_checks = []
for original, entry in file_map.items():
    path = Path(entry['local_file'])
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    HASHES[str(path)] = actual
    assert actual == entry['sha256'], original
    assert len(raw) == entry['bytes'], original
    direct_byte_checks.append(original)
for path, expected in precheck['input_hashes'].items():
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == expected, path
    HASHES[path] = actual


def mapped_source(original):
    entry = source_map[original]
    path = SOURCE / entry['local_file']
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == entry['sha256'], original
    assert len(raw) == entry['bytes'], original
    HASHES[str(path)] = actual
    return path


for original in [
    'modeling/trifusion/incremental_role_objectives.py',
    'modeling/trifusion/global_task_role_heads.py',
    'tools/run_incremental_role_objective.py',
    'tools/run_incremental_role_objective_checked.py',
    'tools/queue_incremental_qualified_five.py',
    'tools/report_incremental_qualified_five.py',
    'tools/analyze_correspondence_distances.py',
    'refine-logs/incremental_role_objective_v1/QUALIFIED_FIVE_PLAN_20261007_888.md',
]:
    mapped_source(original)

protocols = {}
for dataset in ['RGBNT201', 'MSVR310', 'RGBNT100']:
    original = 'logs/training_feature_scale_protocols_20261002/' + dataset + '.json'
    protocol = load(mapped_source(original))
    expected_protocol_sha = next(row['initializer']['protocol_sha256'] for row in summary['rows'] if row['dataset'] == dataset)
    assert source_map[original]['sha256'] == expected_protocol_sha
    env_key = 'scene' if dataset == 'MSVR310' else 'camera'
    by_name = {}
    for records in protocol['records'].values():
        for record in records:
            key = Path(record['paths'][0]).name
            if key in by_name:
                assert by_name[key] == record[env_key]
            by_name[key] = record[env_key]
    protocols[dataset] = by_name

rows = []
for row in summary['rows']:
    dataset, objective = row['dataset'], row['variant']
    run = 'trained-model/' + Path(row['run_dir']).name + '/'
    data = {}
    for name in ['training.json', 'training_steps.jsonl', 'training_batch_order.jsonl', 'official_metrics.json']:
        path = Path(file_map[run + name]['local_file'])
        data[name] = lines(path) if name.endswith('.jsonl') else load(path)
    training = data['training.json']
    steps = data['training_steps.jsonl']
    batches = data['training_batch_order.jsonl']
    official = data['official_metrics.json']
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert official['status'] == 'COMPLETE'
    assert training['seed'] == official['seed'] == 42
    assert training['epochs'] == official['training_epochs'] == 50
    assert training['condition']['deployment'] == 'L2_1536'
    assert training['initializer']['added_model_parameters'] == 0
    assert row['initializer'] == training['initializer']
    assert row['history'] == training['history']
    assert [x['epoch'] for x in training['history']] == list(range(1, 51))
    assert len(steps) == len(batches) == sum(x['steps'] for x in training['history']) == row['formal_steps']
    assert [(x['epoch'], x['batch']) for x in steps] == [(x['epoch'], x['batch']) for x in batches]
    best = max(training['history'], key=lambda x: (x['official_fused']['mAP'], x['epoch']))
    assert best['epoch'] == official['selected_epoch'] == row['best_epoch']
    for metric in row['metrics']:
        assert abs(row['metrics'][metric] - best['official_fused'][metric]) < 1e-5
        assert row['metrics'][metric] == official['metrics'][metric]
    receipt_row = next(x for x in receipt['rows'] if (x['dataset'], x['objective']) == (dataset, objective))
    assert receipt_row['last_epoch_metrics'] == training['history'][-1]['official_fused']
    assert receipt_row['metrics'] == official['metrics']
    derived_row = next(x for x in derived['rows'] if (x['dataset'], x['objective']) == (dataset, objective))
    assert derived_row['steps'] == len(steps)
    positive_loss_steps = sum(x['incremental_loss'] > 0 for x in steps)
    assert positive_loss_steps == derived_row['positive_incremental_loss_steps']
    measured = dict(dataset=dataset, objective=objective, steps=len(steps), epochs=50,
                    positive_incremental_loss_steps=positive_loss_steps,
                    best_epoch=row['best_epoch'], metrics=row['metrics'],
                    last_epoch_metrics=training['history'][-1]['official_fused'],
                    best_to_last_map_drop=receipt_row['best_to_last_map_drop'],
                    parameter_count=training['initializer']['trainable_parameters'],
                    parameter_tensors=training['initializer']['trainable_parameter_tensors'],
                    raw_semantic_batch_byte_equality_reported=row['actual_training_batch_order_equal'])
    for point, epoch in [('best', row['best_epoch']), ('epoch50', 50)]:
        chosen = [x for x in steps if x['epoch'] == epoch]
        result = dict(epoch=epoch, steps=len(chosen),
                      mean_incremental_loss=statistics.mean(x['incremental_loss'] for x in chosen),
                      mean_scaled_correction_global_ratio=statistics.mean(x['actual_scaled_correction_global_ratio_mean'] for x in chosen))
        derived_point = derived_row['at_best_epoch_training' if point == 'best' else 'at_epoch50_training']
        for key in result:
            assert abs(result[key] - derived_point[key]) < 1e-12
        measured[point + '_training'] = result
    if objective == 'repair_keep':
        names = ['eligible_queries', 'multiple_positive_queries', 'legal_positive_pairs', 'legal_triplets',
                 'repair_triplets', 'keep_triplets', 'repair_active_triplets', 'keep_active_triplets']
        totals = {key: sum(x[key] for x in steps) for key in names}
        assert totals == derived_row['support_totals']
        independently_counted = collections.Counter()
        for batch, step in zip(batches, steps):
            labels = batch['labels']
            envs = [protocols[dataset][Path(path).name] for path in batch['paths']]
            by_identity = collections.Counter(labels)
            by_identity_environment = collections.Counter(zip(labels, envs))
            positive_counts = [by_identity[label] - by_identity_environment[(label, env)] for label, env in zip(labels, envs)]
            counts = dict(eligible_queries=sum(p > 0 for p in positive_counts),
                          multiple_positive_queries=sum(p >= 2 for p in positive_counts),
                          legal_positive_pairs=sum(positive_counts),
                          legal_triplets=sum(p * (len(labels) - by_identity[label]) for p, label in zip(positive_counts, labels)))
            for key, value in counts.items():
                assert value == step[key], (dataset, step['epoch'], step['batch'], key)
            assert step['repair_triplets'] + step['keep_triplets'] == step['legal_triplets']
            independently_counted.update(counts)
        measured['support_totals'] = totals
        measured['independently_counted_static_support'] = dict(independently_counted)
        for cell in ['repair', 'keep']:
            for suffix, field in [('supported_steps', cell + '_triplets'), ('nonzero_loss_steps', cell + '_loss')]:
                count = sum(x[field] > 0 for x in steps)
                assert count == derived_row[cell + '_' + suffix]
                measured[cell + '_' + suffix] = count
        exposures = sum(len(x['labels']) for x in batches)
        assert exposures == derived_row['query_exposures']
        measured['query_exposures'] = exposures
        measured['eligible_query_exposure_ratio'] = totals['eligible_queries'] / exposures
        measured['multiple_positive_among_eligible_ratio'] = totals['multiple_positive_queries'] / totals['eligible_queries']
    rows.append(measured)

pairs = []
for pair in summary['pairs']:
    diagnostic = pair['paired_diagnosis']
    queries = diagnostic['query_changes']
    changes = [q['candidate_ap'] - q['control_ap'] for q in queries]
    repairs = sum(q['control_first_rank'] > 1 and q['candidate_first_rank'] == 1 for q in queries)
    harms = sum(q['control_first_rank'] == 1 and q['candidate_first_rank'] > 1 for q in queries)
    assert repairs == diagnostic['rank1_repairs'] and harms == diagnostic['rank1_new_errors']
    assert sum(x > 1e-8 for x in changes) == diagnostic['query_ap_improved']
    assert sum(x < -1e-8 for x in changes) == diagnostic['query_ap_worsened']
    assert abs(statistics.mean(changes) * 100 - diagnostic['delta_metrics']['mAP']) < 1e-10
    for label, role in [(pair['control'], 'control'), (pair['objective'], 'candidate')]:
        assert abs(statistics.mean(q[role + '_ap'] for q in queries) * 100 - diagnostic['metrics'][label]['mAP']) < 1e-10
        for rank in [1, 5, 10]:
            actual = sum(q[role + '_first_rank'] <= rank for q in queries) * 100 / len(queries)
            assert abs(actual - diagnostic['metrics'][label]['Rank-' + str(rank)]) < 1e-10
    groups = collections.defaultdict(list)
    for q, delta in zip(queries, changes):
        groups[q['identity']].append(delta * 100)
    for identity in diagnostic['identity_changes']:
        assert identity['queries'] == len(groups[identity['identity']])
        assert abs(identity['mean_delta_ap_points'] - statistics.mean(groups[identity['identity']])) < 1e-10
    macro = statistics.mean(statistics.mean(values) for values in groups.values())
    assert abs(macro - diagnostic['identity_macro_mean_delta_ap_points']) < 1e-10
    assert not pair['phase_progress']
    assert pair['phase_progress'] == (diagnostic['delta_metrics']['mAP'] >= 0.5 and diagnostic['delta_metrics']['Rank-1'] >= 0)
    pairs.append(dict(dataset=pair['dataset'], objective=pair['objective'], control=pair['control'],
                      queries=len(queries), identities=len(groups), delta=diagnostic['delta_metrics'],
                      rank1_repairs=repairs, rank1_new_errors=harms,
                      query_ap_improved=diagnostic['query_ap_improved'], query_ap_worsened=diagnostic['query_ap_worsened'],
                      identity_ap_improved=diagnostic['identity_ap_improved'], identity_ap_worsened=diagnostic['identity_ap_worsened'],
                      identity_macro_mean_delta_ap_points=macro,
                      identity_bootstrap_95_percentile_interval=diagnostic['identity_bootstrap_95_percentile_interval'],
                      phase_progress=pair['phase_progress']))

assert len(rows) == 5 and len(pairs) == 12
assert sum(row['steps'] for row in rows) == summary['formal_steps'] == 9839
assert summary['formal_epochs'] == 250
assert len(summary['missing']) == 1
assert summary['missing'][0]['dataset'] == 'RGBNT100' and summary['missing'][0]['objective'] == 'repair_keep'
value = dict(status='REVIEW_TEXT_CROSSCHECK_PASS', scope='Read-only stdlib CPU recount; no tensor/model evaluation, no external access.',
             canonical_evidence_check_status=precheck['canonical_evidence_check_status'],
             direct_primary_file_byte_checks=len(direct_byte_checks), precheck_listed_input_byte_checks=len(precheck['input_hashes']),
             rows=rows, pairs=pairs, missing=summary['missing'], input_sha256=HASHES,
             bootstrap_recomputed=False,
             historical_control_batch_bytes_independently_available=False,
             historical_control_batch_claim_basis='Frozen accepted_row() byte equality assertion and received accepted rows; current candidate pair batch files directly hashed. Integrity review separate.',
             limitations=['Recounting serialized per-query ranks/AP is not rerunning the ranker on raw distance tensors.',
                          'Recorded bootstrap intervals inspected with generation source; no training-seed distribution exists.',
                          'Full-run saved losses and correction norms are not full-run isolated per-role gradients.'])
(TRACE / 'independent_text_crosscheck.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
(TRACE / 'input_sha256.json').write_text(json.dumps(HASHES, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: value[k] for k in ['status', 'direct_primary_file_byte_checks', 'precheck_listed_input_byte_checks', 'missing']}))
for row in rows:
    print(json.dumps(row))

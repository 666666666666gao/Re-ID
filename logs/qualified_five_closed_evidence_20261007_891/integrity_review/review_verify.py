"""Independent, standard-library-only audit of supplied immutable text artifacts."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

BASE = Path(r'C:\Users\gb\.codex_tmp\independent_evidence_draft')
OUT = BASE / 'qualified_five_integrity_audit891'
SRC = BASE / 'qualified_five_audit_sources891'
FULL = BASE / 'qualified_five_full_complete889'
REMOTE_ROOT = '/data/gaob/Re-ID/Trifusion/'
HASHES = {}


def data(path):
    raw = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def read(path):
    return json.loads(data(path).decode('utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def mapped(root, mapping, original):
    return root / mapping[original]['local_file']


def finite(value):
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


sm = read(SRC / 'FILE_MAP.json')
fm = read(FULL / 'FILE_MAP.json')
for root, mapping in [(SRC, sm), (FULL, fm)]:
    for original, entry in mapping.items():
        raw = data(mapped(root, mapping, original))
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'], original
        assert len(raw) == entry['bytes'], original
listed = read(OUT / 'AUDIT_INPUT_PATHS.json')
assert len(listed) == 36
assert all(hashlib.sha256(data(Path(item['local_path']))).hexdigest() == item['sha256'] for item in listed)
remote = read(FULL / 'REMOTE.json')
for original, entry in remote['files'].items():
    assert entry['text'].encode('utf-8') == data(mapped(FULL, fm, original)), original
    assert entry['sha256'] == fm[original]['sha256'], original

# Preserve exact primary M0 texts from the provided transport receipts for citations.
m0maps = {}
m0transcripts = {}
for group in ['checked_m0_matrix887', 'checked_m0_terminal_failure887']:
    transcript = read(BASE / group / 'REMOTE.json')
    m0transcripts[group] = transcript
    destination = OUT / 'm0_primary' / group
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for i, (original, entry) in enumerate(transcript['files'].items()):
        raw = entry['text'].encode('utf-8')
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'], original
        local = destination / f'{i:04d}_{Path(original).name}'
        local.write_bytes(raw)
        mapping[original] = {'local_file': str(local), 'sha256': entry['sha256'], 'bytes': len(raw)}
        HASHES[str(local)] = entry['sha256']
    write(destination / 'FILE_MAP.json', mapping)
    m0maps[group] = mapping

protocols = {}
protocol_checks = []
for dataset in ['RGBNT201', 'MSVR310', 'RGBNT100']:
    original = f'logs/training_feature_scale_protocols_20261002/{dataset}.json'
    protocol = read(mapped(SRC, sm, original))
    protocols[dataset] = protocol
    records = protocol['records']
    assert {k: len(v) for k, v in records.items()} == protocol['counts']
    env = protocol['environment_key']
    assert env == ('scene' if dataset == 'MSVR310' else 'camera')
    for split, rows in records.items():
        assert [row['index'] for row in rows] == list(range(len(rows)))
        assert len({tuple(row['paths']) for row in rows}) == len(rows)
        for row in rows:
            name = Path(row['paths'][0]).name
            if dataset == 'RGBNT201':
                identity, camera = re.match(r'(\d+)_cam(\d+)_', name).groups()
                assert (row['identity'], row['camera']) == (int(identity), int(camera) - 1)
            elif dataset == 'RGBNT100':
                identity, camera = re.match(r'([-\d]+)_c([-\d]+)_', name).groups()
                assert (row['identity'], row['camera']) == (int(identity), int(camera) - 1)
            else:
                identity, scene, camera = re.match(r'(\d+)_s(\d+)_v(\d+)_', name).groups()
                assert (row['identity'], row['scene'], row['camera']) == (int(identity), int(scene), int(camera))
            if split == 'train':
                assert row['label'] == protocol['train_label_map'][str(row['identity'])]
            else:
                assert row['label'] is None
    training_ids = {row['identity'] for row in records['train']}
    query_ids = {row['identity'] for row in records['query']}
    gallery_ids = {row['identity'] for row in records['gallery']}
    assert not training_ids & (query_ids | gallery_ids)
    gallery_id_counts = Counter(row['identity'] for row in records['gallery'])
    gallery_cell_counts = Counter((row['identity'], row[env]) for row in records['gallery'])
    positives = []
    for q, audit in zip(records['query'], protocol['query_rows'], strict=True):
        excluded = gallery_cell_counts[q['identity'], q[env]]
        positive = gallery_id_counts[q['identity']] - excluded
        assert positive > 0
        assert audit == dict(index=q['index'], identity=q['identity'], valid_positive_count=positive,
                             excluded_same_identity_same_environment=excluded)
        positives.append(positive)
    protocol_checks.append(dict(dataset=dataset, counts=protocol['counts'], train_identities=len(training_ids),
        query_identities=len(query_ids), gallery_identities=len(gallery_ids),
        gallery_only_identities=sorted(gallery_ids-query_ids),
        gallery_only_records=sum(row['identity'] not in query_ids for row in records['gallery']),
        all_filename_identity_and_environment_parity=True, train_test_identity_disjoint=True,
        full_legal_query_count=len(positives), legal_positive_count_range=[min(positives), max(positives)]))

def primary(name):
    return next((original, mapped(FULL, fm, original)) for original in fm if original.endswith('/' + name))

campaign_original, campaign_path = primary('campaign.json')
campaign = read(campaign_path)
_, manifest_path = primary('manifest.json')
manifest = read(manifest_path)
_, matrix_path = primary('accepted_matrix.json')
matrix = read(matrix_path)
_, summary_path = primary('SUMMARY.json')
summary = read(summary_path)
controls = read(SRC / '0421_INPUT_SEAL.json')
source_scope = read(SRC / '0420_QUALIFIED_FIVE_SOURCE_SCOPE.json')
assert manifest['source_sha256'] == source_scope['source_sha256']
assert all(sm[name]['sha256'] == digest for name, digest in manifest['source_sha256'].items())
assert len(manifest['source_sha256']) == 420
assert manifest['control_seal_sha256'] == HASHES[str(SRC / '0421_INPUT_SEAL.json')]
assert campaign['status'] == 'COMPLETE' and campaign['report_invocations'] == 1 and campaign['report_exit_code'] == 0
jobs = {('RGBNT201', 'md_batch_ratio'), ('RGBNT201', 'repair_keep'), ('MSVR310', 'md_batch_ratio'),
        ('MSVR310', 'repair_keep'), ('RGBNT100', 'md_batch_ratio')}
assert len(campaign['jobs']) == 5
assert {(row['dataset'], row['objective']) for row in campaign['jobs']} == jobs
assert {(row['dataset'], row['variant']) for row in matrix['rows']} == jobs
assert {(row['dataset'], row['variant']) for row in summary['rows']} == jobs
assert matrix['accepted'] == matrix['expected'] == summary['accepted'] == 5
observer = read(BASE / 'qualified_five_full_observer888' / 'COMPLETE.json')
_, exit_path = primary('EXIT.json')
assert read(exit_path) == observer['original_exit'] == remote['actual_parent_exit']
assert observer['original_exit']['exit_code'] == 0

rows = []
old_state_differences = []
for row in matrix['rows']:
    dataset, objective = row['dataset'], row['variant']
    run = row['run_dir'].removeprefix(REMOTE_ROOT)
    training_path = mapped(FULL, fm, run + '/training.json')
    official_path = mapped(FULL, fm, run + '/official_metrics.json')
    order_path = mapped(FULL, fm, run + '/training_batch_order.jsonl')
    step_path = mapped(FULL, fm, run + '/training_steps.jsonl')
    training, official = read(training_path), read(official_path)
    steps = [json.loads(line) for line in data(step_path).splitlines()]
    orders = [json.loads(line) for line in data(order_path).splitlines()]
    assert finite(steps) and finite(training) and finite(official)
    history = training['history']
    assert [h['epoch'] for h in history] == list(range(1, 51))
    assert len(steps) == len(orders) == sum(h['steps'] for h in history)
    assert [(s['epoch'], s['batch']) for s in steps] == [(s['epoch'], s['batch']) for s in orders]
    expected_indices = [(h['epoch'], i) for h in history for i in range(h['steps'])]
    assert [(s['epoch'], s['batch']) for s in steps] == expected_indices
    for h in history:
        values = [s['loss'] for s in steps if s['epoch'] == h['epoch']]
        assert abs(math.fsum(values) / len(values) - h['mean_loss']) < 1e-10
    assert max(abs(s['loss'] - s['global_loss'] - s['fused_role_loss'] - s['incremental_loss']) for s in steps) < 1e-5
    old = next(r for r in controls['rows'] if r['dataset'] == dataset and r['variant'] == 'semantic')
    old_order = old['run_dir'] + '/training_batch_order.jsonl'
    assert HASHES[str(order_path)] == controls['artifact_sha256'][old_order]
    for key, value in old['initializer'].items():
        if key not in ['architecture', 'entry_sha256', 'scope']:
            assert training['initializer'][key] == value, (dataset, objective, key)
    binding = training['initializer']
    assert binding == row['initializer']
    initializer_original = f"logs/incremental_role_objective_m0_v1_20261007_886/initialization/{dataset}_{objective}.json"
    init_entry = m0maps['checked_m0_matrix887'][initializer_original]
    initializer = read(Path(init_entry['local_file']))
    assert initializer['binding'] == binding
    assert manifest['initialization_sha256'][REMOTE_ROOT + initializer_original] == init_entry['sha256']
    assert official['seed'] == training['seed'] == 42 and official['training_epochs'] == 50
    assert official['condition'] == training['condition']
    best = sorted(history, key=lambda h: (h['official_fused']['mAP'], h['epoch']))[-1]
    assert official['selected_epoch'] == training['best_epoch'] == row['best_epoch'] == best['epoch']
    assert max(abs(official['metrics'][k] - best['official_fused'][k]) for k in official['metrics']) < 1e-5
    assert official['metrics'] == row['metrics']
    assert official['checkpoint_sha256'] == row['checkpoint_sha256']
    assert official['distance_sha256'] == row['distance_sha256']
    assert HASHES[str(official_path)] == row['receipt_sha256']
    assert official['protocol_sha256'] == binding['protocol_sha256']
    assert official['independent_upstream_metrics_equal'] and not official['reranking']
    known = {Path(r['paths'][0]).name: r for r in protocols[dataset]['records']['train']}
    exposures = 0
    for order in orders:
        assert len(order['paths']) == len(order['labels']) == len(order['cameras']) == binding['batch_size']
        for name, label, camera in zip(order['paths'], order['labels'], order['cameras'], strict=True):
            r = known[Path(name).name]
            assert (label, camera) == (r['label'], r['camera'])
            exposures += 1
    job = next(j for j in campaign['jobs'] if (j['dataset'], j['objective']) == (dataset, objective))
    assert job['status'] == 'COMPLETE' and job['exit_code'] == 0
    assert [step['mode'] for step in job['steps']] == ['train', 'evaluate']
    assert all(step['exit_code'] == 0 and step['status'] == 'COMPLETE' for step in job['steps'])
    assert all(step['command'][2].endswith('/tools/run_incremental_role_objective_checked.py') for step in job['steps'])
    train_step, eval_step = job['steps']
    parse = datetime.fromisoformat
    wall = (parse(training['completed_at']) - parse(training['started_at'])).total_seconds()
    history_seconds = math.fsum(h['seconds'] for h in history)
    cli_wall = (parse(train_step['completed_at']) - parse(train_step['started_at'])).total_seconds()
    assert parse(eval_step['started_at']) >= parse(train_step['completed_at'])
    assert wall > history_seconds and cli_wall >= wall
    report_row = next(r for r in summary['rows'] if (r['dataset'], r['variant']) == (dataset, objective))
    assert report_row['history'] == history
    assert report_row['training_and_epoch_eval_seconds'] == wall
    rows.append(dict(dataset=dataset, objective=objective, formal_steps=len(steps), logged_exposures=exposures,
        best_epoch=best['epoch'], metrics=official['metrics'], last_epoch_metrics=history[-1]['official_fused'],
        training_receipt_wall_seconds=wall, train_loop_seconds=history_seconds, train_process_seconds=cli_wall,
        strict_evaluation_seconds=official['final_evaluation_seconds'],
        best_to_last_map_drop=best['official_fused']['mAP']-history[-1]['official_fused']['mAP'],
        raw_semantic_order_sha256=HASHES[str(order_path)],
        unchanged_initializer=True, full_order_matches_sealed_raw_semantic=True,
        training_file=str(training_path), official_file=str(official_path), step_file=str(step_path), order_file=str(order_path)))
assert sum(r['formal_steps'] for r in rows) == summary['formal_steps'] == 9839
assert sum(len(row['history']) for row in summary['rows']) == summary['formal_epochs'] == 250

pairs = []
expected_pairs = {(d, o, c) for d, o in jobs for c in ['raw_semantic', 'raw_global_only']}
expected_pairs |= {(d, 'repair_keep', 'md_batch_ratio') for d in ['RGBNT201', 'MSVR310']}
assert len(summary['pairs']) == 12
assert {(p['dataset'], p['objective'], p['control']) for p in summary['pairs']} == expected_pairs
for pair in summary['pairs']:
    diag = pair['paired_diagnosis']
    changes = diag['query_changes']
    queries = protocols[pair['dataset']]['records']['query']
    assert len(changes) == len(queries)
    assert all((r['query_index'], r['identity']) == (q['index'], q['identity']) for r, q in zip(changes, queries, strict=True))
    assert all(0 <= r[side + '_ap'] <= 1 and r[side + '_first_rank'] >= 1 for r in changes for side in ['control', 'candidate'])
    for side, label in [('control', pair['control']), ('candidate', pair['objective'])]:
        recomputed = {'mAP': math.fsum(r[side + '_ap'] for r in changes) * 100 / len(changes)}
        recomputed.update({f'Rank-{k}': sum(r[side + '_first_rank'] <= k for r in changes) * 100 / len(changes) for k in [1, 5, 10]})
        assert all(abs(recomputed[k] - diag['metrics'][label][k]) < 1e-10 for k in recomputed)
    deltas = [r['candidate_ap'] - r['control_ap'] for r in changes]
    repairs = sum(r['control_first_rank'] != 1 and r['candidate_first_rank'] == 1 for r in changes)
    harms = sum(r['control_first_rank'] == 1 and r['candidate_first_rank'] != 1 for r in changes)
    assert repairs == diag['rank1_repairs'] and harms == diag['rank1_new_errors']
    assert abs((repairs - harms) * 100 / len(changes) - diag['delta_metrics']['Rank-1']) < 1e-10
    assert sum(d > 1e-8 for d in deltas) == diag['query_ap_improved']
    assert sum(d < -1e-8 for d in deltas) == diag['query_ap_worsened']
    grouped = defaultdict(list)
    for r, delta in zip(changes, deltas):
        grouped[r['identity']].append(delta * 100)
    identity_rows = diag['identity_changes']
    assert len(identity_rows) == len(grouped)
    for r in identity_rows:
        values = grouped[r['identity']]
        assert len(values) == r['queries']
        assert abs(math.fsum(values) / len(values) - r['mean_delta_ap_points']) < 1e-10
    assert sum(r['mean_delta_ap_points'] > 1e-6 for r in identity_rows) == diag['identity_ap_improved']
    assert sum(r['mean_delta_ap_points'] < -1e-6 for r in identity_rows) == diag['identity_ap_worsened']
    assert abs(math.fsum(r['mean_delta_ap_points'] for r in identity_rows) / len(identity_rows) - diag['identity_macro_mean_delta_ap_points']) < 1e-10
    assert all(abs(diag['metrics'][pair['objective']][k] - diag['metrics'][pair['control']][k] - v) < 1e-10 for k, v in diag['delta_metrics'].items())
    expected_progress = diag['delta_metrics']['mAP'] >= .5 and diag['delta_metrics']['Rank-1'] >= 0
    assert pair['phase_progress'] == expected_progress
    pairs.append(dict(dataset=pair['dataset'], objective=pair['objective'], control=pair['control'],
        queries=len(changes), identities=len(grouped), delta_metrics=diag['delta_metrics'], repairs=repairs, harms=harms,
        phase_progress=expected_progress, identity_bootstrap_interval=diag['identity_bootstrap_95_percentile_interval']))

m0map = m0maps['checked_m0_terminal_failure887']
m0campaign = read(Path(m0map['logs/incremental_role_objective_m0_v1_20261007_886/campaign.json']['local_file']))
assert len(m0campaign['jobs']) == 6 and sum('acceptance' in j for j in m0campaign['jobs']) == 5
m0rows = []
for job in m0campaign['jobs']:
    dataset, objective = job['dataset'], job['objective']
    run = f'trained-model/incremental_role_objective_m0_v1_20261007_886_m0_{objective}_{dataset}'
    training = read(Path(m0map[run + '/training.json']['local_file']))
    steps = [json.loads(line) for line in data(Path(m0map[run + '/training_steps.jsonl']['local_file'])).splitlines()]
    assert training['status'] == 'M0_PASS' and len(steps) == 8
    assert training['production_m0_diagnostics']['effective_optimizer_updates'] == 8
    assert all(v == 8 for v in training['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    assert training['m0']['reload_max_abs_difference'] <= 1e-5
    assert all(s['incremental_isolated_shared_global_gradient_absent'] and s['incremental_isolated_vjp_autocast_enabled'] is False for s in steps)
    assert all(not v for s in steps for v in s['incremental_isolated_query_key_unused'].values())
    sums = {name: math.fsum(s['incremental_isolated_query_key_gradient_norms'][name] for s in steps)
            for name in steps[0]['incremental_isolated_query_key_gradient_norms']}
    zero = [name for name, total in sums.items() if total == 0]
    if (dataset, objective) == ('RGBNT100', 'repair_keep'):
        assert set(zero) == {'evidence_model.roles.query_projections.2.weight', 'evidence_model.roles.key_projections.2.weight'}
        assert 'acceptance' not in job
    else:
        assert not zero and 'acceptance' in job
        assert all(abs(job['acceptance']['query_key_gradient_norm_sums'][k] - v) < 1e-15 for k, v in sums.items())
    m0rows.append(dict(dataset=dataset, objective=objective, production_updates=8, accepted='acceptance' in job,
        zero_auxiliary_gradient_parameters=zero, query_key_unused=False, reload_max_abs_difference=training['m0']['reload_max_abs_difference']))
assert m0transcripts['checked_m0_terminal_failure887']['exit']['exit_code'] == 1
assert 'AssertionError: evidence_model.roles.key_projections.2.weight' in m0transcripts['checked_m0_terminal_failure887']['controller_log']

verification = dict(status='PASS_DETERMINISTIC_TEXT_CHECKS', generated_at=datetime.now().astimezone().isoformat(),
    byte_verification=dict(listed_inputs=36, source_snapshot_entries=423, full_primary_entries=45,
        full_remote_text_exact_entries=45, campaign_manifest_sealed_entries=420,
        m0_matrix_embedded_text_entries=len(m0maps['checked_m0_matrix887']),
        m0_failure_embedded_text_entries=len(m0maps['checked_m0_terminal_failure887'])),
    protocols=protocol_checks, rows=rows, pairs=pairs, m0=m0rows,
    report_invocations=campaign['report_invocations'], report_exit_code=campaign['report_exit_code'],
    original_full_exit=observer['original_exit'], original_m0_exit=m0transcripts['checked_m0_terminal_failure887']['exit'],
    limits=['No SSH, GPU, neural forward, training or package installation performed.',
            'Remote checkpoint/distance physical hashes rely on the supplied original collection transcript and inspected collector code.',
            'Raw binary distances and image inventory were not independently opened; query-table arithmetic is not a fresh tensor reranking.',
            'NumPy bootstrap random draws were not rerun; identity grouping and macro means were recomputed.',
            'Byte verification does not establish semantic assurance or unseen-test/generalization evidence.'])
write(OUT / 'DETERMINISTIC_CHECKS.json', verification)
write(OUT / 'AUDITED_INPUT_HASHES.json', HASHES)
print(json.dumps({k:v for k,v in verification.items() if k not in ['pairs','m0']}, indent=2))

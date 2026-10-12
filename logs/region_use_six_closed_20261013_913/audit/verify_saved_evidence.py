"""Independent stdlib-only, read-only input audit; writes CHECKS.json here only."""
import ast
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
BASE = Path(r'D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903')
REMOTE = '/data/gaob/Re-ID/Trifusion/'
HASHES = {}
COUNTS = Counter()
FAILURES = []
DETAILS = {}


def digest(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = value
    return value


def read(path):
    path = Path(path)
    digest(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check(section, label, condition, actual=None, expected=None):
    COUNTS[section] += 1
    if not condition:
        FAILURES.append(dict(section=section, label=label, actual=actual, expected=expected))


def equal(section, label, actual, expected):
    check(section, label, actual == expected, actual, expected)


def near(section, label, actual, expected, tolerance=1e-10):
    check(section, label, math.isfinite(actual) and math.isfinite(expected)
          and abs(actual - expected) < tolerance, actual, expected)


def metrics(rows, prefix):
    count = len(rows)
    return {'mAP': math.fsum(r[prefix + '_ap'] for r in rows) * 100 / count,
            **{f'Rank-{rank}': sum(r[prefix + '_first_rank'] <= rank for r in rows) * 100 / count
               for rank in (1, 5, 10)}}


def group_values(rows, before, after):
    groups = {k: [] for k in ('repaired', 'new_error', 'first_still_correct', 'first_still_wrong')}
    for r in rows:
        old = r[before + '_first_rank'] == 1
        new = r[after + '_first_rank'] == 1
        name = ('first_still_correct' if new else 'new_error') if old else ('repaired' if new else 'first_still_wrong')
        groups[name].append((r[after + '_ap'] - r[before + '_ap']) * 100)
    return groups


def verify_groups(section, label, rows, before, after, supplied):
    groups = group_values(rows, before, after)
    for name, values in groups.items():
        equal(section, label + '/' + name + '/count', supplied[name]['queries'], len(values))
        near(section, label + '/' + name + '/AP', supplied[name]['contribution_to_total_delta_map_points'],
             math.fsum(values) / len(rows))
    return groups


request = HERE / 'REQUEST.md'
digest(request)
for line in request.read_text(encoding='utf-8-sig').splitlines():
    if line.startswith('- D:\\'):
        digest(Path(line[2:]))

manifests = {}
for suffix, expected in (('908', 30), ('911', 11), ('912', 18)):
    folder = BASE / ('results' + suffix)
    manifest = read(folder / ('INTAKE_MANIFEST_' + suffix + '.json'))
    manifests[suffix] = manifest
    equal('manifests', suffix + '/count', len(manifest['files']), expected)
    for name, expected_hash in manifest['files'].items():
        path = folder / name
        check('manifests', name + '/within-intake', path.resolve().is_relative_to(folder.resolve()))
        equal('manifests', suffix + '/' + name, digest(path), expected_hash)
DETAILS['manifest_files'] = {k: len(v['files']) for k, v in manifests.items()}

meta = read(BASE / 'SIXTH_RESULTS_912.meta.json')
archive = BASE / 'SIXTH_RESULTS_912.zip'
equal('archive', 'bytes', archive.stat().st_size, meta['bytes'])
equal('archive', 'hash', digest(archive), meta['sha256'])
equal('archive', 'file-count', meta['files'], 18)
with zipfile.ZipFile(archive) as z:
    equal('archive', 'members', set(z.namelist()), set(manifests['912']['files']) | {'INTAKE_MANIFEST_912.json'})
    equal('archive', 'manifest', json.loads(z.read('INTAKE_MANIFEST_912.json')), manifests['912'])
    for name, expected_hash in manifests['912']['files'].items():
        equal('archive', name, hashlib.sha256(z.read(name)).hexdigest(), expected_hash)

aggregate = read(BASE / 'ALL_SIX_RESULTS_912.json')
equal('aggregate', 'status', aggregate['status'], 'SIX_ORIGINAL_FULL50_FIRST_STRICT_RESULTS_COLLECTED')
equal('aggregate', 'rows', len(aggregate['rows']), 6)
equal('aggregate', 'contrasts', len(aggregate['ambiguity_minus_uniform']), 3)
for path, expected_hash in aggregate['inputs'].items():
    equal('aggregate', 'input/' + path, digest(Path(path)), expected_hash)

summaries = {}
training_records = {}
receipts = {}
row_results = []
max_cmc_difference = 0.0
expected_queries = {'RGBNT201': 836, 'MSVR310': 591, 'RGBNT100': 1715}
expected_gallery = {'RGBNT201': 836, 'MSVR310': 1055, 'RGBNT100': 8575}
for index, row in enumerate(aggregate['rows']):
    dataset, condition = row['dataset'], row['comparison']
    key = (dataset, condition)
    suffix = '908' if dataset != 'RGBNT100' else ('911' if condition == 'uniform' else '912')
    folder = BASE / ('results' + suffix)
    branch = 'region_use_remaining_endpoint_20261013_912' if suffix == '912' else 'region_use_endpoint_diagnosis_20261012_907'
    summary = read(folder / 'logs' / branch / (dataset + '_' + condition) / 'SUMMARY.json')
    summaries[key] = summary
    run_relative = summary['candidate']['run_dir'].removeprefix(REMOTE)
    run = folder / run_relative
    training = read(run / 'training.json')
    receipt = read(run / 'official_metrics.json')
    training_records[key], receipts[key] = training, receipt
    steps_path, order_path = run / 'training_steps.jsonl', run / 'training_batch_order.jsonl'
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    batches = [json.loads(line) for line in order_path.read_text().splitlines()]
    label = dataset + '/' + condition
    equal('six_results', label + '/summary-status', summary['status'], 'ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE')
    equal('six_results', label + '/training-status', training['status'], 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE')
    equal('six_results', label + '/receipt-status', receipt['status'], 'COMPLETE')
    equal('six_results', label + '/identity', (summary['dataset'], summary['comparison']), key)
    equal('six_results', label + '/seed', (training['seed'], receipt['seed']), (42, 42))
    equal('six_results', label + '/epochs', (training['epochs'], receipt['training_epochs']), (50, 50))
    equal('six_results', label + '/history-sequence', [r['epoch'] for r in training['history']], list(range(1, 51)))
    equal('six_results', label + '/summary-history', summary['history'], training['history'])
    equal('six_results', label + '/condition', receipt['condition'], training['condition'])
    equal('six_results', label + '/comparison', training['condition']['region_use_comparison'], condition)
    equal('six_results', label + '/protocol-pin', receipt['protocol_sha256'], training['initializer']['protocol_sha256'])
    equal('six_results', label + '/no-rerank', receipt['reranking'], False)
    equal('six_results', label + '/upstream-equal-recorded', receipt['independent_upstream_metrics_equal'], True)
    selected = max(training['history'], key=lambda r: (r['official_fused']['mAP'], r['epoch']))
    equal('six_results', label + '/best-epoch', [row['best_epoch'], summary['candidate']['best_epoch'], training['best_epoch'], receipt['selected_epoch']], [selected['epoch']] * 4)
    for metric, value in receipt['metrics'].items():
        near('six_results', label + '/strict-vs-selected/' + metric, value, selected['official_fused'][metric], 1e-5)
    for name in ('metrics', 'history', 'formal_recorded_steps', 'candidate', 'best_to_last_map_drop'):
        if name in row:
            equal('aggregate', label + '/' + name, row[name], summary[name])
    equal('six_results', label + '/candidate-metrics', summary['candidate']['metrics'], receipt['metrics'])
    equal('six_results', label + '/summary-metrics', summary['metrics'], receipt['metrics'])
    for name in ('checkpoint_sha256', 'distance_sha256'):
        equal('six_results', label + '/' + name, summary['candidate'][name], receipt[name])
    equal('aggregate', label + '/last-metrics', row['last_metrics'], training['history'][-1]['official_fused'])
    near('aggregate', label + '/drop', row['best_to_last_map_drop'], receipt['metrics']['mAP'] - row['last_metrics']['mAP'])
    count = sum(r['steps'] for r in training['history'])
    equal('training_records', label + '/counts', [len(steps), len(batches), summary['formal_recorded_steps']], [count] * 3)
    pairs = [(r['epoch'], r['batch']) for r in steps]
    equal('training_records', label + '/order-keys', pairs, [(r['epoch'], r['batch']) for r in batches])
    equal('training_records', label + '/sequential-keys', pairs, [(r['epoch'], i) for r in training['history'] for i in range(r['steps'])])
    equal('training_records', label + '/full-batch-size', set(len(r['labels']) for r in batches), {training['initializer']['batch_size']})
    equal('training_records', label + '/aux-weight-mode', set(r['region_use_support']['ambiguity_weighted'] for r in steps), {condition == 'ambiguity'})
    for history, trace in zip(training['history'], summary['auxiliary_traces'], strict=True):
        ep = history['epoch']
        subset = [r for r in steps if r['epoch'] == ep]
        near('training_records', label + f'/epoch{ep}/mean-loss', history['mean_loss'], math.fsum(r['loss'] for r in subset) / len(subset))
        near('training_records', label + f'/epoch{ep}/auxiliary-mean', trace['auxiliary_loss_mean'], math.fsum(r['region_use_loss'] for r in subset) / len(subset))
        equal('training_records', label + f'/epoch{ep}/trace-epoch', trace['epoch'], ep)
        equal('training_records', label + f'/epoch{ep}/eligible', trace['eligible_query_exposures'], sum(r['region_use_support']['eligible_queries'] for r in subset))
        equal('training_records', label + f'/epoch{ep}/triplets', trace['legal_triplets'], sum(r['region_use_support']['legal_triplets'] for r in subset))
        equal('training_records', label + f'/epoch{ep}/exposures', trace['source_query_exposures'], len(subset) * training['initializer']['batch_size'])
    near('training_records', label + '/loop-seconds', summary['training_loop_seconds'], math.fsum(r['seconds'] for r in training['history']))
    near('training_records', label + '/interval', summary['training_and_epoch_evaluation_interval_seconds'], (datetime.fromisoformat(training['completed_at']) - datetime.fromisoformat(training['started_at'])).total_seconds())
    for source, expected_hash in summary['inputs'].items():
        if source.startswith(REMOTE + 'trained-model/'):
            equal('six_results', label + '/input/' + source, digest(folder / source.removeprefix(REMOTE)), expected_hash)
    equal('saved_pairs', label + '/pair-names', [p['control'] for p in summary['pairs']], ['raw_semantic', 'independent_global'])
    candidate_queries = []
    for pair, projected in zip(summary['pairs'], row['controls'], strict=True):
        p_label = label + '/' + pair['control']
        for field in projected:
            equal('aggregate', p_label + '/' + field, projected[field], pair[field])
        queries = pair['query_changes']
        equal('saved_pairs', p_label + '/count', len(queries), expected_queries[dataset])
        equal('saved_pairs', p_label + '/query-indices', [q['query_index'] for q in queries], list(range(len(queries))))
        candidate_queries.append([(q['identity'], q['candidate_ap'], q['candidate_first_rank']) for q in queries])
        for q in queries:
            for prefix in ('control', 'candidate'):
                check('query_values', p_label + f'/{q["query_index"]}/{prefix}', math.isfinite(q[prefix + '_ap']) and 0 <= q[prefix + '_ap'] <= 1 and 1 <= q[prefix + '_first_rank'] <= expected_gallery[dataset])
        calculated = {prefix: metrics(queries, prefix) for prefix in ('control', 'candidate')}
        for prefix, name in (('control', pair['control']), ('candidate', condition)):
            for metric, value in calculated[prefix].items():
                near('saved_pairs', p_label + '/' + prefix + '/' + metric, pair['metrics'][name][metric], value)
                if prefix == 'candidate':
                    near('saved_pairs', p_label + '/official/' + metric, receipt['metrics'][metric], value, 1e-5)
                    if metric != 'mAP':
                        max_cmc_difference = max(max_cmc_difference, abs(receipt['metrics'][metric] - value))
        groups = verify_groups('saved_pairs', p_label, queries, 'control', 'candidate', pair['rank1_group_ap_contributions'])
        equal('saved_pairs', p_label + '/repairs', pair['rank1_repairs'], len(groups['repaired']))
        equal('saved_pairs', p_label + '/new-errors', pair['rank1_new_errors'], len(groups['new_error']))
        delta = [q['candidate_ap'] - q['control_ap'] for q in queries]
        equal('saved_pairs', p_label + '/improved', pair['query_ap_improved'], sum(v > 1e-8 for v in delta))
        equal('saved_pairs', p_label + '/worsened', pair['query_ap_worsened'], sum(v < -1e-8 for v in delta))
        for metric in calculated['control']:
            near('saved_pairs', p_label + '/delta/' + metric, pair['delta_metrics'][metric], calculated['candidate'][metric] - calculated['control'][metric])
        n_tail = math.ceil(.05 * len(queries))
        near('saved_pairs', p_label + '/tail', pair['worst_five_percent_query_ap_delta_mean_points'], math.fsum(sorted(v * 100 for v in delta)[:n_tail]) / n_tail)
        identities = defaultdict(list)
        for q in queries:
            identities[q['identity']].append((q['candidate_ap'] - q['control_ap']) * 100)
        equal('saved_pairs', p_label + '/identity-list', [r['identity'] for r in pair['identity_changes']], sorted(identities))
        for ir in pair['identity_changes']:
            values = identities[ir['identity']]
            equal('saved_pairs', p_label + f'/identity{ir["identity"]}/queries', ir['queries'], len(values))
            near('saved_pairs', p_label + f'/identity{ir["identity"]}/AP', ir['mean_delta_ap_points'], math.fsum(values) / len(values))
        near('saved_pairs', p_label + '/identity-macro', pair['identity_macro_mean_delta_ap_points'], math.fsum(math.fsum(v) / len(v) for v in identities.values()) / len(identities))
    equal('saved_pairs', label + '/identical-candidate-across-controls', candidate_queries[0], candidate_queries[1])
    row_results.append(dict(dataset=dataset, comparison=condition, best_epoch=selected['epoch'], metrics=receipt['metrics'], recorded_steps=count, queries=expected_queries[dataset]))

equal('aggregate', 'six-unique-conditions', set(summaries), {(d, c) for d in expected_queries for c in ('uniform', 'ambiguity')})
equal('aggregate', 'total-formal-updates', aggregate['total_formal_updates'], sum(r['recorded_steps'] for r in row_results))
equal('aggregate', 'total-formal-epochs', aggregate['total_formal_epochs'], sum(len(s['history']) for s in summaries.values()))
equal('aggregate', 'reported-zero-calls', (aggregate['neural_calls'], aggregate['optimizer_updates']), (0, 0))
DETAILS['six_results'] = row_results
DETAILS['maximum_official_vs_query_CMC_difference_points'] = max_cmc_difference
contrast_results = []
for contrast in aggregate['ambiguity_minus_uniform']:
    dataset = contrast['dataset']
    u, a = summaries[dataset, 'uniform'], summaries[dataset, 'ambiguity']
    for left, right in zip(u['pairs'], a['pairs'], strict=True):
        equal('contrasts', dataset + '/control-name', left['control'], right['control'])
        equal('contrasts', dataset + '/' + left['control'] + '/control-identical',
              [(q['query_index'], q['identity'], q['control_ap'], q['control_first_rank']) for q in left['query_changes']],
              [(q['query_index'], q['identity'], q['control_ap'], q['control_first_rank']) for q in right['query_changes']])
    q_u, q_a = u['pairs'][0]['query_changes'], a['pairs'][0]['query_changes']
    rebuilt = [dict(query_index=x['query_index'], identity=x['identity'],
                    uniform_ap=x['candidate_ap'], ambiguity_ap=y['candidate_ap'],
                    delta_ap_points=(y['candidate_ap'] - x['candidate_ap']) * 100,
                    uniform_first_rank=x['candidate_first_rank'], ambiguity_first_rank=y['candidate_first_rank'])
               for x, y in zip(q_u, q_a, strict=True)]
    equal('contrasts', dataset + '/all-query-records', contrast['query_changes'], rebuilt)
    equal('contrasts', dataset + '/count', contrast['queries'], len(rebuilt))
    groups = verify_groups('contrasts', dataset, rebuilt, 'uniform', 'ambiguity', contrast['group_ap_contributions'])
    equal('contrasts', dataset + '/repairs', contrast['rank1_repairs'], len(groups['repaired']))
    equal('contrasts', dataset + '/new-errors', contrast['rank1_new_errors'], len(groups['new_error']))
    before, after = metrics(rebuilt, 'uniform'), metrics(rebuilt, 'ambiguity')
    for metric in before:
        near('contrasts', dataset + '/delta/' + metric, contrast['delta_metrics'][metric], after[metric] - before[metric], 1e-5)
        equal('contrasts', dataset + '/reported-subtraction/' + metric, contrast['delta_metrics'][metric], a['metrics'][metric] - u['metrics'][metric])
    near('contrasts', dataset + '/net-repair', contrast['delta_metrics']['Rank-1'], (len(groups['repaired']) - len(groups['new_error'])) * 100 / len(rebuilt), 1e-5)
    near('contrasts', dataset + '/AP-group-sum', contrast['delta_metrics']['mAP'], math.fsum(math.fsum(v) for v in groups.values()) / len(rebuilt))
    tail = math.ceil(.05 * len(rebuilt))
    near('contrasts', dataset + '/tail', contrast['worst_five_percent_query_ap_delta_mean_points'], math.fsum(sorted(q['delta_ap_points'] for q in rebuilt)[:tail]) / tail)
    contrast_results.append({k: v for k, v in contrast.items() if k != 'query_changes'})
DETAILS['contrast_results'] = contrast_results

old = read(BASE / 'results912/logs/region_use_remaining_20261013_911/campaign.json')
new = read(BASE / 'results912/logs/region_use_remaining_strict_20261013_912/region_use_remaining_20261013_911/campaign.json')
old_exit = read(BASE / 'results912/logs/region_use_remaining_admin911/EXIT.json')
new_exit = read(BASE / 'results912/logs/region_use_remaining_strict_admin912/EXIT.json')
terminal = read(BASE / 'TERMINAL_STORAGE_912.json')
node = read(BASE / 'FIRST_STRICT_NODE_912.json')
receipt = receipts['RGBNT100', 'ambiguity']
training = training_records['RGBNT100', 'ambiguity']
equal('sixth_continuation', 'captured-old-state', terminal['state'], old)
equal('sixth_continuation', 'captured-old-exit', terminal['terminal'], old_exit)
equal('sixth_continuation', 'original-storage-stop', (old['status'], old['stopped_before_mode'], old_exit['exit_code']), ('STOPPED_BEFORE_STAGE_INSUFFICIENT_STORAGE', 'evaluate', 1))
equal('sixth_continuation', 'original-no-strict', terminal['first_strict_artifacts'], {'official_metrics.json': False, 'official_distances.pt': False})
equal('sixth_continuation', 'old-modes', [s['mode'] for s in old['jobs'][0]['stages']], ['prepare', 'm0', 'train'])
equal('sixth_continuation', 'new-modes', [s['mode'] for s in new['jobs'][0]['stages']], ['prepare', 'm0', 'train', 'evaluate'])
equal('sixth_continuation', 'original-three-stage-objects-reused', new['jobs'][0]['stages'][:3], old['jobs'][0]['stages'])
equal('sixth_continuation', 'all-stages-success', [(s['status'], s['exit_code']) for s in new['jobs'][0]['stages']], [('COMPLETE', 0)] * 4)
equal('sixth_continuation', 'terminal-status', (new['status'], new_exit['exit_code']), ('TRAINING_AND_STRICT_EVALUATION_COMPLETE', 0))
equal('sixth_continuation', 'result-receipt', new['jobs'][0]['result'], receipt)
equal('sixth_continuation', 'first-node-receipt', node['strict_receipt'], receipt)
equal('sixth_continuation', 'first-node-exit', node['terminal'], new_exit)
equal('sixth_continuation', 'first-node-stages', node['stages'], new['jobs'][0]['stages'])
equal('sixth_continuation', 'first-node-no-live', set(node['process_live'].values()), {False})
equal('sixth_continuation', 'original-failure-preserved', new['original_remaining_exit_code'], 1)
equal('sixth_continuation', 'source-map-unchanged', new['source_sha256'], old['source_sha256'])
equal('sixth_continuation', 'initializer-map-unchanged', new['initialization_sha256'], old['initialization_sha256'])
equal('sixth_continuation', 'checkpoint-before-and-after', terminal['full_checkpoint']['sha256'], receipt['checkpoint_sha256'])
equal('sixth_continuation', 'captured-best-epoch', terminal['full_training']['best_epoch'], receipt['selected_epoch'])
equal('sixth_continuation', 'captured-best-metrics', terminal['full_training']['best']['official_fused'], receipt['metrics'])
equal('sixth_continuation', 'physical-gpus-recorded', (old['physical_gpus'], new['physical_gpus'], node['launch']['physical_gpus']), ([0, 1], [0, 1], [0, 1]))
equal('sixth_continuation', 'launch-invocations-recorded', node['launch']['launch_invocations'], 1)
evaluate = new['jobs'][0]['stages'][-1]
train_command = old['jobs'][0]['stages'][-1]['command']
expected_command = list(train_command)
expected_command[expected_command.index('--mode') + 1] = 'evaluate'
equal('sixth_continuation', 'only-mode-change-from-original-train', evaluate['command'], expected_command)
check('sixth_continuation', 'time-order', old_exit['completed_at'] < new_exit['started_at'] < evaluate['started_at'] < receipt['completed_at'] < evaluate['completed_at'] < new_exit['completed_at'])
for source in ('remaining_strict912/tools/finish_region_use_evaluation.py', 'review/remaining911/evidence_snapshot/production/tools/report_region_use_endpoint.py'):
    ast.parse((BASE / source).read_text(encoding='utf-8'))
review = read(BASE / 'review/remaining_strict912/SOURCE_REVIEW.json')
controller_hash = digest(BASE / 'remaining_strict912/tools/finish_region_use_evaluation.py')
equal('source_chain', 'review-controller', controller_hash, review['reviewed_controller_sha256'])
equal('source_chain', 'run-controller', controller_hash, new['strict_continuation_controller_sha256'])
equal('source_chain', 'launch-controller', controller_hash, node['launch']['source_sha256'])
equal('source_chain', 'review-seal', digest(BASE / 'remaining_strict912/INPUT_SEAL.json'), review['reviewed_seal_sha256'])
equal('source_chain', 'run-seal', new['strict_continuation_seal_sha256'], review['reviewed_seal_sha256'])
source_names = ['report_prepool_dense_correspondence.py', 'train_rgbnt100_signal_oof.py', 'train_msvr310_signal_oof.py', 'run_foundation_recipe.py', 'run_region_use_probe.py', 'run_clean_clip_joint.py', 'run_visual_update_control.py', 'run_correspondence_roles.py', 'official_three_dataset_data.py', 'run_official_three_dataset_roles.py', 'queue_region_use_probe.py', 'queue_native_research.py']
for name in source_names:
    path = BASE / 'sealed_source_snapshot911/tools' / name
    ast.parse(path.read_text(encoding='utf-8'))
    equal('source_chain', name, digest(path), new['source_sha256']['tools/' + name])
DETAILS['bounded_source_hashes_checked'] = len(source_names)
DETAILS['source_review_reused_limits'] = dict(verdict=review['verdict'], review_independence=review['review_independence'], acceptance_status=review['acceptance_status'], runtime=review['runtime'])

cpu = read(BASE / 'results912/logs/region_use_remaining_strict_admin912/CPU_REPORT_912.json')
reporter_hash = digest(BASE / 'review/remaining911/evidence_snapshot/production/tools/report_region_use_endpoint.py')
equal('cpu_receipt', 'complete-exit', (cpu['status'], cpu['exit_code']), ('COMPLETE', 0))
equal('cpu_receipt', 'reporter-pin', cpu['reporter_sha256'], reporter_hash)
equal('cpu_receipt', 'recorded-zero-nn', (cpu['model_inference'], cpu['optimizer_updates']), (0, 0))
check('cpu_receipt', 'after-strict-exit', new_exit['completed_at'] < cpu['started_at'] < cpu['completed_at'])
cpu_log = (BASE / 'results912/logs/region_use_remaining_strict_admin912/CPU_REPORT_912.log').read_text().splitlines()
equal('cpu_receipt', 'one-stdout-result', len(cpu_log), 1)
cpu_stdout = json.loads(cpu_log[0])
equal('cpu_receipt', 'stdout-metrics', cpu_stdout['metrics'], receipt['metrics'])
equal('cpu_receipt', 'stdout-steps', cpu_stdout['formal_steps'], summaries['RGBNT100', 'ambiguity']['formal_recorded_steps'])
collector = (BASE / 'collect_remaining_strict_terminal912.ps1').read_text()
equal('cpu_receipt', 'one-subprocess-run-source-site', collector.count('subprocess.run('), 1)
check('cpu_receipt', 'guard-before-call', collector.index('assert not receipt.exists() and not out.exists()') < collector.index('subprocess.run('))
check('cpu_receipt', 'true-exit-source', 'exit_code=run.returncode' in collector and "CUDA_VISIBLE_DEVICES=''" in collector)

cleanup = read(BASE / 'CLEANUP_912.json')
equal('cleanup_receipt', 'retained-duplicate-receipt', cleanup, read(BASE / 'results912/logs/region_use_remaining_admin911/CLEANUP_912.json'))
equal('cleanup_receipt', 'retired-unprotected-closed', (cleanup['retired']['protected_by_current_50_artifacts'], cleanup['retired']['training_status'], cleanup['retired']['epochs']), (False, 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE', 50))
equal('cleanup_receipt', 'counts-recorded', (cleanup['protected_original_artifacts'], cleanup['protected_current_bests'], cleanup['protected_signal_bests'], cleanup['current_best_deleted']), (50, 6, 8, 0))
equal('cleanup_receipt', 'extra-pins-count', len(cleanup['protected_additional_sha256']), 14)
for summary in summaries.values():
    candidate = summary['candidate']
    equal('cleanup_receipt', candidate['run_dir'], cleanup['protected_additional_sha256'][candidate['run_dir'] + '/best_map.pth'], candidate['checkpoint_sha256'])
check('cleanup_receipt', 'retired-outside-protected', cleanup['retired']['path'] not in cleanup['protected_additional_sha256'])
DETAILS['cleanup_assurance'] = 'Receipt and current-six pin consistency only. Local historical archive bytes, remote retirement, original50/Signal binaries and broad archive intactness were not independently re-observed.'

report = (BASE / 'SIX_RESULTS_REPORT_913.md').read_text(encoding='utf-8')
table_rows = [line for line in report.splitlines() if line.startswith('| RGBNT') or line.startswith('| MSVR')]
expected_rows = []
for row in aggregate['rows']:
    cells = [row['dataset'], row['comparison'], str(row['best_epoch'])] + [f'{row["metrics"][k]:.4f}' for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')] + [f'{row["last_metrics"]["mAP"]:.4f}']
    expected_rows.append('| ' + ' | '.join(cells) + ' |')
for contrast in aggregate['ambiguity_minus_uniform']:
    cells = [contrast['dataset'], f'{contrast["delta_metrics"]["mAP"]:+.4f}', f'{contrast["delta_metrics"]["Rank-1"]:+.4f}', f'{contrast["rank1_repairs"]}/{contrast["rank1_new_errors"]}']
    expected_rows.append('| ' + ' | '.join(cells) + ' |')
for row in aggregate['rows']:
    cells = [row['dataset'], row['comparison']] + [f'{c["delta_metrics"]["mAP"]:+.4f}/{c["delta_metrics"]["Rank-1"]:+.4f}' for c in row['controls']]
    expected_rows.append('| ' + ' | '.join(cells) + ' |')
equal('report', 'all-15-table-rows', table_rows, expected_rows)
for token in ('300 epochs and 12968 recorded updates', 'ACTIVE_UNMET', 'single-seed development', 'not statistical significance', 'same-model global harm', 'decomposition remains unexecuted', 'No claim that compression, background, geometry or Mamba caused', 'no stronger teacher', 'EXIT1', 'one unchanged first evaluate', new_exit['completed_at'], receipt['checkpoint_sha256'], receipt['distance_sha256'], 'consumed official benchmarks', 'no full-flow multiple-seed evidence', 'NOT_ADMITTED'):
    check('report', token, token in report)

DETAILS['sixth_exits'] = dict(original=old_exit, new=new_exit, evaluate_stage=evaluate, cpu_report=cpu)
DETAILS['bounds'] = dict(ssh_calls=0, gpu_queries=0, neural_calls=0, optimizer_updates=0, production_imports=0, source_or_result_writes=0, new_agents=0, raw_remote_binary_rechecks=0)
result = dict(status='PASS' if not FAILURES else 'FAIL', generated_at=datetime.now().astimezone().isoformat(),
              interpreter=sys.executable, python_version=sys.version, script_sha256=digest(Path(__file__)),
              command=[sys.executable, '-B', str(Path(__file__))], check_count=sum(COUNTS.values()),
              checks_by_section=dict(COUNTS), failures=FAILURES, details=DETAILS, reviewed_hashes=HASHES,
              scope='Deterministic local saved-byte consistency; no authentication of remote runtime, raw distances, checkpoints or dataset annotations.')
(HERE / 'CHECKS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], checks=result['check_count'], failures=FAILURES, sections=dict(COUNTS), hashes=len(HASHES)), ensure_ascii=False))
raise SystemExit(bool(FAILURES))

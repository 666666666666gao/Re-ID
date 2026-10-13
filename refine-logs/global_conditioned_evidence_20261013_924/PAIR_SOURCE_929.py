"""Combine accepted saved-query summaries; no model or ranking replay."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

w = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = [w / 'EVIDENCE_GRADIENT_CPU_SUMMARY_928.json', w / 'EVIDENCE_RANDOM_CPU_SUMMARY_929.json']
exposures = ['gradient', 'random']
audited_root = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903')
audit_paths = []
for exposure, section, audit_name, accepted_key in (
    ('gradient', 928, 'RUNTIME_AUDIT.json', 'first_endpoint_cpu_evidence_accepted'),
    ('random', 929, 'SECOND_RUNTIME_AUDIT.json', 'second_endpoint_cpu_evidence_accepted'),
):
    audit_path = w / 'review_endpoint_report928' / audit_name
    audit = json.loads(audit_path.read_text())
    assert audit[accepted_key] and not audit['unresolved_blocking_evidence_defects']
    bound_paths = [w / f'EVIDENCE_{exposure.upper()}_CPU_SUMMARY_{section}.json',
                   w / f'EVIDENCE_{exposure.upper()}_CPU_EXIT_{section}.json',
                   w / f'evidence_{exposure}_results{section}/output/training_batch_order.jsonl',
                   w / f'evidence_{exposure}_results{section}/output/training_steps.jsonl']
    if exposure == 'random':
        bound_paths.append(w / 'EVIDENCE_TERMINAL_OBSERVATION_929.json')
    for p in bound_paths:
        key = (audited_root / p.relative_to(w)).as_posix()
        assert sha(p) == audit['audited_input_hashes'][key]
    audit_paths.append(audit_path)
summaries = [json.loads(p.read_text()) for p in paths]
paths.extend(audit_paths)
for summary, exposure, section in zip(summaries, exposures, [928, 929]):
    assert summary['status'] == 'ONE_NEW_FORMATION_ENDPOINT_CPU_DIAGNOSIS_COMPLETE'
    assert summary['exposure'] == exposure and summary['dataset'] == 'RGBNT201'
    assert summary['formal_recorded_steps'] == 2649 and summary['selected_epoch'] == 8
    assert summary['neural_calls'] == summary['optimizer_updates'] == 0
    assert not summary['same_model_global_decomposition_available']
    exit_path = w / f'EVIDENCE_{exposure.upper()}_CPU_EXIT_{section}.json'
    assert json.loads(exit_path.read_text())['exit_code'] == 0
    paths.append(exit_path)

terminal_path = w / 'EVIDENCE_TERMINAL_OBSERVATION_929.json'
terminal = json.loads(terminal_path.read_text())
assert terminal['campaign']['status'] == 'COMPLETE_TWO_FULL50_FIRST_STRICT'
assert terminal['controller_exit']['exit_code'] == 0
assert not terminal['supervisor_live'] and not terminal['worker_live']
paths.append(terminal_path)

query_sets = [next(p for p in s['pairs'] if p['control'] == 'raw_semantic')['query_changes'] for s in summaries]
assert len(query_sets[0]) == len(query_sets[1]) == 836
assert [q['query_index'] for q in query_sets[0]] == list(range(836))
queries = []
groups = {'repair': [], 'new_error': [], 'kept_correct': [], 'kept_wrong': []}
identities = {}
for gradient, random in zip(*query_sets):
    for k in ('query_index', 'identity', 'control_ap', 'control_first_rank'):
        assert gradient[k] == random[k]
    row = {'query_index': gradient['query_index'], 'identity': gradient['identity'],
           'control_ap': random['candidate_ap'], 'candidate_ap': gradient['candidate_ap'],
           'control_first_rank': random['candidate_first_rank'],
           'candidate_first_rank': gradient['candidate_first_rank']}
    delta = 100 * (row['candidate_ap'] - row['control_ap'])
    before, after = row['control_first_rank'] == 1, row['candidate_first_rank'] == 1
    group = ('kept_correct' if after else 'new_error') if before else ('repair' if after else 'kept_wrong')
    groups[group].append(delta)
    identities.setdefault(row['identity'], []).append(delta)
    queries.append(row)
assert len(identities) == 30

metrics = {}
for label, key in [('random', 'control'), ('gradient', 'candidate')]:
    metrics[label] = {'mAP': 100 * math.fsum(q[key + '_ap'] for q in queries) / len(queries)}
    metrics[label].update({f'Rank-{k}': 100 * sum(q[key + '_first_rank'] <= k for q in queries) / len(queries) for k in (1, 5, 10)})
    summary = summaries[exposures.index(label)]
    assert all(abs(metrics[label][k] - summary['metrics'][k]) < 1e-5 for k in metrics[label])
delta_metrics = {k: metrics['gradient'][k] - metrics['random'][k] for k in metrics['gradient']}
group_result = {k: {'queries': len(v), 'contribution_to_total_map_points': math.fsum(v) / len(queries)} for k, v in groups.items()}
assert abs(math.fsum(v['contribution_to_total_map_points'] for v in group_result.values()) - delta_metrics['mAP']) < 1e-10
identity_rows = [{'identity': k, 'queries': len(v), 'mean_delta_ap_points': math.fsum(v) / len(v)} for k, v in sorted(identities.items())]
deltas = sorted(100 * (q['candidate_ap'] - q['control_ap']) for q in queries)
tail = deltas[:math.ceil(.05 * len(queries))]

orders = [w / f'evidence_{name}_results{section}' / 'output/training_batch_order.jsonl' for name, section in zip(exposures, [928, 929])]
assert orders[0].read_bytes() == orders[1].read_bytes()
steps_paths = [p.parent / 'training_steps.jsonl' for p in orders]
steps = [[json.loads(line) for line in p.read_text().splitlines()] for p in steps_paths]
assert len(steps[0]) == len(steps[1]) == 2649
for a, b in zip(*steps):
    assert (a['epoch'], a['batch'], a['lr']) == (b['epoch'], b['batch'], b['lr'])
    for k in ('source', 'vjp_scale', 'support', 'active_sample_modalities', 'available_sample_modalities', 'excluded_patches'):
        assert a['evidence_exposure'][k] == b['evidence_exposure'][k]
max_difference = {'global_loss': max(abs(a['global_loss'] - b['global_loss']) for a, b in zip(*steps))}
max_difference.update({k: max(abs(a['evidence_exposure'][k] - b['evidence_exposure'][k]) for a, b in zip(*steps)) for k in ('target', 'gradient_norm_min', 'gradient_norm_max', 'gradient_norm_mean')})
paths.extend(orders + steps_paths)
value = {'status': 'TWO_ACCEPTED_ENDPOINT_SAVED_QUERY_COMBINATION', 'at': datetime.now().astimezone().isoformat(),
         'dataset': 'RGBNT201', 'seed': 42, 'control': 'random', 'candidate': 'gradient',
         'metrics': metrics, 'delta_metrics': delta_metrics,
         'official_delta_metrics': {k: summaries[0]['metrics'][k] - summaries[1]['metrics'][k] for k in summaries[0]['metrics']},
         'rank1_repairs': len(groups['repair']), 'rank1_new_errors': len(groups['new_error']),
         'rank1_group_ap_contributions': group_result, 'query_changes': queries,
         'query_ap_improved': sum(d > 0 for d in deltas), 'query_ap_worsened': sum(d < 0 for d in deltas),
         'identity_changes': identity_rows,
         'identity_macro_mean_delta_ap_points': math.fsum(r['mean_delta_ap_points'] for r in identity_rows) / len(identity_rows),
         'worst_five_percent_query_ap_delta_mean_points': math.fsum(tail) / len(tail),
         'formal_steps': 5298, 'training_batch_order_equal': True,
         'support_and_exclusion_counts_equal': True, 'recorded_teacher_max_abs_difference': max_difference,
         'same_model_global_decomposition_available': False,
         'neural_calls': 0, 'optimizer_updates': 0, 'ranking_replays': 0,
         'source_sha256': sha(Path(__file__)), 'input_sha256': {str(p): sha(p) for p in paths},
         'boundary': 'Stdlib arithmetic on two independently accepted saved-query CPU reports. No new dataset/tensor ranking replay or same-model global/correction/fused decomposition; unchanged recorded teacher scalars are not feature equality. One seed and one dataset, selected mAP-best per arm; no causal proof, seed variance, equivalence or all-Signal superiority.'}
out = w / 'EVIDENCE_PAIR_SUMMARY_929.json'
assert not out.exists()
out.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: value[k] for k in ('status', 'delta_metrics', 'rank1_repairs', 'rank1_new_errors', 'identity_macro_mean_delta_ap_points', 'recorded_teacher_max_abs_difference')}), flush=True)

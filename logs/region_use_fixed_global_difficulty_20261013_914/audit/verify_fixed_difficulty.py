"""Fresh CPU-only saved-query arithmetic audit; never import the producer."""
import ast
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 50
HERE = Path(__file__).resolve().parent
REQUEST = HERE / 'REQUEST.md'
PATHS = [Path(line) for line in REQUEST.read_text(encoding='utf-8').splitlines()
         if line.startswith(('C:\\', 'D:\\'))]
SOURCE, RESULT, REPORT, AGGREGATE, PRIOR, *SUMMARIES = PATHS
input_paths = [REQUEST, *PATHS]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
texts = {str(p): p.read_text(encoding='utf-8') for p in input_paths}

def read_json(p):
    return json.loads(texts[str(p)], parse_float=Decimal)

result = read_json(RESULT)
aggregate = read_json(AGGREGATE)
prior = read_json(PRIOR)
counts = {}
failures = []
max_errors = {}

def check(category, name, passed, detail=None):
    bucket = counts.setdefault(category, {'checks': 0, 'passed': 0, 'failed': 0})
    bucket['checks'] += 1
    bucket['passed' if passed else 'failed'] += 1
    if not passed:
        failures.append({'category': category, 'name': name, 'detail': detail})

def close(category, name, expected, observed, tolerance='1e-10'):
    error = abs(Decimal(expected) - Decimal(observed))
    if error > max_errors.get(category, Decimal(0)):
        max_errors[category] = error
    check(category, name, error <= Decimal(tolerance),
          {'expected': expected, 'observed': observed, 'absolute_error': error,
           'absolute_tolerance': tolerance})

def compare_tree(expected, observed, name):
    if isinstance(expected, dict):
        check('result_schema', name + '.keys', set(expected) == set(observed))
        for key in expected:
            compare_tree(expected[key], observed[key], name + '.' + key)
    elif isinstance(expected, list):
        check('result_schema', name + '.length', len(expected) == len(observed))
        for index, value in enumerate(expected):
            compare_tree(value, observed[index], name + f'[{index}]')
    elif isinstance(expected, Decimal):
        close('result_numeric_fields', name, expected, observed)
    else:
        check('result_exact_fields', name, expected == observed,
              {'expected': expected, 'observed': observed})

def canonical_pins(mapping):
    return {Path(p).resolve(): h for p, h in mapping.items()}

derived_pins = canonical_pins(result['inputs'])
aggregate_pins = canonical_pins(aggregate['inputs'])
prior_pins = canonical_pins(prior['reviewed_hashes'])
check('input_pins', 'derived_exact_input_set',
      set(derived_pins) == {p.resolve() for p in [SOURCE, AGGREGATE, *SUMMARIES]})
for p in [SOURCE, AGGREGATE, *SUMMARIES]:
    check('input_pins', 'derived:' + str(p), hashes[str(p)] == derived_pins[p.resolve()])
for p in SUMMARIES:
    check('input_pins', 'accepted_aggregate:' + str(p),
          hashes[str(p)] == aggregate_pins[p.resolve()])
for p in [AGGREGATE, *SUMMARIES]:
    check('input_pins', 'prior_bounded_audit:' + str(p),
          hashes[str(p)] == prior_pins[p.resolve()])

tree = ast.parse(texts[str(SOURCE)], filename=str(SOURCE))
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.extend(a.name for a in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.append(node.module)
check('source_path', 'only_declared_standard_library_imports',
      set(imports) == {'hashlib', 'json', 'math', 'collections', 'datetime', 'pathlib'})
calls = Counter(node.func.id for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name))
for function in ('rank_bin', 'summarize'):
    check('source_path', function + '_called', calls[function] > 0)
check('result_metadata', 'complete_status',
      result['status'] == 'SAVED_QUERY_FIXED_GLOBAL_DIFFICULTY_CPU_DESCRIPTION')
check('result_metadata', 'recorded_neural_calls_zero', result['neural_calls'] == 0)
check('result_metadata', 'recorded_optimizer_updates_zero', result['optimizer_updates'] == 0)
check('result_metadata', 'accepted_aggregate_status',
      aggregate['status'] == 'SIX_ORIGINAL_FULL50_FIRST_STRICT_RESULTS_COLLECTED')
check('result_metadata', 'prior_blockers_empty', prior['blockers'] == [])
check('result_metadata', 'prior_same_family_provisional',
      (prior['review_independence'], prior['acceptance_status']) == ('same-family', 'provisional'))

endpoints = {}
for path in SUMMARIES:
    summary = read_json(path)
    ds, condition = summary['dataset'], summary['comparison']
    pairs = [pair for pair in summary['pairs'] if pair['control'] == 'independent_global']
    check('endpoint_metadata', ds + ':' + condition + ':one_global_pair', len(pairs) == 1)
    pair = pairs[0]
    check('endpoint_metadata', ds + ':' + condition + ':pair_labels',
          (pair['dataset'], pair['candidate']) == (ds, condition))
    check('endpoint_metadata', ds + ':' + condition + ':complete_status',
          summary['status'] == 'ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE')
    check('endpoint_metadata', ds + ':' + condition + ':saved_epoch_coverage',
          [row['epoch'] for row in summary['history']] == list(range(1, 51)))
    queries = pair['query_changes']
    indices = [row['query_index'] for row in queries]
    check('query_alignment', ds + ':' + condition + ':unique_query_indices',
          len(indices) == len(set(indices)))
    check('query_alignment', ds + ':' + condition + ':complete_index_range',
          sorted(indices) == list(range(len(indices))))
    for q in queries:
        name = ds + ':' + condition + ':' + str(q['query_index'])
        for key in ('control_ap', 'candidate_ap'):
            value = Decimal(q[key])
            check('query_domain', name + ':' + key,
                  value.is_finite() and 0 <= value <= 1)
        for key in ('control_first_rank', 'candidate_first_rank'):
            check('query_domain', name + ':' + key,
                  type(q[key]) is int and q[key] >= 1)
    endpoints[(ds, condition)] = (summary, pair, {q['query_index']: q for q in queries})
check('endpoint_metadata', 'all_six_unique_endpoints', len(endpoints) == 6)

recomputed = []
totals = []
memberships = []
expected_counts = {'RGBNT201': 836, 'RGBNT100': 1715, 'MSVR310': 591}
predicates = [
    ('1', lambda rank: rank == 1),
    ('2-5', lambda rank: 2 <= rank <= 5),
    ('6-10', lambda rank: 6 <= rank <= 10),
    ('>10', lambda rank: rank >= 11),
    ('global first correct', lambda rank: rank == 1),
    ('global first wrong', lambda rank: rank > 1),
]

for ds in ('RGBNT201', 'RGBNT100', 'MSVR310'):
    _, uniform_pair, uq = endpoints[(ds, 'uniform')]
    _, ambiguity_pair, aq = endpoints[(ds, 'ambiguity')]
    n = len(uq)
    check('query_alignment', ds + ':expected_legal_query_count', n == expected_counts[ds])
    check('query_alignment', ds + ':same_index_set', set(uq) == set(aq))
    check('query_alignment', ds + ':same_original_query_order',
          [q['query_index'] for q in uniform_pair['query_changes']] ==
          [q['query_index'] for q in ambiguity_pair['query_changes']])
    check('query_alignment', ds + ':same_global_metrics',
          uniform_pair['metrics']['independent_global'] ==
          ambiguity_pair['metrics']['independent_global'])
    observations = []
    for index in sorted(uq):
        u, a = uq[index], aq[index]
        for key in ('query_index', 'identity', 'control_ap', 'control_first_rank'):
            check('query_alignment', ds + ':' + str(index) + ':' + key, u[key] == a[key])
        observations.append({
            'index': index, 'identity': u['identity'],
            'global': (Decimal(u['control_ap']), u['control_first_rank']),
            'uniform': (Decimal(u['candidate_ap']), u['candidate_first_rank']),
            'ambiguity': (Decimal(a['candidate_ap']), a['candidate_first_rank']),
        })

    group_results = []
    all_members = []
    for label, predicate in predicates:
        selected = [q for q in observations if predicate(q['global'][1])]
        count = len(selected)
        indices = [q['index'] for q in selected]
        all_members.append(indices)
        ap_sums = {name: sum((q[name][0] for q in selected), Decimal(0))
                   for name in ('global', 'uniform', 'ambiguity')}
        group = {
            'dataset': ds, 'global_first_positive_rank_bin': label,
            'queries': count, 'identities': len({q['identity'] for q in selected}),
            'global_mean_ap': ap_sums['global'] * 100 / count,
        }
        for key, baseline, candidate in (
            ('uniform', 'global', 'uniform'),
            ('ambiguity', 'global', 'ambiguity'),
            ('ambiguity_minus_uniform', 'uniform', 'ambiguity'),
        ):
            points = (ap_sums[candidate] - ap_sums[baseline]) * 100
            entry = {
                'within_bin_mean_delta_map_points': points / count,
                'contribution_to_all_query_delta_map_points': points / n,
                'rank1_repairs': sum(q[baseline][1] > 1 and q[candidate][1] == 1 for q in selected),
                'rank1_new_errors': sum(q[baseline][1] == 1 and q[candidate][1] > 1 for q in selected),
            }
            if key != 'ambiguity_minus_uniform':
                entry['mean_ap'] = ap_sums[candidate] * 100 / count
            group[key] = entry
            close('denominator_identity', ds + ':' + label + ':' + key,
                  entry['within_bin_mean_delta_map_points'] * count / n,
                  entry['contribution_to_all_query_delta_map_points'])
        group_results.append(group)
        memberships.append({'dataset': ds, 'label': label, 'indices': indices})
    check('partition_coverage', ds + ':strata_partition',
          sorted(index for group in all_members[:4] for index in group) == list(range(n)))
    check('partition_coverage', ds + ':pooled_partition',
          sorted(index for group in all_members[4:] for index in group) == list(range(n)))
    check('partition_coverage', ds + ':pooled_correct_equals_rank1', all_members[4] == all_members[0])
    check('partition_coverage', ds + ':pooled_wrong_equals_rank_above1',
          set(all_members[5]) == set(all_members[1] + all_members[2] + all_members[3]))
    recomputed.append({'dataset': ds, 'queries': n,
                       'strata': group_results[:4], 'pooled': group_results[4:]})

    contrast_rows = [row for row in aggregate['ambiguity_minus_uniform'] if row['dataset'] == ds]
    check('aggregate_alignment', ds + ':one_contrast', len(contrast_rows) == 1)
    contrast = contrast_rows[0]
    check('aggregate_alignment', ds + ':contrast_count', contrast['queries'] == n)
    contrast_queries = {q['query_index']: q for q in contrast['query_changes']}
    check('aggregate_alignment', ds + ':contrast_unique_full_index_set',
          len(contrast_queries) == len(contrast['query_changes']) == n and
          set(contrast_queries) == set(uq))
    for q in observations:
        stored = contrast_queries[q['index']]
        for field, expected in (
            ('identity', q['identity']), ('uniform_ap', q['uniform'][0]),
            ('ambiguity_ap', q['ambiguity'][0]), ('uniform_first_rank', q['uniform'][1]),
            ('ambiguity_first_rank', q['ambiguity'][1]),
        ):
            check('aggregate_alignment', ds + ':' + str(q['index']) + ':' + field,
                  stored[field] == expected)
        close('aggregate_query_delta', ds + ':' + str(q['index']),
              (q['ambiguity'][0] - q['uniform'][0]) * 100, stored['delta_ap_points'])

    for key, baseline, candidate in (
        ('uniform', 'global', 'uniform'),
        ('ambiguity', 'global', 'ambiguity'),
        ('ambiguity_minus_uniform', 'uniform', 'ambiguity'),
    ):
        direct_delta = sum(((q[candidate][0] - q[baseline][0]) * 100
                            for q in observations), Decimal(0)) / n
        repairs = sum(q[baseline][1] > 1 and q[candidate][1] == 1 for q in observations)
        harms = sum(q[baseline][1] == 1 and q[candidate][1] > 1 for q in observations)
        rank1_delta = Decimal(repairs - harms) * 100 / n
        for group_name, groups in (('strata', group_results[:4]), ('pooled', group_results[4:])):
            close('additivity', ds + ':' + key + ':' + group_name + ':AP', direct_delta,
                  sum(g[key]['contribution_to_all_query_delta_map_points'] for g in groups))
            check('additivity', ds + ':' + key + ':' + group_name + ':repairs',
                  sum(g[key]['rank1_repairs'] for g in groups) == repairs)
            check('additivity', ds + ':' + key + ':' + group_name + ':new_errors',
                  sum(g[key]['rank1_new_errors'] for g in groups) == harms)
        if key == 'ambiguity_minus_uniform':
            accepted = contrast
        else:
            summary, pair, _ = endpoints[(ds, key)]
            saved_rows = [row for row in aggregate['rows']
                          if (row['dataset'], row['comparison']) == (ds, key)]
            check('aggregate_alignment', ds + ':' + key + ':one_row', len(saved_rows) == 1)
            saved = saved_rows[0]
            accepted = next(c for c in saved['controls'] if c['control'] == 'independent_global')
            check('aggregate_alignment', ds + ':' + key + ':best_epoch',
                  saved['best_epoch'] == summary['candidate']['best_epoch'])
            check('aggregate_alignment', ds + ':' + key + ':candidate_metrics',
                  saved['metrics'] == summary['metrics'])
            close('endpoint_arithmetic', ds + ':' + key + ':pair_mAP_delta',
                  direct_delta, pair['delta_metrics']['mAP'])
            close('endpoint_arithmetic', ds + ':' + key + ':pair_R1_delta',
                  rank1_delta, pair['delta_metrics']['Rank-1'])
            for name, pair_name in (('global', 'independent_global'), (key, key)):
                mean_ap = sum((q[name][0] for q in observations), Decimal(0)) * 100 / n
                rank1 = Decimal(sum(q[name][1] == 1 for q in observations)) * 100 / n
                close('endpoint_arithmetic', ds + ':' + key + ':' + name + ':mAP',
                      mean_ap, pair['metrics'][pair_name]['mAP'])
                close('endpoint_arithmetic', ds + ':' + key + ':' + name + ':R1',
                      rank1, pair['metrics'][pair_name]['Rank-1'])
            close('saved_official_float32_CMC', ds + ':' + key + ':R1',
                  Decimal(sum(q[key][1] == 1 for q in observations)) * 100 / n,
                  saved['metrics']['Rank-1'], '1e-5')
        close('accepted_totals', ds + ':' + key + ':mAP', direct_delta,
              accepted['delta_metrics']['mAP'])
        # A-U uses subtraction of original float32 official CMC; its quantization
        # differs from integer-count R1. Retain the producer's existing 1e-5 bound.
        close('accepted_R1_totals', ds + ':' + key + ':R1', rank1_delta,
              accepted['delta_metrics']['Rank-1'], '1e-5')
        check('accepted_totals', ds + ':' + key + ':repairs', accepted['rank1_repairs'] == repairs)
        check('accepted_totals', ds + ':' + key + ':new_errors', accepted['rank1_new_errors'] == harms)
        totals.append({'dataset': ds, 'comparison': key, 'queries': n,
                       'delta_map_points': direct_delta, 'rank1_repairs': repairs,
                       'rank1_new_errors': harms, 'count_based_delta_rank1_points': rank1_delta,
                       'accepted_delta_rank1_points': accepted['delta_metrics']['Rank-1']})

compare_tree(recomputed, result['rows'], 'rows')

table_lines = [line for line in texts[str(REPORT)].splitlines()
               if line.startswith(('| RGBNT201 |', '| RGBNT100 |', '| MSVR310 |'))]
expected_groups = [s for row in recomputed for s in row['strata'] + row['pooled']]
check('report_table', 'exactly_18_rows', len(table_lines) == len(expected_groups) == 18)
for index, (line, group) in enumerate(zip(table_lines, expected_groups)):
    cells = [cell.strip() for cell in line.strip('|').split('|')]
    key = 'row_' + str(index + 1)
    check('report_table', key + ':labels_and_count',
          cells[:3] == [group['dataset'], group['global_first_positive_rank_bin'], str(group['queries'])])
    values = [group['uniform']['within_bin_mean_delta_map_points'],
              group['ambiguity']['within_bin_mean_delta_map_points'],
              group['ambiguity_minus_uniform']['within_bin_mean_delta_map_points'],
              group['ambiguity_minus_uniform']['contribution_to_all_query_delta_map_points']]
    for column, value in enumerate(values, start=3):
        check('report_table', key + ':formatted_column_' + str(column), cells[column] == f'{value:+.4f}')
    contrast = group['ambiguity_minus_uniform']
    check('report_table', key + ':repairs_and_new_errors',
          cells[7] == f"{contrast['rank1_repairs']}/{contrast['rank1_new_errors']}")
check('report_wording', 'identical_json_markdown_boundary', result['boundary'] in texts[str(REPORT)])
for phrase in ('Post-development dataset-GT diagnosis only', 'identity counts are not additive',
               'within-bin AP means differ from contributions to overall mAP',
               'not the actual train-time global margin/weight', 'a deployable GT-based selector',
               'multiple-seed uncertainty', 'current same-model decomposition',
               'stronger-teacher evidence', 'a unique failure cause'):
    check('report_wording', phrase, phrase in texts[str(REPORT)])

for p in input_paths:
    check('input_immutability', str(p), hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)])

out = {
    'status': 'PASS' if not failures else 'FAIL',
    'generated_at': datetime.now().astimezone().isoformat(),
    'verifier': str(Path(__file__).resolve()),
    'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'interpreter': {'executable': sys.executable, 'version': sys.version, 'utf8_mode': sys.flags.utf8_mode},
    'method': 'Independent Decimal(50) sums and index-keyed join; producer parsed, never imported or executed.',
    'numeric_tolerances': {'arithmetic_absolute_points': '1e-10',
                           'original_float32_CMC_absolute_points': '1e-5',
                           'report_format': 'exact signed four decimal places'},
    'reviewed_hashes': hashes,
    'counts': counts,
    'total_checks': sum(c['checks'] for c in counts.values()),
    'failed_checks': len(failures),
    'failures': failures,
    'max_observed_absolute_errors': max_errors,
    'datasets': expected_counts,
    'aligned_unique_dataset_query_records': sum(expected_counts.values()),
    'saved_endpoint_query_records': sum(len(entry[2]) for entry in endpoints.values()),
    'strata_count': 12,
    'pooled_group_count': 6,
    'recomputed_rows': recomputed,
    'memberships': memberships,
    'recomputed_totals': totals,
    'bounds': {'ssh_calls': 0, 'gpu_calls': 0, 'neural_calls': 0, 'optimizer_updates': 0,
               'production_imports': 0, 'producer_executions': 0, 'source_or_result_writes': 0,
               'prior_full_audit_replays': 0, 'raw_distance_or_annotation_reads': 0},
}
(HERE / 'CHECKS.json').write_text(json.dumps(out, indent=2, default=str) + '\n', encoding='utf-8')
print(json.dumps({key: out[key] for key in ('status', 'total_checks', 'failed_checks',
                                         'aligned_unique_dataset_query_records', 'saved_endpoint_query_records')}))
print(json.dumps({'counts': counts, 'max_observed_absolute_errors': max_errors,
                  'recomputed_totals': totals}, default=str))
raise SystemExit(1 if failures else 0)

"""Independent, bounded saved-query audit. Never import or execute the producer."""
import ast
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path

BASE = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903')
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
OUT = BASE / 'review/damage915'
SOURCE = BASE / 'analyze_region_use_damage915.py'
RESULT = BASE / 'damage915/SAVED_QUERY_DAMAGE_915.json'
REPORT = BASE / 'damage915/SAVED_QUERY_DAMAGE_915.md'
MAPPING = BASE / 'FIXED_DIFFICULTY_REPOSITORY_INPUTS_914.json'
DIFFICULTY = BASE / 'FIXED_GLOBAL_DIFFICULTY_914.json'
PREVIOUS = BASE / 'review/fixed_difficulty914/AUDIT.json'
RECEIPT = BASE / 'damage915/EXECUTION_RECEIPT_915.json'
DRAFT = BASE / 'MASTER_SECTION_915_DRAFT.md'
REQUEST = OUT / 'REQUEST.md'
TOL = Fraction(1, 10**10)
CMC_TOL = Fraction(1, 10**5)
counts = Counter()
failures = []
max_errors = {}
texts = {}
hashes = {}


def read(path):
    data = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(data).hexdigest()
    texts[str(path)] = data.decode('utf-8')
    return texts[str(path)]


def read_json(path):
    return json.loads(read(path))


def check(category, label, passed, actual=None, expected=None):
    counts[category] += 1
    if not passed:
        failures.append({'category': category, 'check': label,
                         'actual': actual, 'expected': expected})


def near(category, label, actual, expected, tolerance=TOL):
    a = Fraction(actual)
    e = Fraction(expected)
    error = abs(a - e)
    max_errors[category] = max(max_errors.get(category, Fraction(0)), error)
    check(category, label, error <= tolerance, str(a), str(e))


def compare(expected, actual, label, category):
    if isinstance(expected, dict):
        check(category, label + '.keys', set(actual) == set(expected),
              sorted(actual), sorted(expected))
        for key, value in expected.items():
            compare(value, actual[key], label + '.' + str(key), category)
    elif isinstance(expected, list):
        check(category, label + '.length', len(actual) == len(expected), len(actual), len(expected))
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare(e, a, label + '[' + str(i) + ']', category)
    elif isinstance(expected, Fraction):
        near(category, label, actual, expected)
    else:
        check(category, label, actual == expected, actual, expected)


def exact_sum(values):
    return sum(values, Fraction(0))


def stats(items, n):
    # The per-query binary64 differences are the explicitly declared sign
    # convention. Rational accumulation is independent of producer math.fsum.
    vals = [q['delta'] for q in items]
    total = exact_sum(vals)
    gain = exact_sum(v for v in vals if v > 0)
    loss = exact_sum(v for v in vals if v < 0)
    return {
        'queries': len(items), 'identities': len({q['identity'] for q in items}),
        'within_group_mean_delta_ap_points': total / len(items),
        'contribution_to_all_query_delta_map_points': total / n,
        'improved_queries': sum(v > 0 for v in vals),
        'harmed_queries': sum(v < 0 for v in vals),
        'unchanged_queries': sum(v == 0 for v in vals),
        'gross_gain_contribution_points': gain / n,
        'gross_loss_contribution_points': loss / n,
    }


def serial(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


read(REQUEST)
source = read(SOURCE)
report = read(REPORT)
draft = read(DRAFT)
result = read_json(RESULT)
mapping = read_json(MAPPING)
difficulty = read_json(DIFFICULTY)
previous = read_json(PREVIOUS)
receipt = read_json(RECEIPT)
datasets = {'RGBNT201': (836, 30), 'RGBNT100': (1715, 50), 'MSVR310': (591, 52)}
conditions = ('uniform', 'ambiguity')
expected_labels = {d + '_' + c for d in datasets for c in conditions} | {'ALL_SIX_RESULTS_912.json'}
check('input_pins', 'mapping labels', set(mapping['inputs']) == expected_labels)
paths = {name: REPO / spec['repository_path'] for name, spec in mapping['inputs'].items()}
loaded = {name: read_json(path) for name, path in paths.items()}
accepted = loaded.pop('ALL_SIX_RESULTS_912.json')
expected_pins = {str(p): hashes[str(p)] for p in (SOURCE, MAPPING, DIFFICULTY, *paths.values())}
canonical_pins = {str(Path(k).resolve()): v for k, v in result['inputs'].items()}
check('input_pins', 'result exact input set',
      canonical_pins == {str(Path(k).resolve()): v for k, v in expected_pins.items()})
check('input_pins', 'result 10 input pins', len(result['inputs']) == 10)
for name, path in paths.items():
    actual_hash = hashes[str(path)]
    check('input_pins', name + '.mapping sha256', mapping['inputs'][name]['sha256'] == actual_hash)
    prior_matches = [v for k, v in previous['reviewed_hashes'].items()
                     if k.replace('\\', '/').endswith('/' + (name + '/SUMMARY.json' if name != 'ALL_SIX_RESULTS_912.json' else name))]
    check('input_pins', name + '.prior audit sha256', prior_matches == [actual_hash])
    difficulty_matches = [v for k, v in difficulty['inputs'].items()
                          if k.replace('\\', '/').endswith('/' + (name + '/SUMMARY.json' if name != 'ALL_SIX_RESULTS_912.json' else name))]
    check('input_pins', name + '.difficulty sha256', difficulty_matches == [actual_hash])
    if name != 'ALL_SIX_RESULTS_912.json':
        old_matches = [v for k, v in accepted['inputs'].items()
                       if k.replace('\\', '/').endswith('/' + name + '/SUMMARY.json')]
        check('input_pins', name + '.accepted aggregate sha256', old_matches == [actual_hash])
check('input_pins', 'difficulty prior audit pin',
      [v for k, v in previous['reviewed_hashes'].items() if k.replace('\\', '/').endswith('/FIXED_GLOBAL_DIFFICULTY_914.json')] == [hashes[str(DIFFICULTY)]])
check('prior_acceptance', 'previous WARN with no blockers', previous['verdict'] == 'WARN' and previous['blockers'] == [])
check('prior_acceptance', 'previous deterministic checks passed', previous['actual_checks']['failed_checks'] == 0)

tree = ast.parse(source, filename=str(SOURCE))
imports = sorted({(node.module if isinstance(node, ast.ImportFrom) else alias.name).split('.')[0]
                  for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                  for alias in (node.names if isinstance(node, ast.Import) else [None])})
check('source_path', 'stdlib-only producer imports', imports == ['argparse', 'collections', 'datetime', 'hashlib', 'json', 'math', 'pathlib'])
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
summary_call_lines = sorted(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'summarize')
check('source_path', 'summarize is used for whole groups identities', len(summary_call_lines) == 3)
check('source_path', 'no exec/eval/importlib dynamic execution',
      not any(isinstance(node.func, ast.Name) and node.func.id in ('exec', 'eval', '__import__') for node in calls))
check('source_path', 'explicit UTF8 text I/O',
      all(any(k.arg == 'encoding' and isinstance(k.value, ast.Constant) and k.value.value == 'utf-8' for k in node.keywords)
          for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr in ('read_text', 'write_text')))
check('result_metadata', 'status', result['status'] == 'SAVED_QUERY_AP_DAMAGE_AND_IDENTITY_CPU_DESCRIPTION')
check('result_metadata', 'zero neural and optimizer work', result['neural_calls'] == result['optimizer_updates'] == 0)
check('result_metadata', 'result timestamp has offset', datetime.fromisoformat(result['at']).tzinfo is not None)
check('result_metadata', 'all six endpoints once',
      [(r['dataset'], r['condition']) for r in result['rows']] == [(d, c) for d in datasets for c in conditions])

groups = {(True, True): 'still_correct', (True, False): 'new_error',
          (False, True): 'repair', (False, False): 'still_wrong'}
legacy_groups = {'still_correct': 'first_still_correct', 'new_error': 'new_error',
                 'repair': 'repaired', 'still_wrong': 'first_still_wrong'}
reconstructed = []
rows_by_endpoint = {}
query_maps = {}
missing_groups = []
for dataset, (n, identities_n) in datasets.items():
    fixed = next(r for r in difficulty['rows'] if r['dataset'] == dataset)
    check('query_alignment', dataset + '.accepted full N', fixed['queries'] == n)
    for condition in conditions:
        label = dataset + '_' + condition
        s = loaded[label]
        old = next(r for r in accepted['rows'] if (r['dataset'], r['comparison']) == (dataset, condition))
        check('endpoint_metadata', label + '.status', s['status'] == 'ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE')
        check('endpoint_metadata', label + '.labels', (s['dataset'], s['comparison']) == (dataset, condition))
        check('endpoint_metadata', label + '.selected best identity', old['candidate'] == s['candidate'])
        pairs = [p for p in s['pairs'] if p['control'] == 'independent_global']
        check('endpoint_metadata', label + '.one independent-global pair', len(pairs) == 1)
        pair = pairs[0]
        check('endpoint_metadata', label + '.pair labels', (pair['dataset'], pair['candidate']) == (dataset, condition))
        rows = pair['query_changes']
        check('query_alignment', label + '.N', len(rows) == n)
        check('query_alignment', label + '.contiguous unique index order', [q['query_index'] for q in rows] == list(range(n)))
        query_maps[(dataset, condition)] = {q['query_index']: q for q in rows}
        items = []
        for q in rows:
            index = str(q['query_index'])
            check('query_domain', label + '.' + index + '.identity', type(q['identity']) is int)
            for key in ('control_ap', 'candidate_ap'):
                check('query_domain', label + '.' + index + '.' + key,
                      isinstance(q[key], (int, float)) and math.isfinite(q[key]) and 0 <= q[key] <= 1)
            for key in ('control_first_rank', 'candidate_first_rank'):
                check('query_domain', label + '.' + index + '.' + key, type(q[key]) is int and q[key] >= 1)
            delta = Fraction.from_float((q['candidate_ap'] - q['control_ap']) * 100.0)
            check('exact_signs', label + '.' + index,
                  (delta > 0) == (q['candidate_ap'] > q['control_ap']) and
                  (delta < 0) == (q['candidate_ap'] < q['control_ap']) and
                  (delta == 0) == (q['candidate_ap'] == q['control_ap']))
            items.append({'index': q['query_index'], 'identity': q['identity'], 'delta': delta,
                          'global_ap': Fraction(q['control_ap']), 'candidate_ap': Fraction(q['candidate_ap']),
                          'global_correct': q['control_first_rank'] == 1, 'candidate_correct': q['candidate_first_rank'] == 1})
        whole = stats(items, n)
        check('identity_partition', label + '.identity total', whole['identities'] == identities_n)
        group_items = {name: [q for q in items if (q['global_correct'], q['candidate_correct']) == bits]
                       for bits, name in groups.items()}
        group_stats = {name: stats(qs, n) for name, qs in group_items.items() if qs}
        for name, qs in group_items.items():
            if not qs:
                missing_groups.append({'dataset': dataset, 'condition': condition, 'group': name, 'observed_queries': 0})
        identity_rows = []
        for identity in sorted({q['identity'] for q in items}):
            qs = [q for q in items if q['identity'] == identity]
            identity_rows.append(dict(identity=identity, **stats(qs, n),
                global_mean_ap=exact_sum(q['global_ap'] for q in qs) * 100 / len(qs),
                candidate_mean_ap=exact_sum(q['candidate_ap'] for q in qs) * 100 / len(qs),
                rank1_repairs=sum(not q['global_correct'] and q['candidate_correct'] for q in qs),
                rank1_new_errors=sum(q['global_correct'] and not q['candidate_correct'] for q in qs)))
        macro = exact_sum(i['within_group_mean_delta_ap_points'] for i in identity_rows) / identities_n
        worst = min(identity_rows, key=lambda i: (i['gross_loss_contribution_points'], i['identity']))
        fraction = worst['gross_loss_contribution_points'] / whole['gross_loss_contribution_points']
        reconstructed_row = dict(dataset=dataset, condition=condition, whole=whole,
            identity_macro_delta_ap_points=macro,
            improved_identities=sum(i['within_group_mean_delta_ap_points'] > 0 for i in identity_rows),
            harmed_identities=sum(i['within_group_mean_delta_ap_points'] < 0 for i in identity_rows),
            unchanged_identities=sum(i['within_group_mean_delta_ap_points'] == 0 for i in identity_rows),
            rank1_groups=group_stats, identities=identity_rows,
            largest_gross_loss_identity=dict(identity=worst['identity'], fraction_of_all_negative_ap_changes=fraction))
        observed = next(r for r in result['rows'] if (r['dataset'], r['condition']) == (dataset, condition))
        compare(reconstructed_row, observed, label, 'all_result_fields')
        rows_by_endpoint[(dataset, condition)] = reconstructed_row
        reconstructed.append(reconstructed_row)
        check('gross_loss_denominator', label + '.negative-only denominator', whole['gross_loss_contribution_points'] < 0)
        check('gross_loss_denominator', label + '.bounded share', 0 < fraction <= 1)
        check('gross_loss_denominator', label + '.different from net denominator',
              fraction != worst['gross_loss_contribution_points'] / whole['contribution_to_all_query_delta_map_points'])
        for name, group in [('whole', whole)] + list(group_stats.items()) + [(str(i['identity']), i) for i in identity_rows]:
            near('denominators', label + '.' + name + '.mean times query share',
                 group['contribution_to_all_query_delta_map_points'], group['within_group_mean_delta_ap_points'] * group['queries'] / n)
            near('gross_gain_loss', label + '.' + name + '.positive plus negative equals net',
                 group['gross_gain_contribution_points'] + group['gross_loss_contribution_points'], group['contribution_to_all_query_delta_map_points'])
            check('exact_signs', label + '.' + name + '.sign partition',
                  group['improved_queries'] + group['harmed_queries'] + group['unchanged_queries'] == group['queries'])
        for partition_name, partition in [('rank1', list(group_stats.values())), ('identity', identity_rows)]:
            check('partition_additivity', label + '.' + partition_name + '.queries', sum(g['queries'] for g in partition) == n)
            for field in ('contribution_to_all_query_delta_map_points', 'gross_gain_contribution_points', 'gross_loss_contribution_points'):
                near('partition_additivity', label + '.' + partition_name + '.' + field, exact_sum(g[field] for g in partition), whole[field])
        near('accepted_endpoint_totals', label + '.pair mAP delta', pair['delta_metrics']['mAP'], whole['contribution_to_all_query_delta_map_points'])
        near('accepted_endpoint_totals', label + '.pair global mAP', pair['metrics']['independent_global']['mAP'], exact_sum(q['global_ap'] for q in items) * 100 / n)
        near('accepted_endpoint_totals', label + '.pair candidate mAP', pair['metrics'][condition]['mAP'], exact_sum(q['candidate_ap'] for q in items) * 100 / n)
        near('accepted_endpoint_totals', label + '.accepted mAP', old['metrics']['mAP'], exact_sum(q['candidate_ap'] for q in items) * 100 / n)
        near('accepted_identity_macro', label, pair['identity_macro_mean_delta_ap_points'], macro)
        check('accepted_sign_counts', label + '.AP improved', pair['query_ap_improved'] == whole['improved_queries'])
        check('accepted_sign_counts', label + '.AP harmed', pair['query_ap_worsened'] == whole['harmed_queries'])
        old_identities = {i['identity']: i for i in pair['identity_changes']}
        check('accepted_identity_rows', label + '.identity keys', sorted(old_identities) == [i['identity'] for i in identity_rows])
        for i in identity_rows:
            old_i = old_identities[i['identity']]
            check('accepted_identity_rows', label + '.' + str(i['identity']) + '.N', old_i['queries'] == i['queries'])
            near('accepted_identity_rows', label + '.' + str(i['identity']) + '.delta', old_i['mean_delta_ap_points'], i['within_group_mean_delta_ap_points'])
        control = next(c for c in old['controls'] if c['control'] == 'independent_global')
        repairs = len(group_items['repair'])
        new_errors = len(group_items['new_error'])
        for name, old_pair in [('summary', pair), ('all_six', control)]:
            check('accepted_rank1_groups', label + '.' + name + '.repairs', old_pair['rank1_repairs'] == repairs)
            check('accepted_rank1_groups', label + '.' + name + '.new_errors', old_pair['rank1_new_errors'] == new_errors)
            near('accepted_endpoint_totals', label + '.' + name + '.mAP', old_pair['delta_metrics']['mAP'], whole['contribution_to_all_query_delta_map_points'])
            for group_name, qs in group_items.items():
                old_group = old_pair['rank1_group_ap_contributions'][legacy_groups[group_name]]
                check('accepted_rank1_groups', label + '.' + name + '.' + group_name + '.N', old_group['queries'] == len(qs))
                near('accepted_rank1_groups', label + '.' + name + '.' + group_name + '.contribution',
                     old_group['contribution_to_total_delta_map_points'], exact_sum(q['delta'] for q in qs) / n)
        near('accepted_CMC_float32', label, old['metrics']['Rank-1'], Fraction(sum(q['candidate_correct'] for q in items) * 100, n), CMC_TOL)
        for correct, pooled_label in [(True, 'global first correct'), (False, 'global first wrong')]:
            qs = [q for q in items if q['global_correct'] == correct]
            pooled = next(g for g in fixed['pooled'] if g['global_first_positive_rank_bin'] == pooled_label)
            check('difficulty_reconstruction', label + '.' + pooled_label + '.N', pooled['queries'] == len(qs))
            check('difficulty_reconstruction', label + '.' + pooled_label + '.identities', pooled['identities'] == len({q['identity'] for q in qs}))
            near('difficulty_reconstruction', label + '.' + pooled_label + '.globalAP', pooled['global_mean_ap'], exact_sum(q['global_ap'] for q in qs) * 100 / len(qs))
            near('difficulty_reconstruction', label + '.' + pooled_label + '.candidateAP', pooled[condition]['mean_ap'], exact_sum(q['candidate_ap'] for q in qs) * 100 / len(qs))
            near('difficulty_reconstruction', label + '.' + pooled_label + '.within', pooled[condition]['within_bin_mean_delta_map_points'], exact_sum(q['delta'] for q in qs) / len(qs))
            near('difficulty_reconstruction', label + '.' + pooled_label + '.contribution', pooled[condition]['contribution_to_all_query_delta_map_points'], exact_sum(q['delta'] for q in qs) / n)
            check('difficulty_reconstruction', label + '.' + pooled_label + '.repairs', pooled[condition]['rank1_repairs'] == sum(not q['global_correct'] and q['candidate_correct'] for q in qs))
            check('difficulty_reconstruction', label + '.' + pooled_label + '.new_errors', pooled[condition]['rank1_new_errors'] == sum(q['global_correct'] and not q['candidate_correct'] for q in qs))
    uq = query_maps[(dataset, 'uniform')]
    aq = query_maps[(dataset, 'ambiguity')]
    au = next(r for r in accepted['ambiguity_minus_uniform'] if r['dataset'] == dataset)
    check('accepted_query_alignment', dataset + '.AU count', len(au['query_changes']) == n)
    check('accepted_query_alignment', dataset + '.AU sequential indices', [q['query_index'] for q in au['query_changes']] == list(range(n)))
    for saved in au['query_changes']:
        index = saved['query_index']
        u, a = uq[index], aq[index]
        for field in ('identity', 'control_ap', 'control_first_rank'):
            check('query_alignment', dataset + '.' + str(index) + '.' + field, u[field] == a[field])
        for saved_field, expected in [('identity', u['identity']), ('uniform_ap', u['candidate_ap']), ('ambiguity_ap', a['candidate_ap']),
                                      ('uniform_first_rank', u['candidate_first_rank']), ('ambiguity_first_rank', a['candidate_first_rank'])]:
            check('accepted_query_alignment', dataset + '.' + str(index) + '.' + saved_field, saved[saved_field] == expected)
        near('accepted_query_delta', dataset + '.' + str(index), saved['delta_ap_points'], Fraction((a['candidate_ap'] - u['candidate_ap']) * 100.0))
    urow, arow = rows_by_endpoint[(dataset, 'uniform')], rows_by_endpoint[(dataset, 'ambiguity')]
    near('accepted_endpoint_totals', dataset + '.A-U', au['delta_metrics']['mAP'],
         arow['whole']['contribution_to_all_query_delta_map_points'] - urow['whole']['contribution_to_all_query_delta_map_points'])

check('observed_groups', 'exactly 22 observed groups', sum(len(r['rank1_groups']) for r in reconstructed) == 22)
check('observed_groups', 'only absent groups are MSVR310 repairs', missing_groups == [
    {'dataset': 'MSVR310', 'condition': c, 'group': 'repair', 'observed_queries': 0} for c in conditions])
check('identity_partition', '264 endpoint identity rows', sum(len(r['identities']) for r in reconstructed) == 264)

table = [line for line in report.splitlines() if any(line.startswith('| ' + d + ' |') for d in datasets)]
expected_table = []
for r in reconstructed:
    w = r['whole']
    expected_table.append(f"| {r['dataset']} | {r['condition']} | {w['queries']} / {w['identities']} | {float(w['contribution_to_all_query_delta_map_points']):+.4f} | {float(r['identity_macro_delta_ap_points']):+.4f} | {r['improved_identities']} / {r['harmed_identities']} / {r['unchanged_identities']} |")
for r in reconstructed:
    for name, g in r['rank1_groups'].items():
        expected_table.append(f"| {r['dataset']} | {r['condition']} | {name} | {g['queries']} | {g['improved_queries']} / {g['harmed_queries']} / {g['unchanged_queries']} | {float(g['contribution_to_all_query_delta_map_points']):+.4f} |")
check('report_table', 'exact 28 rows and no duplicate rows', len(table) == len(set(table)) == 28)
for line in expected_table:
    check('report_table', line, line in table)
check('report_wording', 'report boundary identical to JSON', result['boundary'] in report)
for phrase in ('Post-development', 'original selected-best', 'single-seed', 'fixed independent global',
               'exact saved floating-point comparisons, not statistical significance', 'equal weight per identity',
               'does not replace query mAP', 'unchanged Rank-1 does not imply unchanged AP',
               'gross-loss denominator, not net mAP', 'No same-model branch causality', 'raw-GT reauthentication'):
    check('report_wording', phrase, phrase in result['boundary'])
draft_table = [line for line in draft.splitlines() if any(line.startswith('| ' + d + ' |') for d in datasets)]
check('draft_table', 'six rows', len(draft_table) == 6)
for line in expected_table[:6]:
    check('draft_table', line, line.replace(' / ', '／') in draft_table)
u100, a100 = rows_by_endpoint[('RGBNT100', 'uniform')], rows_by_endpoint[('RGBNT100', 'ambiguity')]
sc = a100['rank1_groups']['still_correct']
claims = [
    (str(sc['queries']) + '个仍首位正确query', 'still-correct count'),
    (f"{sc['improved_queries']}个AP提高、{sc['harmed_queries']}个下降、{sc['unchanged_queries']}个不变", 'AP signs'),
    (format(float(sc['within_group_mean_delta_ap_points']), '.10f').replace('-', '−'), 'within-group delta'),
    (format(float(a100['whole']['contribution_to_all_query_delta_map_points']), '.10f').replace('-', '−'), 'net query delta'),
    (format(float(a100['identity_macro_delta_ap_points']), '.10f').replace('-', '−'), 'identity macro'),
    (str(a100['largest_gross_loss_identity']['identity']) + '占全部负AP变化', 'largest-loss identity'),
    (format(float(a100['largest_gross_loss_identity']['fraction_of_all_negative_ap_changes'] * 100), '.4f') + '%', 'gross-loss share'),
    (format(float(a100['whole']['contribution_to_all_query_delta_map_points'] - u100['whole']['contribution_to_all_query_delta_map_points']), '.10f'), 'A-U query delta'),
    (format(float(u100['identity_macro_delta_ap_points'] - a100['identity_macro_delta_ap_points']), '.10f'), 'U-A identity macro'),
    ('22个实际首位分组行与264个端侧身份条目', 'row counts'),
]
for group_name in ('still_correct', 'new_error', 'repair', 'still_wrong'):
    claims.append((format(float(a100['rank1_groups'][group_name]['contribution_to_all_query_delta_map_points']), '+.10f').replace('-', '−'), group_name + ' contribution'))
for text, label in claims:
    check('draft_numeric_claims', label, text in draft, text, 'present in draft')
for phrase in ('MSVR未观察到修复组，不补造零样本均值', '不代表统计显著性', '不是占净mAP下降的比例',
               '不等于同模型内部角色干预', 'raw GT、距离和旧远端二进制没有重新认证', '整体科学目标保持ACTIVE_UNMET'):
    check('draft_wording', phrase, phrase in draft)

cmd = receipt['command']
check('producer_receipt', 'executor records true exit0/chunk', receipt['exit_code'] == 0 and receipt['tool'] == 'exec_command' and bool(receipt['chunk_id']))
check('producer_receipt', 'honest missing exact times', receipt['exact_start_finish_timestamps'] == 'NOT_RECORDED_BY_ROOT_COMMAND_WRAPPER')
check('producer_receipt', 'working interpreter', Path(cmd[0]).resolve() == Path(sys.executable).resolve())
check('producer_receipt', 'explicit UTF8 and no pyc', cmd[1:4] == ['-X', 'utf8', '-B'])
check('producer_receipt', 'same producer path through alias', Path(cmd[4]).resolve() == SOURCE.resolve())
for flag, path in [('--repository', REPO), ('--input-map', MAPPING), ('--difficulty', DIFFICULTY), ('--output', RESULT.parent)]:
    check('producer_receipt', flag, Path(cmd[cmd.index(flag) + 1]).resolve() == path.resolve())
check('producer_receipt', 'zero work counters', receipt['neural_calls'] == receipt['optimizer_updates'] == 0)
for path, sha in hashes.items():
    check('input_immutability', path, hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha)

failed_counts = Counter(f['category'] for f in failures)
summary = {
    'status': 'PASS' if not failures else 'FAIL', 'generated_at': datetime.now().astimezone().isoformat(),
    'total_checks': sum(counts.values()), 'failed_checks': len(failures),
    'by_category': {k: {'checks': v, 'passed': v - failed_counts[k], 'failed': failed_counts[k]} for k, v in counts.items()},
    'failures': failures, 'reviewed_hashes': hashes,
    'exact_scope': {'datasets': 3, 'conditions_per_dataset': 2, 'endpoints': 6,
                    'unique_aligned_dataset_queries': sum(v[0] for v in datasets.values()),
                    'endpoint_query_records': 2 * sum(v[0] for v in datasets.values()),
                    'observed_rank1_groups': sum(len(r['rank1_groups']) for r in reconstructed),
                    'endpoint_identity_rows': sum(len(r['identities']) for r in reconstructed),
                    'report_table_rows': len(table), 'draft_table_rows': len(draft_table),
                    'missing_observed_groups': missing_groups},
    'method': 'Independent exact rational accumulation of declared binary64 per-query differences; index-keyed joins; producer read and AST-parsed, never imported or executed.',
    'numeric_tolerances': {'AP_arithmetic_absolute_points': str(float(TOL)), 'inherited_float32_CMC_absolute_points': str(float(CMC_TOL)),
                           'signs_and_counts': 'exact, no epsilon', 'report_rows': 'exact four-decimal strings'},
    'max_absolute_error_by_category': {k: float(v) for k, v in max_errors.items()},
    'source_imports': imports, 'source_summarize_call_lines': summary_call_lines,
    'producer_receipt_limit': 'Executor-supplied historical tool receipt read directly. This agent cannot independently replay/attest its prior tool event or missing exact start/finish timestamps. No producer rerun.',
    'reconstructed_rows': serial(reconstructed),
}
check_path = OUT / 'CHECKS.json'
assert not check_path.exists(), 'Preserve earlier attempts; use a new output name rather than overwrite.'
check_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: summary[k] for k in ('status', 'total_checks', 'failed_checks', 'exact_scope', 'max_absolute_error_by_category')}, ensure_ascii=False))
if failures:
    print(json.dumps(failures, ensure_ascii=False, indent=2))
sys.exit(0 if not failures else 1)

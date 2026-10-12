"""Describe saved-query AP damage; no inference, training or method selection."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repository', type=Path, required=True)
parser.add_argument('--input-map', type=Path, required=True)
parser.add_argument('--difficulty', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
input_map = json.loads(args.input_map.read_text(encoding='utf-8'))
difficulty = json.loads(args.difficulty.read_text(encoding='utf-8'))
pins = {str(args.input_map): sha(args.input_map), str(args.difficulty): sha(args.difficulty),
        str(Path(__file__)): sha(Path(__file__))}
summaries = {}
for label, item in input_map['inputs'].items():
    path = args.repository / item['repository_path']
    assert sha(path) == item['sha256']
    pins[str(path)] = item['sha256']
    if label != 'ALL_SIX_RESULTS_912.json':
        summaries[label] = json.loads(path.read_text(encoding='utf-8'))

def summarize(queries, denominator):
    values = [q['delta_ap_points'] for q in queries]
    return dict(queries=len(queries), identities=len({q['identity'] for q in queries}),
                within_group_mean_delta_ap_points=math.fsum(values) / len(queries),
                contribution_to_all_query_delta_map_points=math.fsum(values) / denominator,
                improved_queries=sum(d > 0 for d in values), harmed_queries=sum(d < 0 for d in values),
                unchanged_queries=sum(d == 0 for d in values),
                gross_gain_contribution_points=math.fsum(d for d in values if d > 0) / denominator,
                gross_loss_contribution_points=math.fsum(d for d in values if d < 0) / denominator)

rows = []
for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
    pooled = next(r for r in difficulty['rows'] if r['dataset'] == dataset)['pooled']
    baseline = None
    for condition in ('uniform', 'ambiguity'):
        summary = summaries[dataset + '_' + condition]
        assert summary['status'] == 'ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE'
        assert (summary['dataset'], summary['comparison']) == (dataset, condition)
        pair = next(p for p in summary['pairs'] if p['control'] == 'independent_global')
        queries = []
        groups = defaultdict(list)
        by_identity = defaultdict(list)
        for q in pair['query_changes']:
            delta = (q['candidate_ap'] - q['control_ap']) * 100
            assert math.isfinite(delta)
            item = dict(query_index=q['query_index'], identity=q['identity'],
                        global_ap=q['control_ap'], candidate_ap=q['candidate_ap'], delta_ap_points=delta,
                        global_first_rank=q['control_first_rank'], candidate_first_rank=q['candidate_first_rank'])
            global_correct = item['global_first_rank'] == 1
            candidate_correct = item['candidate_first_rank'] == 1
            group = ('still_correct' if candidate_correct else 'new_error') if global_correct else ('repair' if candidate_correct else 'still_wrong')
            groups[group].append(item)
            by_identity[item['identity']].append(item)
            queries.append(item)
        indices = [q['query_index'] for q in queries]
        assert indices == list(range(len(queries)))
        baseline_values = [(q['identity'], q['global_ap'], q['global_first_rank']) for q in queries]
        if condition == 'uniform':
            baseline = baseline_values
        else:
            assert baseline_values == baseline
        count = len(queries)
        whole = summarize(queries, count)
        assert abs(whole['contribution_to_all_query_delta_map_points'] - pair['delta_metrics']['mAP']) < 1e-5
        group_results = {name: summarize(group, count) for name, group in groups.items()}
        assert sum(g['queries'] for g in group_results.values()) == count
        assert abs(math.fsum(g['contribution_to_all_query_delta_map_points'] for g in group_results.values()) - whole['contribution_to_all_query_delta_map_points']) < 1e-10
        for correct, names in ((True, ('still_correct', 'new_error')), (False, ('repair', 'still_wrong'))):
            expected = next(g for g in pooled if g['global_first_positive_rank_bin'] == ('global first correct' if correct else 'global first wrong'))
            actual = [g for name, g in group_results.items() if name in names]
            assert sum(g['queries'] for g in actual) == expected['queries']
            assert abs(math.fsum(g['contribution_to_all_query_delta_map_points'] for g in actual) - expected[condition]['contribution_to_all_query_delta_map_points']) < 1e-10
        identities = []
        for identity, iq in sorted(by_identity.items()):
            result = dict(identity=identity, **summarize(iq, count))
            result['global_mean_ap'] = math.fsum(q['global_ap'] for q in iq) * 100 / len(iq)
            result['candidate_mean_ap'] = math.fsum(q['candidate_ap'] for q in iq) * 100 / len(iq)
            result['rank1_repairs'] = sum(q['global_first_rank'] != 1 and q['candidate_first_rank'] == 1 for q in iq)
            result['rank1_new_errors'] = sum(q['global_first_rank'] == 1 and q['candidate_first_rank'] != 1 for q in iq)
            identities.append(result)
        assert sum(i['queries'] for i in identities) == count
        assert abs(math.fsum(i['contribution_to_all_query_delta_map_points'] for i in identities) - whole['contribution_to_all_query_delta_map_points']) < 1e-10
        macro_delta = math.fsum(i['within_group_mean_delta_ap_points'] for i in identities) / len(identities)
        losses = sorted(identities, key=lambda i: i['gross_loss_contribution_points'])
        assert whole['gross_loss_contribution_points'] < 0
        rows.append(dict(dataset=dataset, condition=condition, whole=whole,
                         identity_macro_delta_ap_points=macro_delta,
                         improved_identities=sum(i['within_group_mean_delta_ap_points'] > 0 for i in identities),
                         harmed_identities=sum(i['within_group_mean_delta_ap_points'] < 0 for i in identities),
                         unchanged_identities=sum(i['within_group_mean_delta_ap_points'] == 0 for i in identities),
                         rank1_groups=group_results, identities=identities,
                         largest_gross_loss_identity=dict(identity=losses[0]['identity'],
                             fraction_of_all_negative_ap_changes=losses[0]['gross_loss_contribution_points'] / whole['gross_loss_contribution_points'])))

boundary = ('Post-development description of original selected-best, single-seed saved query results versus fixed independent global. '
            'AP gain/loss signs are exact saved floating-point comparisons, not statistical significance. '
            'All queries and identities retained; identity macro is equal weight per identity and does not replace query mAP. '
            'Four Rank-1 outcome groups partition queries; unchanged Rank-1 does not imply unchanged AP. '
            'Gross loss sums only negative AP changes; the largest identity fraction uses that gross-loss denominator, not net mAP. '
            'No same-model branch causality, unique failure cause, raw-GT reauthentication, deployable GT gate, NN, training, candidate or method selection.')
args.output.mkdir(parents=True, exist_ok=True)
result_path = args.output / 'SAVED_QUERY_DAMAGE_915.json'
assert not result_path.exists()
result = dict(status='SAVED_QUERY_AP_DAMAGE_AND_IDENTITY_CPU_DESCRIPTION', at=datetime.now().astimezone().isoformat(),
              inputs=pins, rows=rows, boundary=boundary, neural_calls=0, optimizer_updates=0)
result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
lines = ['# Saved-query AP damage and identity distribution', '', boundary, '',
         '| Dataset | Condition | Queries / identities | Query mAP delta | Identity macro AP delta | Identities gain / loss / unchanged |',
         '|---|---|---:|---:|---:|---:|']
for row in rows:
    w = row['whole']
    lines.append(f"| {row['dataset']} | {row['condition']} | {w['queries']} / {w['identities']} | {w['contribution_to_all_query_delta_map_points']:+.4f} | {row['identity_macro_delta_ap_points']:+.4f} | {row['improved_identities']} / {row['harmed_identities']} / {row['unchanged_identities']} |")
lines += ['', '| Dataset | Condition | Rank-1 outcome | Queries | AP gain / loss / unchanged | All-query mAP contribution |',
          '|---|---|---|---:|---:|---:|']
for row in rows:
    for name, group in row['rank1_groups'].items():
        lines.append(f"| {row['dataset']} | {row['condition']} | {name} | {group['queries']} | {group['improved_queries']} / {group['harmed_queries']} / {group['unchanged_queries']} | {group['contribution_to_all_query_delta_map_points']:+.4f} |")
(args.output / 'SAVED_QUERY_DAMAGE_915.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(dict(status=result['status'], rows=[dict(dataset=r['dataset'], condition=r['condition'], whole=r['whole'],
                   macro=r['identity_macro_delta_ap_points'], identity_gain_loss=[r['improved_identities'], r['harmed_identities']],
                   groups=r['rank1_groups'], largest_loss_identity=r['largest_gross_loss_identity']) for r in rows])))

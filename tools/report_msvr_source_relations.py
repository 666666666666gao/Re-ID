"""Verify and summarize every received source-census text row without model access."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path


KEYS = ('fold', 'state', 'view', 'output', 'protocol')
COUNTS = ('query_records', 'eligible_queries', 'ineligible_queries', 'triplets',
          'nonpositive_triplets', 'hinge_positive_triplets', 'wrong_order', 'margin_only',
          'margin_satisfied', 'prototype_correct_instance_rank1_wrong',
          'prototype_correct_instance_some_positive_wrong')
MEANS = {'mAP': 'ap_sum', 'rank1': 'rank1_sum',
         'mean_hardest_hinge': 'hardest_sum', 'mean_worst_positive_margin': 'worst_sum',
         'mean_best_positive_margin': 'best_sum'}


def fresh():
    return dict.fromkeys((*COUNTS, *MEANS.values(), 'hinge_sum'), 0)


def update(acc, row):
    eligible = row['eligible'] == 'True'
    acc['query_records'] += 1
    acc['eligible_queries'] += int(eligible)
    acc['ineligible_queries'] += int(not eligible)
    if not eligible:
        assert int(row['triplets']) == 0
        return
    for key in COUNTS[3:]:
        value = row[key]
        acc[key] += int(value == 'True') if value in ('True', 'False') else int(value)
    acc['hinge_sum'] += float(row['hinge_sum'])
    acc['ap_sum'] += 100 * float(row['average_precision'])
    acc['rank1_sum'] += 100 * (int(row['first_match_rank']) == 1)
    acc['hardest_sum'] += float(row['batch_hard_hinge'])
    acc['worst_sum'] += float(row['worst_positive_margin'])
    acc['best_sum'] += float(row['best_positive_margin'])
    assert sum(row[k] == 'True' for k in ('wrong_order', 'margin_only', 'margin_satisfied')) == 1


def finalize(acc):
    result = {k: acc[k] for k in (*COUNTS, 'hinge_sum')}
    if acc['eligible_queries']:
        result.update({k: acc[v] / acc['eligible_queries'] for k, v in MEANS.items()})
        result['mean_hinge'] = acc['hinge_sum'] / acc['triplets']
    return result


def verify(got, expected):
    for name in COUNTS:
        assert got[name] == expected[name], (name, got[name], expected[name])
    for name in ('hinge_sum', 'mean_hinge', *MEANS):
        if name in got:
            assert abs(got[name] - expected[name]) <= 1e-8 * max(1, abs(expected[name])), name


def write_csv(path, rows):
    assert rows
    with path.open('w', encoding='utf-8', newline='') as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    root = args.root
    inventory = json.loads((root / 'remote_terminal_inventory.json').read_bytes())
    assert inventory['pipeline']['status'] == 'COMPLETE_VERIFIED_SOURCE_DIAGNOSTIC'
    for name, proof in inventory['files'].items():
        data = (root / name).read_bytes()
        assert len(data) == proof['bytes'] and hashlib.sha256(data).hexdigest() == proof['sha256'], name
    cpu = json.loads((root / 'cpu_verification.json').read_bytes())
    assert cpu['status'] == 'PASS_COMPLETE_SOURCE_RELATION_CENSUS'
    expected = {tuple(str(r[k]) for k in KEYS): r for r in cpu['all972_conditions']}
    groups = defaultdict(fresh)
    identity_groups = defaultdict(fresh)
    count = 0
    with gzip.open(root / 'all_source_query_relations.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in KEYS)
            update(groups[key], row)
            update(identity_groups[key + (row['identity'],)], row)
            count += 1
    assert count == 668736 and set(groups) == set(expected) and len(groups) == 972
    for key, acc in groups.items():
        verify(finalize(acc), expected[key])
    identities = {}
    with gzip.open(root / 'all_source_identity_relations.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(str(row[k]) for k in (*KEYS, 'identity'))
            assert key not in identities
            verify(finalize(identity_groups[key]), row)
            identities[key] = row
    assert set(identities) == set(identity_groups) and len(identities) == 100440

    # Aggregate fold memberships, never features from different coordinate systems.
    totals = defaultdict(fresh)
    for key, acc in groups.items():
        for name, value in acc.items():
            totals[key[1:]][name] += value
    aggregate_rows = []
    for key, acc in sorted(totals.items()):
        row = dict(zip(KEYS[1:], key))
        row.update(finalize(acc))
        folds = [expected[(str(f),) + key] for f in range(3)]
        row['fold_mean_mAP'] = sum(r['mAP'] for r in folds) / 3
        row['fold_mean_rank1'] = sum(r['rank1'] for r in folds) / 3
        row['wrong_order_percent'] = 100 * row['wrong_order'] / row['eligible_queries']
        row['margin_only_percent'] = 100 * row['margin_only'] / row['eligible_queries']
        row['hinge_positive_triplet_percent'] = 100 * row['hinge_positive_triplets'] / row['triplets']
        aggregate_rows.append(row)
    assert len(aggregate_rows) == 324
    lookup = {tuple(row[k] for k in KEYS[1:]): row for row in aggregate_rows}
    comparisons = []
    identity_pairs = []
    for key, control in sorted(lookup.items()):
        if key[0] != 'control_final':
            continue
        style = lookup[('style_final',) + key[1:]]
        initial = lookup[('initial',) + key[1:]]
        comparisons.append(dict(view=key[1], output=key[2], protocol=key[3],
            **{f'{state}_{metric}': row[metric] for state, row in
               (('initial', initial), ('control', control), ('style', style))
               for metric in ('mAP', 'rank1', 'wrong_order', 'margin_only', 'margin_satisfied',
                              'mean_hardest_hinge', 'prototype_correct_instance_some_positive_wrong')},
            style_minus_control_mAP=style['mAP']-control['mAP'],
            style_minus_control_rank1=style['rank1']-control['rank1'],
            style_minus_control_wrong_order=style['wrong_order']-control['wrong_order']))
    for key, control in sorted(identities.items()):
        if key[1] != 'control_final':
            continue
        style = identities[(key[0], 'style_final') + key[2:]]
        row = dict(zip((*KEYS, 'identity'), key))
        row.pop('state')
        row.update(eligible_queries=control['eligible_queries'],
                   control_wrong_order=control['wrong_order'], style_wrong_order=style['wrong_order'])
        if control['eligible_queries']:
            row.update(control_mAP=control['mAP'], style_mAP=style['mAP'],
                       delta_mAP=style['mAP']-control['mAP'],
                       weighted_mAP_change=(style['mAP']-control['mAP'])*control['eligible_queries'])
        identity_pairs.append(row)
    assert len(comparisons) == 108 and len(identity_pairs) == 33480
    args.output.mkdir(exist_ok=True)
    write_csv(args.output/'all324_aggregate_conditions.csv', aggregate_rows)
    write_csv(args.output/'all108_model_comparisons.csv', comparisons)
    write_csv(args.output/'all33480_paired_identity_memberships.csv', identity_pairs)
    report = dict(status='PASS_COMPLETE_TEXT_REAGGREGATION', query_rows=count,
                  verified_conditions=len(groups), verified_identity_rows=len(identities),
                  aggregate_conditions=len(aggregate_rows), comparisons=len(comparisons),
                  paired_identity_memberships=len(identity_pairs),
                  all_input_text_sha256_verified=True, model_forwards=0, optimizer_updates=0,
                  boundary='Source memberships repeat across folds. Descriptive paired source statistics only.')
    (args.output/'verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    selected = [r for r in aggregate_rows if r['output'] in ('fused', 'pure_bank')]
    (args.output/'fused_and_bank_complete.json').write_text(json.dumps(selected, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

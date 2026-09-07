"""Replay complete terminal MSVR rankings; never load models or distance arrays.

Extracted from the executed 2026-09-07 instance-memory text reaggregation.
This executor-side deterministic check is not an independent reviewer.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


OUTPUTS = ('baseline_only', 'fused', 'cnn', 'transformer', 'mamba')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(ap, first):
    return dict(mAP=float(np.mean(ap) * 100),
                **{f'Rank-{k}': float(np.mean(np.asarray(first) <= k) * 100) for k in (1, 5, 10)})


def lower_bound(delta, labels):
    identities = np.unique(labels)
    sums = np.array([delta[labels == i].sum() for i in identities])
    counts = np.array([(labels == i).sum() for i in identities])
    draws = np.random.default_rng(42).integers(0, len(identities), (10000, len(identities)))
    return float(np.quantile(sums[draws].sum(1) / counts[draws].sum(1), .025, method='linear'))


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root = args.input_dir
    summary_path = root/'q1/summary.json'
    summary = json.loads(summary_path.read_bytes())
    cpu = json.loads((root/'q1_cpu.json').read_bytes())
    pipeline = json.loads((root/'pipeline.json').read_bytes())
    assert digest(summary_path) == args.summary_sha256 == cpu['summary_sha256'] == pipeline['terminal_summary_sha256']
    assert digest(root/'q1_cpu.json') == pipeline['terminal_cpu_sha256']
    assert cpu['status'] == args.cpu_status
    assert summary['status'] in ('Q1_PASS', 'Q1_FAIL')
    assert pipeline['status'] == 'COMPLETE_VERIFIED_' + summary['status']
    assert all(stage['exit_code'] == 0 for stage in pipeline['stages'])
    assert summary['official_image_reads'] == 0 and summary['heldout_record_forwards'] == 2064
    ends = ('control', args.candidate)
    records = {end: {out: [] for out in OUTPUTS} for end in ends}
    fold_metrics = {end: [] for end in ends}
    checked = 0
    inputs = {str(summary_path): digest(summary_path), str(root/'q1_cpu.json'): digest(root/'q1_cpu.json')}
    assert [fold['fold'] for fold in summary['folds']] == [0, 1, 2]
    for fold in summary['folds']:
        assert set(fold['endpoints']) == set(ends)
        manifests = []
        for end in ends:
            local = root/'q1'/f"fold_{fold['fold']}_{end}"
            receipt = json.loads((local/'receipt.json').read_bytes())
            assert receipt == fold['endpoints'][end]
            retrieval = receipt['retrieval']
            ranking_path = local/'rankings.json'
            inputs[str(ranking_path)] = digest(ranking_path)
            assert inputs[str(ranking_path)] == retrieval['rankings_sha256']
            rankings = json.loads(ranking_path.read_bytes())
            assert set(rankings) == set(OUTPUTS)
            gallery, queries = retrieval['gallery_manifest'], retrieval['query_rows']
            manifests.append((gallery, queries))
            per_fold = {}
            for out in OUTPUTS:
                assert len(rankings[out]) == len(queries)
                values = []
                for query, order in zip(queries, rankings[out], strict=True):
                    assert sorted(order) == list(range(len(gallery)))
                    q = gallery[query['gallery_position']]
                    assert q['index'] == query['record_index'] and q['identity'] == query['identity']
                    legal = [g for g in order if not (gallery[g]['identity'] == q['identity'] and gallery[g]['scene'] == q['scene'])]
                    positive = [i + 1 for i, g in enumerate(legal) if gallery[g]['identity'] == q['identity']]
                    negative = [g for g in legal if gallery[g]['identity'] != q['identity']]
                    assert positive and negative
                    row = dict(fold=fold['fold'], record_index=q['index'], identity=q['identity'], scene=q['scene'],
                               ap=sum((i + 1) / rank for i, rank in enumerate(positive)) / len(positive),
                               first_match_rank=positive[0], last_positive_rank=positive[-1], positive_count=len(positive),
                               nearest_negative_record=gallery[negative[0]]['index'],
                               nearest_negative_identity=gallery[negative[0]]['identity'],
                               nearest_negative_scene=gallery[negative[0]]['scene'])
                    values.append(row)
                    checked += len(order)
                ap = [r['ap'] for r in values]
                first = [r['first_match_rank'] for r in values]
                score = retrieval['outputs'][out]
                assert np.allclose(ap, score['average_precision'], rtol=0, atol=1e-14)
                assert first == score['first_match_rank']
                per_fold[out] = metrics(ap, first)
                assert all(abs(v - score['metrics'][k]) < 1e-10 for k, v in per_fold[out].items())
                records[end][out].extend(values)
            fold_metrics[end].append(per_fold)
        assert manifests[0] == manifests[1]
    assert checked == cpu['checked_retrieval_distance_and_rank_elements'] == 2069520
    labels = np.array([r['identity'] for r in records['control']['fused']])
    assert len(labels) == 600 and len(np.unique(labels)) == 60
    ap = {end: {out: np.array([r['ap'] for r in records[end][out]]) for out in OUTPUTS} for end in ends}
    result = {}
    comparison = summary['comparison']
    for end in ends:
        actual = {out: metrics(ap[end][out], [r['first_match_rank'] for r in records[end][out]]) for out in OUTPUTS}
        saved = comparison['endpoints'][end]
        assert all(abs(actual[out][k] - saved['metrics'][out][k]) < 1e-10 for out in OUTPUTS for k in actual[out])
        gains = {out: actual[out]['mAP'] - actual['baseline_only']['mAP'] for out in OUTPUTS}
        folds = [f['fused']['mAP'] - f['baseline_only']['mAP'] for f in fold_metrics[end]]
        lower = lower_bound((ap[end]['fused'] - ap[end]['baseline_only']) * 100, labels)
        gates = dict(fused_gain_at_least_1pp=gains['fused'] >= 1,
                     all_fold_fused_gains_nonnegative=all(v >= 0 for v in folds),
                     all_full_branches_not_below_signal=all(gains[out] >= 0 for out in OUTPUTS[2:]),
                     identity_bootstrap_lower_positive=lower > 0,
                     fused_strictly_best=all(actual['fused']['mAP'] > actual[out]['mAP'] for out in OUTPUTS if out != 'fused'))
        assert gates == saved['scientific_checks'] and all(gates.values()) == saved['scientific_passed']
        assert abs(lower - saved['identity_bootstrap']['lower_bound_pp']) < 1e-10
        result[end] = dict(metrics=actual, gains_over_signal=gains, fold_gains=folds, lower_bound=lower, gates=gates)
    assert np.array_equal(ap['control']['baseline_only'], ap[args.candidate]['baseline_only'])
    delta = {out: (ap[args.candidate][out] - ap['control'][out]) * 100 for out in OUTPUTS}
    gains = {out: float(delta[out].mean()) for out in OUTPUTS}
    lower = lower_bound(delta['fused'], labels)
    folds = [b['fused']['mAP'] - a['fused']['mAP'] for a, b in zip(fold_metrics['control'], fold_metrics[args.candidate], strict=True)]
    gates = dict(fused_gain_at_least_1pp=gains['fused'] >= 1, all_fold_fused_nonnegative=all(v >= 0 for v in folds),
                 all_role_gains_nonnegative=all(gains[out] >= 0 for out in OUTPUTS[2:]),
                 paired_identity_bootstrap_lower_positive=lower > 0,
                 candidate_fused_strictly_best=result[args.candidate]['gates']['fused_strictly_best'])
    assert gates == comparison['paired_checks'] and abs(lower - comparison['paired_bootstrap_lower_pp']) < 1e-10
    assert np.allclose(folds, comparison['fold_fused_gains_mAP'], rtol=0, atol=1e-10)
    assert all(abs(gains[out] - comparison['matched_gains_mAP'][out]) < 1e-10 for out in OUTPUTS)
    qualified = all(gates.values()) and all(result[args.candidate]['gates'].values())
    assert qualified == comparison['next_phase_qualified'] == (summary['status'] == 'Q1_PASS')
    query_rows, identity_rows = [], []
    for out in OUTPUTS:
        for old, new in zip(records['control'][out], records[args.candidate][out], strict=True):
            key = ('fold', 'record_index', 'identity', 'scene', 'positive_count')
            assert all(old[k] == new[k] for k in key)
            row = {k: old[k] for k in key}
            row.update(output=out, delta_ap_pp=(new['ap'] - old['ap']) * 100,
                       rank1_repaired=old['first_match_rank'] > 1 and new['first_match_rank'] == 1,
                       rank1_new_error=old['first_match_rank'] == 1 and new['first_match_rank'] > 1)
            for tag, values in (('control', old), ('candidate', new)):
                row.update({tag + '_' + k: v for k, v in values.items() if k not in key})
            query_rows.append(row)
        for identity in np.unique(labels):
            mask = labels == identity
            saved = [r for r in comparison['paired_per_identity'] if r['identity'] == identity]
            assert len(saved) == 1 and saved[0]['query_count'] == int(mask.sum())
            gain = float(delta[out][mask].mean())
            assert abs(gain - saved[0]['gains_mAP'][out]) < 1e-10
            identity_rows.append(dict(identity=int(identity), output=out, query_count=int(mask.sum()), delta_mAP_pp=gain,
                                      control_mAP=float(ap['control'][out][mask].mean() * 100),
                                      candidate_mAP=float(ap[args.candidate][out][mask].mean() * 100)))
    assert len(query_rows) == 3000 and len(identity_rows) == 300
    args.output_dir.mkdir()
    write_csv(args.output_dir/'all3000_query_output_changes.csv', query_rows)
    write_csv(args.output_dir/'all300_identity_output_changes.csv', identity_rows)
    report = dict(status='PASS_COMPLETE_TERMINAL_RANKING_TEXT', scientific_status=summary['status'],
                  inputs=inputs, candidate=args.candidate, checked_rank_positions=checked, query_count=600, unique_identities=60,
                  endpoints=result, paired_gains=gains, paired_fold_gains=folds, paired_lower_bound=lower, paired_gates=gates,
                  paired_changes={out: dict(ap_improved=sum(r['output'] == out and r['delta_ap_pp'] > 0 for r in query_rows),
                                           ap_declined=sum(r['output'] == out and r['delta_ap_pp'] < 0 for r in query_rows),
                                           ap_unchanged=sum(r['output'] == out and r['delta_ap_pp'] == 0 for r in query_rows),
                                           rank1_repaired=sum(r['output'] == out and r['rank1_repaired'] for r in query_rows),
                                           rank1_new_errors=sum(r['output'] == out and r['rank1_new_error'] for r in query_rows)) for out in OUTPUTS},
                  model_forwards=0, distance_recomputation=False, independent_reviewer=False,
                  scope='All terminal saved rank permutations, legal scene filtering, AP/CMC and fixed identity bootstrap; source OOF, seed42, repeated-development limitation remains.')
    report['csv_sha256'] = {p.name: digest(p) for p in args.output_dir.glob('*.csv')}
    (args.output_dir/'ranking_replay.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k: report[k] for k in ('status', 'scientific_status', 'checked_rank_positions', 'paired_gains', 'paired_gates', 'paired_changes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--cpu-status', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())

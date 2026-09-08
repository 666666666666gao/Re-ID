"""Aggregate complete training texts; parameter derivatives remain runtime evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ENDS = ('control', 'history_gradient')
ROLES = ('cnn', 'transformer', 'mamba')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all()
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), mean=float(a.mean()), minimum=float(a.min()),
                p05=float(np.quantile(a, .05)), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), maximum=float(a.max()))


def summarize(rows, saved):
    historical = [r for r in rows if r['memory']]
    roles = {}
    for role in ROLES:
        selected = [r['roles'][role] for r in historical]
        metrics = {}
        for name in ('total_vs_history', 'total_vs_both'):
            values = [x[name] for x in selected]
            cosines = [x['cosine'] for x in values if x['cosine'] is not None]
            metrics[name] = dict(cosine=moments(cosines),
                                 undefined_cosines=sum(x['cosine'] is None for x in values),
                                 negative_cosines=sum(x < 0 for x in cosines),
                                 first_norm=moments([x['first_norm'] for x in values]),
                                 second_norm=moments([x['second_norm'] for x in values]),
                                 difference_norm=moments([x['difference_norm'] for x in values]))
        ratios = [x['total_vs_history']['second_norm']/x['total_vs_history']['first_norm']
                  for x in selected if x['total_vs_history']['first_norm'] > 0]
        metrics['history_to_current_total_norm_ratio'] = moments(ratios)
        applied = [r['applied_gradients'][role] for r in historical]
        metrics['actually_applied_difference_norm'] = moments([x['difference_norm'] for x in applied])
        metrics['updates_with_applied_difference'] = sum(x['difference_norm'] > 0 for x in applied)
        roles[role] = metrics
    counts = ('memory_positive_pairs', 'memory_negative_pairs', 'memory_cross_scene_positive_pairs',
              'memory_negative_violations_against_batch_hard_positive', 'harder_positive_anchors',
              'harder_negative_anchors', 'current_wrong_order_anchors', 'expanded_wrong_order_anchors',
              'expanded_hinge_positive_anchors')
    ages = Counter(m['age'] for r in rows for m in r['memory'])
    available = sum(len({m['stored_step'] for m in r['memory']}) for r in rows)
    selected_groups = sum(len(r['history_vjp_groups']) for r in rows)
    return dict(updates=len(rows), historical_updates=len(historical),
                historical_current_anchor_exposures=64*len(historical), history_anchor_count=0,
                candidate_record_exposures=sum(ages.values()), ages=dict(sorted(ages.items())),
                available_history_group_exposures=available, selected_vjp_group_exposures=selected_groups,
                zero_upstream_group_skip_exposures=available-selected_groups,
                fresh_role_record_forwards=sum(r['fresh_role_record_forwards'] for r in rows),
                history_vjp_record_forwards=sum(r['history_vjp_record_forwards'] for r in rows),
                mining_exposures={key: sum(r['statistics'][key] for r in rows) for key in counts},
                current_triplet=moments([r['statistics']['current_triplet'] for r in rows]),
                expanded_triplet=moments([r['statistics']['expanded_triplet'] for r in rows]),
                total_loss=moments([r['loss'] for r in saved]),
                all14_components={key: moments([r['components'][key] for r in saved]) for key in saved[0]['components']},
                roles=roles)


def analyze(root, mode, output):
    s = json.loads((root/mode/'summary.json').read_bytes())
    cpu = json.loads((root/(mode+'_cpu.json')).read_bytes())
    assert cpu['status'] == 'PASS_COMPLETE_HISTORY_GRADIENT_'+mode.upper()
    assert cpu['summary_sha256'] == sha(root/mode/'summary.json')
    assert s['config_sha256'] == 'd03c7be1e738a206bf6db3b55580f53fbf33bfcc9050a46c4ddd0c81d7134c4e'
    assert s['seed'] == 42 and s['mode'] == mode and s['official_image_reads'] == 0
    assert s['heldout_record_forwards'] == (0 if mode == 'm0' else 2064)
    assert s['status'] in (('PASS_ENGINEERING_ONLY',) if mode == 'm0' else ('Q1_PASS', 'Q1_FAIL'))
    assert s['update_rules'] == dict(control='fresh_history_detach', history_gradient='fresh_history_candidate_vjp')
    assert s['same_candidate_reencoding_and_vjp_algorithm'] and s['history_anchors'] == 0
    endpoints, epochs, paired, total, vjp = [], [], {}, 0, 0
    items = [(f['fold'], e, r, f'fold_{f["fold"]}_{e}') for f in s['folds'] for e, r in f['endpoints'].items()]
    items += [(0, e, r, 'overfit_'+e) for e, r in s['overfit'].items()]
    for fold, end, receipt, name in items:
        local = root/mode/name
        tr = json.loads((local/'training.json').read_bytes())
        assert tr == receipt['training']
        assert sha(local/'memory_steps.jsonl') == tr['audit_files']['memory_steps.jsonl']['sha256']
        rows = [json.loads(x) for x in (local/'memory_steps.jsonl').read_text().splitlines()]
        assert len(rows) == tr['optimizer_steps'] == (100 if name.startswith('overfit') else 8 if mode == 'm0' else 260)
        assert tr['trainable_tensors'] == tr['nonzero_gradient_tensors'] == 203
        assert tr['overflow_events'] == 0 and not tr['missing_nonzero_gradients']
        assert tr['initial_state_sha256'] == receipt['initialization']['initial_state_sha256']
        assert all(receipt['checks' if name.startswith('overfit') else 'engineering_checks'].values())
        for row, saved in zip(rows, tr['steps'], strict=True):
            assert row['step'] == saved['step'] and row['record_indices'] == saved['sampled_record_indices']
            assert row['coordinate_rule'] == 'fresh' and row['current_anchor_count'] == 64 and row['history_anchor_count'] == 0
            assert row['history_candidate_vjp_applied'] == (end == 'history_gradient') and row['gradient_weight'] == 1
            assert len(saved['components']) == 14 and saved['amp_scale_after'] >= saved['amp_scale_before']
            expected = row['statistics']['expanded_triplet'] if row['replacement_active'] else row['original_triplet']
            assert abs(saved['components']['triplet_fused']-expected) < 2e-6
            assert row['selected_reencoding_bitwise'] and row['history_rng_buffers_preserved']
            assert row['history_vjp_leaves_current_grad_unchanged'] and row['final_gradient_addition_bitwise']
            assert len(row['historical_leaf_upstream_norms']) == len(row['memory'])
            groups = sorted({m['stored_step'] for m, n in zip(row['memory'], row['historical_leaf_upstream_norms'], strict=True) if n > 0})
            assert groups == row['history_vjp_groups'] and row['history_vjp_record_forwards'] == 64*len(groups)
            for role in ROLES:
                applied = row['applied_gradients'][role]
                if end == 'control':
                    assert applied['difference_norm'] == 0 and applied['first_norm'] == applied['second_norm']
                else:
                    assert applied == row['roles'][role]['total_vs_both']
        aggregate = summarize(rows, tr['steps'])
        assert aggregate['fresh_role_record_forwards']+64 == tr['extra_fresh_role_record_forwards']
        assert aggregate['history_vjp_record_forwards'] == tr['extra_history_vjp_record_forwards']
        endpoints.append(dict(fold=fold, endpoint=end, mode=tr['mode'], **aggregate,
                              peak_allocated_mib=tr['peak_allocated_mib'],
                              training_seconds=sum(x['elapsed_seconds'] for x in tr['history'])))
        key = ('overfit' if name.startswith('overfit') else fold, end)
        paired[key] = (tr['initial_state_sha256'], [(r['record_indices'], r['pixel_sha256'], r['memory'], r['fresh_role_record_forwards']) for r in rows])
        for epoch in tr['history']:
            pairs = [(r, t) for r, t in zip(rows, tr['steps'], strict=True) if t['epoch'] == epoch['epoch']]
            assert len(pairs) == epoch['optimizer_steps']
            epochs.append(dict(fold=fold, endpoint=end, mode=tr['mode'], epoch=epoch['epoch'],
                               **summarize([x[0] for x in pairs], [x[1] for x in pairs])))
        total += len(rows);vjp += aggregate['history_vjp_record_forwards']
    for fold in (0, 1, 2, 'overfit') if mode == 'm0' else (0, 1, 2):
        assert paired[(fold, ENDS[0])] == paired[(fold, ENDS[1])]
    assert total == s['optimizer_steps'] == cpu['checked_training_steps'] == (248 if mode == 'm0' else 1560)
    assert vjp == cpu['checked_history_vjp_record_forwards']
    result = dict(status='PASS_COMPLETE_HISTORY_GRADIENT_TRAINING_TEXT', mode=mode,
                  total_updates=total, summary_sha256=sha(root/mode/'summary.json'),
                  cpu_sha256=sha(root/(mode+'_cpu.json')), endpoints=endpoints, all_epochs=epochs,
                  matched_indices_pixels_initialization_history_metadata_and_fresh_forwards=True,
                  actual_vjp_work_may_differ=True, gradients_scope='Same role encoder block: all14-task current gradient versus historical fused-Triplet candidate gradient; runtime summaries, not independent model backward.',
                  counts_scope='Repeated batch/anchor/candidate exposures, not independent relationships or heldout retrieval errors.',
                  scientific_qualification=s['comparison']['next_phase_qualified'] if mode == 'q1' else None)
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('endpoints', 'all_epochs')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('m0', 'q1'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input_dir, args.mode, args.output)

"""Reaggregate every terminal coordinate-comparison text row, without models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    if len(values) == 0:
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()), minimum=float(values.min()),
                p05=float(np.quantile(values, .05)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def summarize(rows):
    history = [r for r in rows if r['memory']]
    roles = {}
    for role in ('cnn', 'transformer', 'mamba'):
        values = [r['coordinate_parameter_gradients'][role] for r in history]
        cosine = [r['fresh']['cosine'] for r in values]
        roles[role] = dict(
            cosine=moments([x for x in cosine if x is not None]),
            undefined_cosines=sum(x is None for x in cosine),
            negative_cosines=sum(x is not None and x < 0 for x in cosine),
            difference_norm=moments([r['fresh']['difference_norm'] for r in values]),
            repeated_noise_norm=moments([r['duplicate_noise']['difference_norm'] for r in values]),
            differences_above_noise=sum(r['fresh']['difference_norm'] > r['duplicate_noise']['difference_norm'] for r in values))
    deltas = [r['fresh_statistics']['expanded_triplet']-r['stale_statistics']['expanded_triplet'] for r in history]
    age = Counter(v['age'] for r in rows for v in r['memory'])
    return dict(updates=len(rows), historical_updates=len(history), historical_anchor_exposures=64*len(history),
                historical_candidate_exposures=sum(age.values()), ages=dict(sorted(age.items())),
                fresh_minus_stale_expanded_loss=moments(deltas),
                positive_loss_difference_updates=sum(v > 0 for v in deltas),
                negative_loss_difference_updates=sum(v < 0 for v in deltas),
                cache_l2_drift=moments([v for r in history for v in r['cache_l2_drift']]),
                roles=roles, reencoded_role_record_forwards=sum(r['fresh_role_record_forwards'] for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('m0', 'q1'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, mode = args.input_dir, args.mode
    summary = json.loads((root/mode/'summary.json').read_bytes())
    cpu = json.loads((root/(mode+'_cpu.json')).read_bytes())
    assert cpu['status'] == 'PASS_COMPLETE_FRESH_COORDINATE_'+mode.upper()
    assert cpu['summary_sha256'] == sha(root/mode/'summary.json')
    assert summary['config_sha256'] == '32e22d3a0cd858e96430e4b9f9b0a2093c0fdf30f026d8d334ad92d46cfb94aa'
    assert summary['mode'] == mode and summary['seed'] == 42
    assert summary['status'] in (('PASS_ENGINEERING_ONLY',) if mode == 'm0' else ('Q1_PASS', 'Q1_FAIL'))
    assert summary['official_image_reads'] == 0
    assert summary['heldout_record_forwards'] == (0 if mode == 'm0' else 2064)
    assert summary['update_rules'] == dict(control='stale_history', fresh_memory='current_reencoded_history')
    assert summary['reencoding_compute_matched'] and summary['historical_gradients_detached']
    endpoints, paired, total, epochs = [], {}, 0, []
    for fold in summary['folds']:
        assert set(fold['endpoints']) == {'control', 'fresh_memory'}
        for end, result in fold['endpoints'].items():
            local = root/mode/f"fold_{fold['fold']}_{end}"
            tr = result['training']
            assert tr == json.loads((local/'training.json').read_bytes())
            assert sha(local/'memory_steps.jsonl') == tr['audit_files']['memory_steps.jsonl']['sha256']
            rows = [json.loads(line) for line in (local/'memory_steps.jsonl').read_text().splitlines()]
            assert len(rows) == tr['optimizer_steps'] == (8 if mode == 'm0' else 260)
            assert all(result['engineering_checks'].values())
            assert tr['initial_state_sha256'] == result['initialization']['initial_state_sha256']
            assert tr['trainable_tensors'] == tr['nonzero_gradient_tensors'] == 203
            assert tr['missing_nonzero_gradients'] == [] and tr['overflow_events'] == 0
            for row, saved in zip(rows, tr['steps'], strict=True):
                assert row['step'] == saved['step']
                assert row['record_indices'] == saved['sampled_record_indices']
                assert row['historical_features_detached'] and row['zero_age_reencoding_bitwise']
                assert row['coordinate_rule'] == ('fresh' if end == 'fresh_memory' else 'stale')
                chosen = row['fresh_statistics'] if end == 'fresh_memory' else row['stale_statistics']
                assert row['statistics'] == chosen
                expected = chosen['expanded_triplet'] if row['replacement_active'] else row['original_triplet']
                assert abs(saved['components']['triplet_fused']-expected) < 2e-6
                assert saved['amp_scale_after'] >= saved['amp_scale_before']
                assert len(row['cache_l2_drift']) == len(row['memory'])
            stats = summarize(rows)
            assert stats['reencoded_role_record_forwards']+64 == tr['extra_fresh_role_record_forwards']
            endpoints.append(dict(fold=fold['fold'], endpoint=end, **stats,
                                  initial_state_sha256=tr['initial_state_sha256'],
                                  extra_zero_age_record_forwards=64, peak_allocated_mib=tr['peak_allocated_mib']))
            paired[(fold['fold'], end)] = [(r['record_indices'], r['pixel_sha256'], r['memory'], r['fresh_role_record_forwards']) for r in rows]
            for epoch in tr['history']:
                selected = [r for r, saved in zip(rows, tr['steps'], strict=True) if saved['epoch'] == epoch['epoch']]
                assert len(selected) == epoch['optimizer_steps']
                epochs.append(dict(fold=fold['fold'], endpoint=end, epoch=epoch['epoch'], **summarize(selected)))
            total += len(rows)
        a, b = (fold['endpoints'][e]['training'] for e in ('control', 'fresh_memory'))
        assert a['initial_state_sha256'] == b['initial_state_sha256']
        assert paired[(fold['fold'], 'control')] == paired[(fold['fold'], 'fresh_memory')]
    overfit = {}
    if mode == 'm0':
        assert set(summary['overfit']) == {'control', 'fresh_memory'}
        for end, result in summary['overfit'].items():
            local = root/mode/('overfit_'+end)
            tr = result['training']
            assert tr == json.loads((local/'training.json').read_bytes())
            assert sha(local/'memory_steps.jsonl') == tr['audit_files']['memory_steps.jsonl']['sha256']
            rows = [json.loads(line) for line in (local/'memory_steps.jsonl').read_text().splitlines()]
            assert len(rows) == tr['optimizer_steps'] == 100 and all(result['checks'].values())
            assert all(r['memory'] == [] for r in rows)
            assert tr['extra_fresh_role_record_forwards'] == 64
            overfit[end] = dict(gate=result['gate'], checks=result['checks'], **summarize(rows))
            paired[('overfit', end)] = [(r['record_indices'], r['pixel_sha256'], r['memory'], r['fresh_role_record_forwards']) for r in rows]
            total += 100
        assert paired[('overfit', 'control')] == paired[('overfit', 'fresh_memory')]
        assert summary['overfit']['control']['training']['initial_state_sha256'] == summary['overfit']['fresh_memory']['training']['initial_state_sha256']
    assert total == summary['optimizer_steps'] == cpu['checked_training_steps'] == (248 if mode == 'm0' else 1560)
    assert len(epochs) == (6 if mode == 'm0' else 120)
    output = dict(status='PASS_COMPLETE_FRESH_COORDINATE_TEXT', mode=mode, total_updates=total,
                  summary_sha256=sha(root/mode/'summary.json'), cpu_sha256=sha(root/(mode+'_cpu.json')),
                  endpoints=endpoints, all_epochs=epochs, overfit=overfit,
                  raw_runtime_gradient_witnesses=True, independent_model_backpropagation=False,
                  all_paired_indices_pixels_memory_and_reencoding_counts_equal=True,
                  scientific_qualification=summary['comparison']['next_phase_qualified'] if mode == 'q1' else None)
    assert not args.output.exists()
    args.output.write_bytes((json.dumps(output, indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in output.items() if k not in ('endpoints','all_epochs','overfit')}))


if __name__ == '__main__':
    main()

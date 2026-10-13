"""Decompose saved G-to-F cosine changes; no model or retrieval evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(x):
    assert x.ndim == 1 and len(x) and torch.isfinite(x).all()
    return dict(count=len(x), mean=float(x.mean()), mean_absolute=float(x.abs().mean()),
                minimum=float(x.min()), maximum=float(x.max()))


def decompose(query, gallery, gain):
    gq, cq, hq, fq = [query[k].double() for k in ('shared_global', 'correction', 'raw_fused', 'fused')]
    gg, cg, hg, fg = [gallery[k].double() for k in ('shared_global', 'correction', 'raw_fused', 'fused')]
    assert all(x.shape == (836, 1536) and torch.isfinite(x).all()
               for x in (gq, cq, hq, fq, gg, cg, hg, fg))
    iq, ig = gq + gain * cq, gg + gain * cg
    raw_error = max(float((hq-iq).abs().max()), float((hg-ig).abs().max()))
    assert raw_error <= 1e-6
    dq, dg = cq.mean(0), cg.mean(0)
    rq, rg = cq-dq, cg-dg
    ng = gq.norm(dim=1)[:, None] * gg.norm(dim=1)[None]
    ah = hq.norm(dim=1)[:, None] * hg.norm(dim=1)[None]
    assert (ng > 0).all() and (ah > 0).all()
    gg_dot = gq @ gg.T
    sg, si, sh = gg_dot/ng, (iq @ ig.T)/ah, (hq @ hg.T)/ah
    terms = {
        'global_denominator_change': gg_dot * (ah.reciprocal()-ng.reciprocal()),
        'global_common_correction_cross': gain * ((gq @ dg)[:, None] + (gg @ dq)[None])/ah,
        'global_varying_correction_cross': gain * (gq @ rg.T + rq @ gg.T)/ah,
        'common_correction_pair': gain**2 * (dq @ dg)/ah,
        'common_varying_correction_cross': gain**2 * ((rq @ dg)[:, None] + (rg @ dq)[None])/ah,
        'varying_correction_pair': gain**2 * (rq @ rg.T)/ah,
        'saved_raw_fp32_rounding': sh-si,
    }
    error = float((sum(terms.values())-(sh-sg)).abs().max())
    assert error <= 1e-12
    saved_f_error = float((fq @ fg.T-sh).abs().max())
    assert saved_f_error <= 1e-6
    return sg, sh, terms, dict(raw_reconstruction_max_abs_error=raw_error,
        cosine_term_closure_max_abs_error=error,
        saved_f_cosine_vs_raw_cosine_max_abs_error=saved_f_error)


def analyze(row, protocol):
    diagnosis = json.loads(Path(row['diagnosis']).read_text())
    assert diagnosis['status'] == 'COMPLETE' and diagnosis['selected_epoch'] == 8
    query, gallery = [torch.load(row[split], map_location='cpu', weights_only=True)
                      for split in ('query', 'gallery')]
    assert set(query) == set(gallery) == {'shared_global', 'correction', 'raw_fused', 'fused'}
    data = torch.load(row['distances'], map_location='cpu', weights_only=False)
    for split in ('query', 'gallery'):
        for field, suffix in (('identity', 'ids'), ('camera', 'cameras'), ('scene', 'scenes')):
            expected = torch.tensor([r[field] for r in protocol['records'][split]])
            assert torch.equal(torch.as_tensor(data[split+'_'+suffix]), expected)
    qi, gi = [torch.as_tensor(data[k]) for k in ('query_ids', 'gallery_ids')]
    qc, gc = [torch.as_tensor(data[k]) for k in ('query_cameras', 'gallery_cameras')]
    same = qi[:, None] == gi[None]
    positive, negative = same & (qc[:, None] != gc[None]), ~same
    assert positive.any(1).all() and negative.any(1).all()
    legal = positive | negative
    sg, sh, terms, closure = decompose(query, gallery, diagnosis['readout_gain'])
    # This is parity with the original normalized squared-Euclidean arrays,
    # not a new distance/ranking or a replacement for original score receipts.
    for name, cosine in (('global', sg), ('fused', sh)):
        error = float((cosine-(1-data[name].double()/2)).abs().max())
        assert error <= 1e-5
        closure[name+'_saved_distance_cosine_max_abs_error'] = error
    negative_index = data['global'].masked_fill(~negative, float('inf')).argmin(1)
    # One fixed hardest legal negative, selected from the already saved G array.
    # All legal cross-camera positives are retained; ties use the lowest index.
    arange = torch.arange(836)
    def margins(matrix):
        return matrix-matrix[arange, negative_index][:, None]
    before, change = margins(sg), margins(sh-sg)
    component_changes = {name: margins(value) for name, value in terms.items()}
    query_changes = {name: (value*positive).sum(1)/positive.sum(1)
                     for name, value in component_changes.items()}
    query_total = (change*positive).sum(1)/positive.sum(1)
    assert torch.allclose(sum(query_changes.values()), query_total, atol=1e-12, rtol=1e-10)
    old, new = [torch.tensor(diagnosis['scores'][k]['first_match_rank'])
                for k in ('global', 'fused')]
    groups = {'all': torch.ones(836, dtype=torch.bool),
              'repair': (old != 1) & (new == 1),
              'new_error': (old == 1) & (new != 1),
              'kept_correct': (old == 1) & (new == 1),
              'kept_wrong': (old != 1) & (new != 1)}
    assert sum(int(groups[k].sum()) for k in ('repair','new_error','kept_correct','kept_wrong')) == 836
    return {'exposure': row['exposure'], 'selected_epoch': 8,
        'gain': diagnosis['readout_gain'], 'closure': closure,
        'counts': {'queries': 836, 'gallery': 836, 'legal_positive_pairs': int(positive.sum()),
                   'legal_negative_pairs': int(negative.sum()),
                   'removed_same_identity_same_camera_pairs': int((same & ~positive).sum())},
        'legal_pair_cosine_changes': {'total': describe((sh-sg)[legal]),
             'terms': {name: describe(value[legal]) for name, value in terms.items()}},
        'fixed_negative_margin_pairs': {
            label: {'total': describe(change[mask]),
                    'terms': {name: describe(value[mask]) for name, value in component_changes.items()}}
            for label, mask in {'all': positive,
                  'global_margin_positive': positive & (before > 0),
                  'global_margin_nonpositive': positive & (before <= 0)}.items()},
        'equal_query_weighted_margins': {
            label: {'total': describe(query_total[mask]),
                    'terms': {name: describe(value[mask]) for name, value in query_changes.items()}}
            for label, mask in groups.items()},
        'prior_unchanged_metrics': {k: diagnosis['scores'][k]['metrics'] for k in ('global','fused')},
        'prior_unchanged_global_to_fused': diagnosis['comparisons']['same_model_global_to_fused']['delta_metrics']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not args.output.exists()
    torch.set_num_threads(4)
    started = time.perf_counter()
    seal = json.loads(args.seal.read_text())
    assert sha(Path(__file__)) == seal['source_sha256']
    assert all(sha(Path(p)) == h for p, h in seal['artifact_sha256'].items())
    protocol = json.loads(Path(seal['protocol']).read_text())
    assert protocol['dataset'] == 'RGBNT201'
    result = {'schema': 'saved-evidence-cosine-terms-v1', 'evaluation_type': 'real_gt_descriptive_relation_probe',
        'source_sha256': sha(Path(__file__)), 'seal_sha256': sha(args.seal),
        'neural_forward_calls': 0, 'optimizer_updates': 0, 'retrieval_rankings_computed': 0,
        'new_retrieval_performance_metrics': 0, 'rows': [analyze(row, protocol) for row in seal['rows']],
        'boundary': 'Two fixed E8 RGBNT201 seed42 arms. Exact additive cosine accounting, not causal intervention. Test means only label terms, never calibrate embeddings or rank. All negatives retained when selecting the fixed G-hardest negative; all legal cross-camera positives retained. Probe margins are not AP and omit other negatives. Query/gallery overlap. Components share the observed denominator, so contributions are convention-dependent; large term magnitudes can cancel. Existing rank1 groups and metrics are copied, not recomputed. No new model, teacher, weight, hyperparameter selection, evaluation promotion or training.'}
    assert all(sha(Path(p)) == h for p, h in seal['artifact_sha256'].items())
    assert sha(Path(__file__)) == seal['source_sha256']
    result['elapsed_seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'COMPLETE', 'rows': [{k: row[k] for k in ('exposure','counts','closure','equal_query_weighted_margins')} for row in result['rows']]}), flush=True)

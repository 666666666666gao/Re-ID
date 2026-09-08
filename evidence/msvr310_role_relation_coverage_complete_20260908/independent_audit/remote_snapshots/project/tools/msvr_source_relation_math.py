"""Source-only relation census on frozen, unit-normalized embeddings."""
import numpy as np

EXPERTS = ('cnn', 'transformer', 'mamba')
MODALITIES = ('RGB', 'NI', 'TI')
WIDTHS = {'baseline_only': 3072, 'fused': 7680, **{e: 4608 for e in EXPERTS},
          **{f'{e}_{m}_residual': 512 for e in EXPERTS for m in MODALITIES},
          'pure_bank': 4608, **{f'pure_{e}': 1536 for e in EXPERTS}}
STATES = ('initial', 'control_final', 'style_final')
VIEWS = ('clean', 'augmented', 'coupled_style')
PROTOCOLS = ('identity_exclude_record', 'cross_scene')


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (norms > 0).all()
    return x / norms


def source_rows(query, gallery, identities, scenes, protocol):
    """Every source record remains a candidate, including ineligible queries.

    Positive/negative counts enumerate all real instances. Sorted-prefix sums
    give the exact full-triplet hinge sum without materializing all triplets.
    """
    assert protocol in PROTOCOLS and query.shape == gallery.shape
    query, gallery = unit(query), unit(gallery)
    ids, scenes = np.asarray(identities), np.asarray(scenes)
    classes = np.unique(ids)
    prototypes = unit(np.stack([gallery[ids == i].sum(0) for i in classes]))
    matrix = query @ gallery.T
    prototype_matrix = query @ prototypes.T
    for q in range(len(query)):
        same = ids == ids[q]
        positive = same.copy()
        positive[q] = False
        if protocol == 'cross_scene':
            positive &= scenes != scenes[q]
        negative = ~same
        p, n = matrix[q, positive], matrix[q, negative]
        row = dict(query_position=q, identity=int(ids[q]), scene=int(scenes[q]),
                   eligible=bool(len(p)), positive_records=len(p), negative_records=len(n),
                   triplets=len(p)*len(n))
        if not len(p):
            yield row
            continue
        assert len(n) > 0
        dp = np.sqrt(np.maximum(0., 2-2*p))
        dn = np.sort(np.sqrt(np.maximum(0., 2-2*n)))
        boundary = dp + .3
        counts = np.searchsorted(dn, boundary, side='left')
        prefix = np.concatenate(([0.], np.cumsum(dn)))
        hinges = counts*boundary-prefix[counts]
        inversions = int((len(n)-np.searchsorted(np.sort(n), p, side='left')).sum())
        pp = unit(gallery[positive].sum(0, keepdims=True))[0]
        proto_margin = float(query[q] @ pp - prototype_matrix[q, classes != ids[q]].max())
        order = np.argsort(-matrix[q], kind='stable')
        legal = order[(positive | negative)[order]]
        ranks = np.flatnonzero(ids[legal] == ids[q])+1
        assert len(ranks) == len(p)
        best_margin = float(p.max()-n.max())
        worst_margin = float(p.min()-n.max())
        row.update(best_positive_margin=best_margin, worst_positive_margin=worst_margin,
                   mean_triplet_margin=float(p.mean()-n.mean()), nonpositive_triplets=inversions,
                   hinge_positive_triplets=int(counts.sum()), hinge_sum=float(hinges.sum()),
                   hinge_mean=float(hinges.sum()/row['triplets']),
                   batch_hard_hinge=float(max(0., dp.max()-dn.min()+.3)),
                   average_precision=float(np.mean(np.arange(1,len(ranks)+1)/ranks)),
                   first_match_rank=int(ranks[0]), prototype_margin=proto_margin,
                   prototype_correct_instance_rank1_wrong=bool(proto_margin > 0 and best_margin <= 0),
                   prototype_correct_instance_some_positive_wrong=bool(proto_margin > 0 and worst_margin <= 0),
                   wrong_order=bool(worst_margin <= 0),
                   margin_only=bool(worst_margin > 0 and counts.sum() > 0),
                   margin_satisfied=bool(counts.sum() == 0),
                   nearest_negative_position=int(np.flatnonzero(negative)[n.argmax()]),
                   hardest_positive_position=int(np.flatnonzero(positive)[p.argmin()]))
        yield row


def reference_check(query, gallery, identities, scenes, protocol, rows):
    """Independent vector-distance/brute-triplet witness, including zero positives."""
    qf, gf = unit(query), unit(gallery)
    maximum = 0.
    for q, row in enumerate(rows):
        p = [j for j in range(len(gf)) if j != q and identities[j] == identities[q]
             and (protocol != 'cross_scene' or scenes[j] != scenes[q])]
        n = [j for j in range(len(gf)) if identities[j] != identities[q]]
        assert row['positive_records'] == len(p) and row['negative_records'] == len(n)
        if not p:
            assert not row['eligible']
            continue
        sp = qf[q] @ gf[p].T
        sn = qf[q] @ gf[n].T
        margins = sp[:,None]-sn[None]
        dp = np.linalg.norm(qf[q]-gf[p], axis=1)
        dn = np.linalg.norm(qf[q]-gf[n], axis=1)
        hinge = np.maximum(0., dp[:,None]-dn[None]+.3)
        assert row['triplets'] == hinge.size
        assert row['nonpositive_triplets'] == int((margins <= 0).sum())
        assert row['hinge_positive_triplets'] == int((hinge > 0).sum())
        error = abs(row['hinge_sum']-float(hinge.sum()))
        maximum = max(maximum, error)
        assert error < 1e-10
    return maximum


def math_check():
    rng = np.random.default_rng(42)
    gallery = unit(rng.normal(size=(8,7)))
    query = unit(gallery+.1*rng.normal(size=gallery.shape))
    ids = np.array([0,0,0,1,1,2,2,3])
    scenes = np.array([0,0,1,0,0,0,1,0])
    errors = []
    for protocol in PROTOCOLS:
        rows = list(source_rows(query,gallery,ids,scenes,protocol))
        errors.append(reference_check(query,gallery,ids,scenes,protocol,rows))
        assert len(rows) == 8
        assert sum(x['eligible'] for x in rows) == (7 if protocol == PROTOCOLS[0] else 5)
        assert all(x['negative_records'] > 0 for x in rows)
    return dict(status='PASS_EXACT_FULL_RELATION_MATH',maximum_brute_hinge_error=max(errors),
                class_zero_valid=True,noneligible_queries_retained_in_gallery=True)

"""NumPy definitions for the V29 source-only role drift diagnostic.

No retrieval scores, model selection, or optimizer. Pairwise matrices are always
computed inside one model; comparisons subtract scalar relations, not vectors
from different folds or identity coordinate systems.
"""
import numpy as np

STATES = ("initial", "control_final", "bounded_final")
VIEWS = ("original", "registered_style")
EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")
OUTPUTS = ("baseline_only", "fused", *EXPERTS,
           *(f"{e}_{m}_residual" for e in EXPERTS for m in MODALITIES),
           "pure_bank", *(f"pure_{e}" for e in EXPERTS))
COMPARISONS = ("initial_to_control", "initial_to_bounded", "control_to_bounded")
STATE_PAIRS = ((0, 1), (0, 2), (1, 2))
VECTOR_COMPARISONS = (*COMPARISONS, "bounded_h_to_y")
VECTOR_FIELDS = ("left2", "right2", "dot", "difference2", "cosine", "chord")
PROTOCOLS = ("identity", "cross_camera")
CHANGE_FIELDS = ("before_margin_sum", "after_margin_sum", "absolute_margin_change_sum",
                 "margin_decreased_count", "lost_correct_count", "repaired_count",
                 "before_nonpositive_count", "after_nonpositive_count",
                 "before_hinge_sum", "after_hinge_sum",
                 "before_hinge_positive_count", "after_hinge_positive_count")


def indices(labels, cameras, protocol):
    labels, cameras = np.asarray(labels), np.asarray(cameras)
    assert protocol in PROTOCOLS
    same = labels[:, None] == labels[None, :]
    positive = same & ~np.eye(len(labels), dtype=bool)
    if protocol == "cross_camera":
        positive &= cameras[:, None] != cameras[None, :]
    q, p, n = np.nonzero(positive[:, :, None] & ~same[:, None, :])
    assert len(q) > 0
    return q, p, n


def margin_and_hinge(matrix, q, p, n):
    a = np.asarray(matrix, dtype=np.float64)
    positive, negative = a[..., q, p], a[..., q, n]
    margin = positive - negative
    hinge = np.maximum(0, np.sqrt(np.maximum(0, 2 - 2 * positive))
                       - np.sqrt(np.maximum(0, 2 - 2 * negative)) + .3)
    return margin, hinge


def change_sums(before, after, before_hinge, after_hinge):
    assert before.shape == after.shape == before_hinge.shape == after_hinge.shape
    return np.stack((before.sum(-1), after.sum(-1), np.abs(after-before).sum(-1),
                     (after < before).sum(-1), ((before > 0) & (after <= 0)).sum(-1),
                     ((before <= 0) & (after > 0)).sum(-1),
                     (before <= 0).sum(-1), (after <= 0).sum(-1),
                     before_hinge.sum(-1), after_hinge.sum(-1),
                     (before_hinge > 0).sum(-1), (after_hinge > 0).sum(-1)), axis=-1)


def case_relations(matrices, labels, cameras, protocol):
    """All ordered registered batch relations; no top-k or hard-example subset."""
    assert matrices.shape == (3, 2, 18, len(labels), len(labels))
    q, p, n = indices(labels, cameras, protocol)
    margins, hinges = margin_and_hinge(matrices, q, p, n)
    changes = np.stack([change_sums(margins[a], margins[b], hinges[a], hinges[b])
                        for a, b in STATE_PAIRS])  # comparison, view, output, field
    stable = []
    stable_by_identity = []
    for comparison, (a, b) in enumerate(STATE_PAIRS):
        reliable = (hinges[a] == 0).all(axis=0) & (margins[a] > 0).all(axis=0)
        lost = reliable[None] & (margins[b] <= 0)
        weakened = reliable[None] & (hinges[b] > 0)
        stable.append(np.stack((np.broadcast_to(reliable.sum(-1), (2, 18)),
                                lost.sum(-1), weakened.sum(-1)), axis=-1))
        for identity in np.unique(labels):
            selected = np.asarray(labels)[q] == identity
            stable_by_identity.append({
                "comparison": comparison, "identity": int(identity), "triplets": int(selected.sum()),
                "reliable_count_per_output": reliable[:, selected].sum(-1).tolist(),
                "lost_per_view_output": lost[..., selected].sum(-1).tolist(),
                "hinge_violated_per_view_output": weakened[..., selected].sum(-1).tolist(),
            })
    # Exact same-forward scalar decomposition, not a new embedding or retrieval evaluation.
    unmodified = .5 * matrices[2, :, 0] + .5 * matrices[2, :, 14]
    um, uh = margin_and_hinge(unmodified, q, p, n)
    joint = change_sums(um, margins[2, :, 1], uh, hinges[2, :, 1])
    joint_reliable = (uh == 0).all(0) & (um > 0).all(0)
    joint_lost = joint_reliable[None] & (margins[2, :, 1] <= 0)
    joint_weakened = joint_reliable[None] & (hinges[2, :, 1] > 0)
    joint_stable = np.stack((np.broadcast_to(joint_reliable.sum(), (2,)),
                             joint_lost.sum(-1), joint_weakened.sum(-1)), axis=-1)
    relation_by_identity = []
    for identity in np.unique(labels):
        selected = np.asarray(labels)[q] == identity
        row_changes = np.stack([
            change_sums(margins[a][..., selected],
                        margins[b][..., selected],
                        hinges[a][..., selected],
                        hinges[b][..., selected])
            for a, b in STATE_PAIRS])
        relation_by_identity.append({"identity": int(identity), "triplets": int(selected.sum()),
                                     "changes": row_changes.tolist(),
                                     "joint_stable": np.stack((np.broadcast_to(joint_reliable[selected].sum(), (2,)),
                                                               joint_lost[..., selected].sum(-1),
                                                               joint_weakened[..., selected].sum(-1)),axis=-1).tolist(),
                                     "joint_changes": change_sums(um[..., selected], margins[2, :, 1][..., selected],
                                                                 uh[..., selected], hinges[2, :, 1][..., selected]).tolist()})
    return {"triplets": len(q), "changes": changes, "stable": np.stack(stable),
            "joint_changes": joint, "joint_stable":joint_stable, "stable_by_identity": stable_by_identity,
            "relation_by_identity": relation_by_identity}


def validate_matrices(a, active):
    assert a.shape == (3, 2, 18, 64, 64) and np.isfinite(a).all()
    assert np.max(np.abs(a - a.swapaxes(-1, -2))) < 2e-5
    assert np.max(np.abs(np.diagonal(a, axis1=-2, axis2=-1) - 1)) < 2e-5
    for state in range(3):
        for view in range(2):
            x = a[state, view].astype(np.float64)
            assert np.array_equal(x[0], a[0, 0, 0])
            if not active:
                assert np.array_equal(a[state, view], a[state, 0])
            assert np.max(np.abs(x[14] - x[15:18].mean(0))) < 2e-5
            assert np.max(np.abs(x[14] - x[5:14].mean(0))) < 2e-5
            for role in range(3):
                assert np.max(np.abs(x[2+role] - (.5*x[0]+.5*x[15+role]))) < 2e-5
            if state != 2:
                assert np.max(np.abs(x[1] - x[2:5].mean(0))) < 2e-5


def validate_vector_statistics(a):
    assert a.shape[-1] == 6 and np.isfinite(a).all()
    left2, right2, dot, diff2, cosine, chord = np.moveaxis(a, -1, 0)
    assert np.all(left2 > 0) and np.all(right2 > 0) and np.all(diff2 >= 0)
    assert np.max(np.abs(left2-1)) < 2e-5 and np.max(np.abs(right2-1)) < 2e-5
    expected_cosine = dot / np.sqrt(left2*right2)
    expected_chord = np.sqrt(np.maximum(0, left2+right2-2*dot))
    assert np.allclose(diff2, left2+right2-2*dot, rtol=1e-5, atol=1e-12)
    assert np.allclose(cosine, expected_cosine, rtol=1e-5, atol=1e-5)
    assert np.allclose(chord, expected_chord, rtol=1e-5, atol=1e-5)
    return max(float(np.max(np.abs(cosine-expected_cosine))),
               float(np.max(np.abs(chord-expected_chord))))

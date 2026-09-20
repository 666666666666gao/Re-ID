"""Source-support-aware, role-block gradient coefficients; no inference module."""
import math


ROLES = ('cnn', 'transformer', 'mamba')
RULE = dict(ema_decay=0.9, exponent=0.5, ratio_min=0.25, ratio_max=4.0,
            epsilon=1e-12, weight_sum=2.0)


def role_indices(names):
    groups = {role: [i for i, name in enumerate(names)
                     if name.startswith('encoder.' + role + '_')] for role in ROLES}
    covered = [i for group in groups.values() for i in group]
    assert len(covered) == len(set(covered)) == len(names)
    assert sorted(covered) == list(range(len(names))) and all(groups.values())
    return groups


class SupportedBalance:
    def __init__(self):
        self.states = {role: None for role in ROLES}

    def observe(self, role, rank_norm, auxiliary_norm, supported):
        assert role in self.states
        assert all(math.isfinite(x) and x >= 0 for x in (rank_norm, auxiliary_norm))
        before = self.states[role]
        after = before
        rank_weight = auxiliary_weight = 1.0
        ratio = None
        if supported:
            if before is None:
                after = dict(rank=rank_norm, auxiliary=auxiliary_norm, supported_steps=1)
            else:
                after = dict(rank=0.9 * before['rank'] + 0.1 * rank_norm,
                             auxiliary=0.9 * before['auxiliary'] + 0.1 * auxiliary_norm,
                             supported_steps=before['supported_steps'] + 1)
            ratio = min(4.0, max(0.25, math.sqrt(
                (after['auxiliary'] + 1e-12) / (after['rank'] + 1e-12))))
            rank_weight = 2 * ratio / (1 + ratio)
            auxiliary_weight = 2 / (1 + ratio)
            self.states[role] = after
        return dict(supported=bool(supported), before=before, after=after, ratio=ratio,
                    proposed_rank_weight=rank_weight, proposed_auxiliary_weight=auxiliary_weight)


def scalar_gradients(loss, parameters, scale):
    import torch
    values = torch.autograd.grad(loss * scale, parameters, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p, dtype=torch.float32) if g is None else g.detach().float()
            for p, g in zip(parameters, values, strict=True)]


def check_reference(actual, reference):
    from tools.probe_msvr_history_candidate_gradients import compare
    result = compare(reference, actual)
    result['relative_l2_error'] = (result['difference_norm'] / result['first_norm']
                                    if result['first_norm'] else None)
    result['passed'] = (result['relative_l2_error'] <= 0.005
                        if result['relative_l2_error'] is not None
                        else result['difference_norm'] <= 1e-8)
    assert result['passed'], result
    return result


def combine(parameters, current_total, current_rank, historical_rank, scale, groups,
            controller, supported, apply):
    """All input gradients are AMP-scaled; history belongs only to ranking."""
    import torch
    from tools.probe_msvr_history_candidate_gradients import compare
    rank = [a + b for a, b in zip(current_rank, historical_rank, strict=True)]
    auxiliary = [a - b for a, b in zip(current_total, current_rank, strict=True)]
    rows = {}
    for role, indexes in groups.items():
        r = [rank[i] / scale for i in indexes]
        a = [auxiliary[i] / scale for i in indexes]
        pair = compare(r, a)
        observation = controller.observe(role, pair['first_norm'], pair['second_norm'], supported)
        wr = observation['proposed_rank_weight'] if apply and supported else 1.0
        wa = observation['proposed_auxiliary_weight'] if apply and supported else 1.0
        # Preserve the original control arithmetic, including unsupported steps.
        for i in indexes:
            expected = (rank[i] * wr + auxiliary[i] * wa if apply and supported
                        else current_total[i] + historical_rank[i])
            parameters[i].grad.copy_(expected)
            assert torch.equal(parameters[i].grad, expected)
        actual = [parameters[i].grad / scale for i in indexes]
        rows[role] = dict(rank_vs_auxiliary=pair,
            current_rank_vs_history=compare([current_rank[i] / scale for i in indexes],
                                             [historical_rank[i] / scale for i in indexes]),
            rank_vs_applied=compare(r, actual), auxiliary_vs_applied=compare(a, actual),
            original_sum_vs_applied=compare([(current_total[i] + historical_rank[i]) / scale for i in indexes], actual),
            **observation, applied_rank_weight=wr, applied_auxiliary_weight=wa)
    return rows, rank, auxiliary


def support_counts(identities, scenes, memory, counts):
    candidates = list(zip(identities, scenes)) + [(r['identity'], r['scene']) for r in memory]
    supported_ids = {identity for identity, n in zip(identities, counts, strict=True) if n > 0}
    relations = {(identity, scene, other_scene)
                 for identity, scene, n in zip(identities, scenes, counts, strict=True) if n > 0
                 for other_id, other_scene in candidates if identity == other_id and scene != other_scene}
    return dict(eligible_anchors=sum(n > 0 for n in counts), eligible_identities=len(supported_ids),
                identity_directed_scene_relations=len(relations))

"""Keep raw author logits, use the deployed joint vector for role metrics."""


def metric_heads(output):
    feature = output['fused']
    assert feature.ndim == 2 and feature.shape[1] == 1536
    return [(score, feature) for score, _raw_part in output['heads']]

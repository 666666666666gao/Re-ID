"""Source-bound scalar geometry analysis; no torch, model or remote execution."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math

ROOT = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
ARCHIVE = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/partition_terminal_intake781/_source')
OUTPUT = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/deployment_metric_geometry847')
SCOPE = ROOT / 'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
SCOPE_SHA = '76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softplus(value):
    return math.log1p(math.exp(value))


def main():
    assert not OUTPUT.exists()
    assert sha(SCOPE) == SCOPE_SHA
    scope = json.loads(SCOPE.read_bytes())
    sources = scope['source_sha256']
    assert len(sources) == 339
    names = (
        'modeling/trifusion/deployment_metric_role.py',
        'modeling/trifusion/global_task_role_heads.py',
        'tools/run_deployment_metric_role.py',
        'tools/run_global_task_role.py',
        'tools/run_role_input_detach.py',
        'tools/run_foundation_recipe.py',
        'comparators/Signal-cd1b0a6/layers/make_loss.py',
        'comparators/Signal-cd1b0a6/layers/triplet_loss.py',
    )
    verified = {}
    for name in names:
        path = ARCHIVE / name if name.startswith('comparators/') else ROOT / name
        assert sha(path) == sources[name], name
        verified[name] = dict(sha256=sources[name], local_source=str(path))

    # Unit vectors have Euclidean distances in [0, 2]. This is an exact-real
    # geometry bound, not a tolerance claim for the float32 implementation.
    floor = softplus(-2)
    sigmoid_floor = 1 / (1 + math.exp(2))
    angles = (0.0, 0.1, math.pi, math.pi + 0.1)
    vectors = [(math.cos(a), math.sin(a)) for a in angles]
    labels = (0, 0, 1, 1)
    unit_distances = [[math.dist(x, y) for y in vectors] for x in vectors]
    margins = [
        min(unit_distances[i][j] for j in range(4) if labels[j] != labels[i])
        - max(unit_distances[i][j] for j in range(4) if labels[j] == labels[i])
        for i in range(4)
    ]
    examples = []
    for scale in (1.0, 2.0, 5.0, 10.0):
        examples.append(dict(
            uniform_raw_scale=scale,
            raw_soft_margin_mean=sum(softplus(-scale * margin) for margin in margins) / 4,
            l2_soft_margin_mean=sum(softplus(-margin) for margin in margins) / 4,
            raw_order_same=True,
            normalized_vectors_same=True,
        ))
    assert len({row['l2_soft_margin_mean'] for row in examples}) == 1
    assert all(row['l2_soft_margin_mean'] >= floor for row in examples)

    h = (3.0, 4.0)
    v = (0.7, -0.2)
    radius = math.hypot(*h)
    z = tuple(value / radius for value in h)
    projection = sum(a * b for a, b in zip(z, v))
    gradient = tuple((a - b * projection) / radius for a, b in zip(v, z))
    radial_dot = sum(a * b for a, b in zip(h, gradient))
    assert abs(radial_dot) < 1e-15

    result = dict(
        status='SOURCE_BOUND_SCALAR_GEOMETRY_ANALYSIS_COMPLETE',
        created_at=datetime.now().astimezone().isoformat(),
        producer_sha256=sha(Path(__file__)), scope_sha256=SCOPE_SHA,
        verified_sources=verified,
        ideal_unit_euclidean_soft_margin_lower_bound=floor,
        ideal_three_copies_unweighted_triplet_lower_bound=3 * floor,
        per_margin_softplus_derivative_lower_bound=sigmoid_floor,
        uniform_scale_examples=examples,
        normalized_gradient_example=dict(h=h, upstream=v, gradient=gradient, radial_dot=radial_dot),
        boundary=(
            'Exact-real scalar derivation and ideal toy vectors only; no PyTorch execution, '
            'model inference, batch replay, new official score or scientific source change. '
            'The universal lower bound is not necessarily attainable for actual multi-identity batches. '
            'The three-copy bound concerns only unweighted Triplet, not CE or the whole training loss. '
            'A nonzero softplus derivative in margin space does not prove a nonzero parameter gradient, '
            'because normalization, mining, feature derivatives and cancellations also matter. '
            'For nonzero h outside the normalization epsilon branch, normalized metric gradients are '
            'tangent to h; this term alone does not directly reward uniform radial growth. '
            'Increasing correction relative to detached global can change directions, but no causal '
            'claim about recorded correction growth, CMC damage or overfitting is established. '
            'All current training, queue, gates, weights and original failures stay unchanged.'
        ),
    )
    OUTPUT.mkdir()
    (OUTPUT / 'ANALYSIS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], lower_bound=floor, three_copy_bound=3 * floor,
                         sources_verified=len(verified), radial_dot=radial_dot, output=str(OUTPUT))))


if __name__ == '__main__':
    main()

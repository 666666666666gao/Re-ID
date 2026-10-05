"""Record source-bound loss algebra; no model, network or training execution."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
OUTPUT = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/deployment_metric_loss_source_invariants850')
scope = json.loads((REPO / 'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_bytes())
runtime = scope['source_sha256']
assert len(runtime) == 339
paths = ('tools/run_foundation_recipe.py', 'tools/run_deployment_metric_role.py',
         'modeling/trifusion/deployment_metric_role.py', 'modeling/trifusion/global_task_role_heads.py',
         'comparators/Signal-cd1b0a6/layers/make_loss.py', 'comparators/Signal-cd1b0a6/layers/triplet_loss.py')
exact_sources = []
for name in paths:
    local = REPO / name
    if name.startswith('comparators/'):
        local = REPO / 'evidence/smooth_ap_m0_audit_20260909/snapshots/remote/root/autodl-tmp/trifusion-v2' / name
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    assert digest == runtime[name]
    exact_sources.append(dict(runtime_source=name, local_exact_copy=str(local.relative_to(REPO)), sha256=digest))

bound = math.log1p(math.exp(-2))
result = dict(status='SOURCE_BOUND_MATHEMATICAL_INTERPRETATION_ONLY',
    at=datetime.now().astimezone().isoformat(), source_scope_sha256=hashlib.sha256(
        (REPO / 'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_bytes()).hexdigest(),
    exact_sources=exact_sources, source_count_checked=6, full_runtime_scope_checked=False,
    model_execution=False, remote_query=False, new_metric_result=False,
    formula='L_tri=mean(log(1+exp(d_ap-d_an))); normalized-vector Euclidean distances are at most2 in ideal real arithmetic.',
    ideal_per_triplet_lower_bound=bound,
    multihead_formula='sum_h(lambda_id*CE_h + lambda_tri*L_tri(z)) = lambda_id*sum_h CE_h + H*lambda_tri*L_tri(z); z is the same joint1536 normalized vector for every head.',
    three_head_unweighted_triplet_lower_bound=3 * bound,
    interpretation='Normalized soft-margin loss cannot approach zero just by scaling features. A nonzero value alone does not prove useful gradient support. Raw and normalized loss magnitudes cannot measure relative fitting quality directly; saved combined head_losses do not separate CE from Triplet.',
    boundary='Ideal mathematical bound, not a floating-point equality/tolerance check, exact attainable batch minimum, diagnosis of retrieval degradation, evidence of an optimal new margin/scale, or authorization to change the live experiment. H and weights follow the configured heads; no new metric or seed claimed.')
assert not OUTPUT.exists()
OUTPUT.mkdir()
(OUTPUT / 'SOURCE_INVARIANTS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
(OUTPUT / 'PRODUCER.py').write_bytes(Path(__file__).read_bytes())
(OUTPUT / 'README.md').write_text('''# Source-bound interpretation of the deployed role Triplet\n\nSix exact local source copies match the current339-source map, including the pinned author loss implementation. This is a source/algebra analysis, with no remote query or model execution.\n\nFor vectors in the unit Euclidean ball, d_ap and d_an lie between0 and2 in ideal real arithmetic. The configured no-margin Triplet uses mean log(1+exp(d_ap-d_an)), so each term is bounded below by log(1+exp(-2)), approximately0.126928. The numerical distance clamp makes this conservative; floating-point roundoff is not being tested here, and this is not an attainable global minimum for an entire multi-identity batch.\n\nThe deployment-metric wrapper supplies the same joint1536 normalized vector to every author head. The author objective sums head losses, so with H heads its metric contribution is H times the configured Triplet weight times the common Triplet loss. For three heads the unweighted lower bound is approximately0.380784; this does not set or change a training weight.\n\nTherefore, old raw-feature observations of zero Triplet loss cannot be used as an activity criterion for this normalized soft-margin objective. Nonzero loss also does not establish useful gradients or successful retrieval learning. Saved head losses combine weighted CE and Triplet, and do not recover their separate histories. Neither raw-versus-normalized aggregate loss values nor this mathematical bound identify the cause of the observed best-to-last mAP degradation. Keep the current experiment and fixed selected checkpoints unchanged.\n''', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'exact_sources'}))

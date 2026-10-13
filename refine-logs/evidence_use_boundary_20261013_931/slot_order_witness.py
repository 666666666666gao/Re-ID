"""A dimension-valid algebraic witness, not a trained-model experiment."""
import hashlib
import json
import math
from pathlib import Path

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
out = Path(__file__).resolve().parent
assert not (out / 'SLOT_ORDER_WITNESS.json').exists()
paths = ['modeling/trifusion/region_use_objective.py',
         'modeling/trifusion/correspondence_evidence_readout.py',
         'modeling/trifusion/region_use_heads.py',
         'modeling/trifusion/global_conditioned_evidence.py']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

# Each of nine role/modality groups contains the sixteen nonzero basis vectors.
# Dimension128 permits this exact construction without normalization error.
query = [list(range(16)) for _ in range(9)]
gallery = [list(range(16)) for _ in range(9)]
permuted = [g.copy() for g in query]
permuted[0][0], permuted[0][2] = permuted[0][2], permuted[0][0]

def symmetric_mean_maxsim(a, b):
    return sum((sum(max(int(i == j) for j in gb) for i in ga) / 16
                + sum(max(int(i == j) for i in ga) for j in gb) / 16) / 2
               for ga, gb in zip(a, b)) / 9

# The production reshape/mean makes four 2x2 blocks of a4x4 slot grid.
groups = [[4*y+x for y in range(2*gy,2*gy+2) for x in range(2*gx,2*gx+2)]
          for gy in range(2) for gx in range(2)]
assert groups == [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]

def readout_example(a):
    # A legal 1536x384 weight tensor with only modality0/group0/output0/
    # CNN-input-channel0 weight=1. The remaining output coordinates are zero.
    return sum(int(a[0][p] == 0) for p in groups[0]) / 4

c_original, c_permuted = map(readout_example, (query, permuted))
assert c_original == 0.25 and c_permuted == 0.0
set_original = symmetric_mean_maxsim(query, gallery)
set_permuted = symmetric_mean_maxsim(permuted, gallery)
assert set_original == set_permuted == 1.0
# g=e0+e1 in1536 dimensions, illustrative gain=1, not any trained gain.
original = [1+c_original, 1]
changed = [1+c_permuted, 1]
cos = sum(x*y for x,y in zip(original,changed)) / math.sqrt(sum(x*x for x in original)*sum(y*y for y in changed))
assert cos < 1
result = {
    'schema': 'source-backed-slot-order-algebra-witness-v1',
    'evaluation_type': 'algebraic_counterexample_only',
    'source_sha256': {p: sha(repo/p) for p in paths},
    'script_sha256': sha(Path(__file__)),
    'tensor_domain': '9 groups x16 slots x128 channels;1536 readout',
    'fixed_permutation': 'Only CNN/modality0 swaps positions0 and2 across readout groups0 and1',
    'readout_slot_groups': groups,
    'nonzero_basis_value_vectors': True,
    'regional_similarity_before': set_original,
    'regional_similarity_after': set_permuted,
    'correction_output0_before': c_original,
    'correction_output0_after': c_permuted,
    'illustrative_gain': 1,
    'fused_pair_cosine_before': 1,
    'fused_pair_cosine_after': cos,
    'official_metric_or_ground_truth_used': False,
    'trained_weight_used': False,
    'neural_forward_calls': 0,
    'optimizer_updates': 0,
    'conclusion': 'The regional set score can remain unchanged while the structured readout and additive cosine change.',
    'limitations': 'This legal tensor/weight witness is not sampled from any trained model. It establishes an unconstrained direction of the auxiliary score, not observed slot drift, retrieval loss, semantic parts, incompatibility of all set losses or a causal root cause. Fused tasks still supervise the deployed representation and may constrain this direction.'}
(out/'SLOT_ORDER_WITNESS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('evaluation_type','regional_similarity_before',
      'regional_similarity_after','correction_output0_before','correction_output0_after',
      'fused_pair_cosine_after','neural_forward_calls')}))

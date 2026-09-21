"""Audit-only in-memory verifier repair; frozen project files remain byte-identical."""
from pathlib import Path
import hashlib
import json
import math
import os
import struct
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

REPO = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
ROOT = Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/q1')
PATH = REPO/'tools/verify_msvr_supported_gradient_balance_stats.py'
source = PATH.read_bytes()
original_sha = hashlib.sha256(source).hexdigest()
assert original_sha == '5f8f157b39a9b11dd6ac5447b2a8619915cf20ddb6815fdc4ac107d7a571b327'
text = source.decode('utf-8')
old_sqrt = "((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))**.5"
new_sqrt = "math.sqrt((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))"
old_norm = '                close(actual**2,wr*wr*r*r+wa*wa*a*a+2*wr*wa*ra)'
new_norm = "                wr32,wa32=(struct.unpack('<f',struct.pack('<f',w))[0] for w in (wr,wa))\n                close(actual**2,wr32*wr32*r*r+wa32*wa32*a*a+2*wr32*wa32*ra)"
assert text.count(old_sqrt) == text.count(old_norm) == 1
assert text.count('import math\n') == 1
repaired = text.replace(old_sqrt, new_sqrt).replace(old_norm, new_norm).replace('import math\n', 'import math\nimport struct\n')
assert 'assert abs(a-b)<=1e-7*max(1,abs(a),abs(b)), (a,b)' in repaired
namespace = {}
exec(compile(repaired, str(PATH)+':audit_minimum_repair', 'exec'), namespace)

def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]

scalars_checked = 0
endpoints = []
for directory in sorted(ROOT.glob('fold_*')):
    endpoint = directory.name.rsplit('_', 1)[1]
    rows = [json.loads(line) for line in (directory/'memory_steps.jsonl').read_text().splitlines()]
    training = json.loads((directory/'training.json').read_bytes())
    check = namespace['verify_balance'](rows, training, endpoint, 65)
    endpoints.append(dict(endpoint=directory.name, result=check))
    for row in rows:
        for b in row['gradient_balance'].values():
            for name in ('applied_rank_weight', 'applied_auxiliary_weight'):
                weight = b[name]
                probe = torch.ones(1, dtype=torch.float32) * weight
                assert probe.dtype == torch.float32 and probe.item() == f32(weight)
                scalars_checked += 1

# Explicit synthetic counterexample, not replay of a saved training vector.
wr = 1.0421102637890658
wa = 0.9578897362109341
rank = torch.ones(1, dtype=torch.float32)
aux = torch.zeros(1, dtype=torch.float32)
actual = rank*wr + aux*wa
lhs = float(actual.double().square().sum())
rhs64 = wr*wr
rhs32 = f32(wr)*f32(wr)
threshold = 1e-7 * max(1, abs(lhs), abs(rhs64))
assert abs(lhs-rhs64) > threshold
assert lhs == rhs32
assert PATH.read_bytes() == source
assert not torch.cuda.is_initialized()
print(json.dumps(dict(
    status='PASS_MINIMUM_REPAIR_SCALAR_ASSERTIONS_ONLY',
    scope='All original verify_balance assertions with exactly sqrt and float32 coefficient representation corrections; not full Q1 verifier.',
    python=sys.version, torch=torch.__version__, device='cpu', cuda_initialized=False,
    source_sha256=original_sha,
    candidate_source_sha256=hashlib.sha256(repaired.encode()).hexdigest(),
    frozen_source_unchanged=True, close_threshold_unchanged=True,
    exact_controller_assertions_unchanged=True, m0_reference_thresholds_unchanged=True,
    torch_python_scalar_conversion_checks=scalars_checked,
    synthetic_counterexample=dict(label='synthetic_float32_one_zero_no_model',
        actual_squared_norm=lhs, python_weight_expected_squared_norm=rhs64,
        float32_weight_expected_squared_norm=rhs32,
        original_error=abs(lhs-rhs64), original_threshold=threshold,
        float32_weight_error=abs(lhs-rhs32)),
    endpoints=endpoints, model_forwards=0, optimizer_updates=0,
    models_arrays_images_loaded_or_downloaded=0), indent=2))

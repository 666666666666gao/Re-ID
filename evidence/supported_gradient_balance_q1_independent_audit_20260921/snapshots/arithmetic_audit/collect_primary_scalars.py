"""Read-only independent scalar audit. No model, array, image, or training load."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import platform
import struct

REPO = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
ROOT = Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/q1')
ROLES = ('cnn', 'transformer', 'mamba')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

def error_record(kind, left, right, location, extra=None):
    tolerance = 1e-7 * max(1., abs(left), abs(right))
    out = dict(kind=kind, **location, left=left, right=right,
               absolute_error=abs(left-right), original_threshold=tolerance,
               ratio_to_original_threshold=abs(left-right)/tolerance)
    if extra:
        out.update(extra)
    return out

hashes = {name: sha(REPO/name) for name in (
    'tools/verify_msvr_supported_gradient_balance_stats.py',
    'tools/msvr_supported_gradient_balance.py',
    'tools/probe_msvr_history_candidate_gradients.py',
    'tools/train_msvr_supported_gradient_balance.py',
    'tools/verify_msvr_supported_gradient_balance.py',
    'tools/recheck_msvr_supported_balance_sqrt.py',
    'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json')}
spec = json.loads((REPO/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json').read_bytes())
summary = json.loads((ROOT/'summary.json').read_bytes())
result = dict(status='READ_ONLY_SCALAR_AUDIT_NOT_FULL_Q1_VERIFICATION',
              run=str(ROOT), python=platform.python_version(), source_sha256=hashes,
              summary_sha256=sha(ROOT/'summary.json'), original_scientific_status=summary['status'],
              downloaded_models_arrays_images=0, optimizer_updates=0, model_forwards=0,
              rows=[], violations=[], maxima={}, failing_witnesses=[],
              ratio_power_mismatches=[], exact_controller_assertions='pending')
counts = Counter()

def check(kind, left, right, location, extra=None):
    assert math.isfinite(left) and math.isfinite(right)
    row = error_record(kind, left, right, location, extra)
    counts[kind] += 1
    previous = result['maxima'].get(kind)
    if previous is None or row['ratio_to_original_threshold'] > previous['ratio_to_original_threshold']:
        result['maxima'][kind] = row
    if row['ratio_to_original_threshold'] > 1:
        result['violations'].append(row)
    return row

for directory in sorted(ROOT.glob('fold_*')):
    endpoint = directory.name.rsplit('_', 1)[1]
    training_path = directory/'training.json'
    training = json.loads(training_path.read_bytes())
    log_path = directory/'memory_steps.jsonl'
    audits = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(audits) == len(training['steps']) == training['optimizer_steps'] == 260
    proof = training['audit_files']['memory_steps.jsonl']
    assert sha(log_path) == proof['sha256'] and log_path.stat().st_size == proof['bytes']
    state = {role: None for role in ROLES}
    supported_steps = unsupported_steps = reference_steps = 0
    scale_values = set()
    zero_support_locations = []
    for index, (audit, training_step) in enumerate(zip(audits, training['steps'])):
        assert audit['step'] == training_step['step'] == index+1
        scale = training_step['amp_scale_before']
        scale_values.add(scale)
        assert math.isfinite(scale) and scale > 0 and math.frexp(scale)[0] == .5
        eligible = sum(x > 0 for x in audit['relation_objective']['cross_scene_positive_counts'])
        supported = index >= spec['memory']['warmup_steps'] and eligible > 0
        supported_steps += int(supported)
        unsupported_steps += int(index >= spec['memory']['warmup_steps'] and not supported)
        if index >= spec['memory']['warmup_steps'] and not supported:
            zero_support_locations.append(index+1)
        reference_steps += int(bool(audit['rank_auxiliary_reference_checks']))
        for role in ROLES:
            b = audit['gradient_balance'][role]
            location = dict(endpoint=directory.name, step=index+1, role=role)
            pair = b['rank_vs_auxiliary']
            rank, aux = pair['first_norm'], pair['second_norm']
            dot = 0 if pair['cosine'] is None else rank*aux*pair['cosine']
            assert b['before'] == state[role] and b['supported'] == supported
            ratio = None
            wr = wa = 1.
            if supported:
                before = state[role]
                state[role] = (dict(rank=rank, auxiliary=aux, supported_steps=1) if before is None else
                    dict(rank=.9*before['rank']+.1*rank, auxiliary=.9*before['auxiliary']+.1*aux,
                         supported_steps=before['supported_steps']+1))
                quotient = (state[role]['auxiliary']+1e-12)/(state[role]['rank']+1e-12)
                ratio = min(4., max(.25, math.sqrt(quotient)))
                ratio_power = min(4., max(.25, quotient**.5))
                if ratio != ratio_power:
                    result['ratio_power_mismatches'].append(dict(**location, sqrt=ratio, power=ratio_power,
                        recorded=b['ratio'], ulps=abs(ratio-ratio_power)/math.ulp(ratio)))
                wr, wa = 2*ratio/(1+ratio), 2/(1+ratio)
            assert b['after'] == state[role] and b['ratio'] == ratio
            assert b['proposed_rank_weight'] == wr and b['proposed_auxiliary_weight'] == wa
            if endpoint == 'control' or not supported:
                wr = wa = 1.
                assert b['original_sum_vs_applied']['difference_norm'] == 0
            assert b['applied_rank_weight'] == wr and b['applied_auxiliary_weight'] == wa
            for name in ('rank_vs_auxiliary', 'current_rank_vs_history', 'rank_vs_applied',
                         'auxiliary_vs_applied', 'original_sum_vs_applied',
                         'subtraction_auxiliary_vs_direct', 'direct_sum_vs_original'):
                p = b[name]
                x, y, d, c = p['first_norm'], p['second_norm'], p['difference_norm'], p['cosine']
                assert all(math.isfinite(v) and v >= 0 for v in (x, y, d))
                assert (c is None) == (x == 0 or y == 0)
                check('pair_'+name, d*d, x*x+y*y-(0 if c is None else 2*x*y*c), location)
            p = b['current_rank_vs_history']
            x, y, c = p['first_norm'], p['second_norm'], p['cosine']
            check('current_plus_history_rank_norm', rank*rank, x*x+y*y+(0 if c is None else 2*x*y*c), location)
            check('direct_sum_norm', b['direct_sum_vs_original']['first_norm']**2,
                  rank*rank+aux*aux+2*dot, location)
            if endpoint == 'balanced' and supported:
                actual = b['rank_vs_applied']['second_norm']
                lhs = actual**2
                rhs = wr*wr*rank*rank+wa*wa*aux*aux+2*wr*wa*dot
                wr32, wa32 = f32(wr), f32(wa)
                rhs32 = wr32*wr32*rank*rank+wa32*wa32*aux*aux+2*wr32*wa32*dot
                old = check('weighted_norm_python_weights', lhs, rhs, location)
                corrected = check('weighted_norm_float32_weights', lhs, rhs32, location)
                # Forward error for two float32 products and one sum, after accounting for scalar conversion.
                u = 2.**-24
                gamma2 = 2*u/(1-2*u)
                vector_bound = gamma2*(abs(wr32)*rank+abs(wa32)*aux)
                ideal_norm = math.sqrt(max(0., rhs32))
                norm_squared_bound = 2*ideal_norm*vector_bound+vector_bound**2
                bound_row = dict(**location, absolute_error=abs(lhs-rhs32),
                    squared_norm_rounding_bound=norm_squared_bound,
                    ratio_to_bound=abs(lhs-rhs32)/max(norm_squared_bound, 1e-300))
                previous = result.get('maximum_operation_bound_usage')
                if previous is None or bound_row['ratio_to_bound'] > previous['ratio_to_bound']:
                    result['maximum_operation_bound_usage'] = bound_row
                if old['ratio_to_original_threshold'] > 1:
                    result['failing_witnesses'].append(dict(**location, scale=scale,
                        gradient_balance=b, applied_gradients=audit['applied_gradients'][role],
                        actual_parameter_updates=audit['actual_parameter_updates'][role],
                        float64_weights=[wr,wa], float32_weights=[wr32,wa32],
                        unrounded_ideal_norm_squared=rhs, coefficient_rounded_ideal_norm_squared=rhs32,
                        original_error=lhs-rhs, coefficient_corrected_error=lhs-rhs32,
                        vector_rounding_bound_gamma2=vector_bound,
                        squared_norm_rounding_bound_gamma2=norm_squared_bound,
                        coefficient_corrected_error_to_operation_bound=abs(lhs-rhs32)/norm_squared_bound))
    assert state == training['gradient_balance_state']
    result['rows'].append(dict(endpoint=directory.name, steps=len(audits),
        training_sha256=sha(training_path), memory_steps_sha256=sha(log_path),
        memory_steps_bytes=log_path.stat().st_size, supported_steps=supported_steps,
        zero_supported_steps=unsupported_steps, zero_support_steps=zero_support_locations,
        direct_reference_steps=reference_steps, amp_scale_values=sorted(scale_values)))
result['comparison_counts'] = dict(counts)
result['exact_controller_assertions'] = 'PASS_ALL_SIX_ENDPOINTS'
print(json.dumps(result, indent=2))

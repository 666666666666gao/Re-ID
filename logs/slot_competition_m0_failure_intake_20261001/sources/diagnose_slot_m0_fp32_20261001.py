"""Eight-step attention-FP32 counterfactual; not a production M0 result."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

import torch
import torch.nn.functional as F

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools import run_slot_competition_roles as slot
from trifusion.slot_competition_roles import allocation_weights

target = root / 'logs/slot_competition_m0_fp32_20261001.json'
run = root / 'trained-model/slot_competition_m0_fp32_20261001'
assert not target.exists() and not run.exists()
memory = subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.used',
                                  '--format=csv,noheader,nounits'], text=True)
assert int(memory.strip()) < 500
original = root / 'logs/slot_competition_roles_20261001/slot_competition_roles_20261001_slot_competition_competitive_RGBNT201/campaign.json'
state = json.loads(original.read_text())
assert state['status'] == 'FAILED' and state['jobs'][0]['exit_code'] == 1
argv = state['jobs'][0]['command'][2:]
argv[argv.index('--output-dir') + 1] = str(run)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.argv = argv


def fp32_sample(roles, patches, positions, context_queries, role):
    with torch.autocast('cuda', enabled=False):
        patches, context_queries = patches.float(), context_queries.float()
        width = patches.shape[-1]
        queries = roles.anchor_queries[None] + context_queries[:, role, None]
        queries = roles.query_projections[role](F.layer_norm(queries, (width,)))
        keys = roles.key_projections[role](F.layer_norm(patches, (width,)))
        values = roles.value_projections[role](patches)
        scores = torch.matmul(queries[:, None], keys.transpose(-1, -2)) * width ** -0.5
        weights = allocation_weights(scores, roles.attention_normalization)
        return roles.output_projections[role](torch.matmul(weights, values))


def capture(frame, event, arg):
    if event == 'line' and frame.f_lineno == 149:
        values = frame.f_locals
        named, live = values['named_parameters'], values['live_parameters']
        result = {'status': 'CAPTURED_COUNTERFACTUAL_BEFORE_M0_ASSERT',
                  'missing_nonzero_gradient': sorted(set(named) - live),
                  'live_tensor_count': len(live), 'trainable_tensor_count': len(named),
                  'qk_last_batch_gradients': {name: None if parameter.grad is None else {
                      'absolute_sum': float(parameter.grad.abs().sum()),
                      'absolute_max': float(parameter.grad.abs().max()),
                  } for name, parameter in named.items() if 'projections' in name},
                  'original_failed_campaign': str(original),
                  'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'scope': 'Disposable method instrumentation: all role attention arithmetic FP32 only; same production M0 loop8 steps/seed/data. Not an unmodified production M0 or formal candidate score. Original assertion remains active.'}
        target.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in ('status', 'missing_nonzero_gradient', 'live_tensor_count', 'trainable_tensor_count')}), flush=True)
    return capture


def trace(frame, event, arg):
    if frame.f_code.co_name == 'build' and frame.f_code.co_filename == slot.__file__ and event == 'return':
        arg[0].roles.sample_context = MethodType(fp32_sample, arg[0].roles)
    if event == 'call' and frame.f_code.co_name == 'train' and frame.f_code.co_filename == slot.entry.__file__:
        return capture
    if frame.f_code.co_name == 'build' and frame.f_code.co_filename == slot.__file__:
        return trace
    return None


sys.settrace(trace)
slot.main()

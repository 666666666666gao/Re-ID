"""Record the real M0 assertion locals; preserve the original failed gate."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
from tools import run_slot_competition_roles as slot

target = root / 'logs/slot_competition_m0_gradient_20261001.json'
run = root / 'trained-model/slot_competition_m0_gradient_20261001'
assert not target.exists() and not run.exists()
memory = subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.used',
                                  '--format=csv,noheader,nounits'], text=True)
assert int(memory.strip()) < 500
original = root / 'logs/slot_competition_roles_20261001/slot_competition_roles_20261001_slot_competition_competitive_RGBNT201/campaign.json'
state = json.loads(original.read_text())
assert state['status'] == 'FAILED' and state['jobs'][0]['exit_code'] == 1
command = state['jobs'][0]['command']
argv = command[2:]
argv[argv.index('--output-dir') + 1] = str(run)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.argv = argv


def capture(frame, event, arg):
    if event == 'line' and frame.f_lineno == 149:
        values = frame.f_locals
        named = values['named_parameters']
        live = values['live_parameters']
        missing = sorted(set(named) - live)
        gradients = {name: None if parameter.grad is None else {
            'absolute_sum': float(parameter.grad.abs().sum()),
            'absolute_max': float(parameter.grad.abs().max()),
        } for name, parameter in named.items()}
        result = {'status': 'CAPTURED_BEFORE_ORIGINAL_M0_ASSERT', 'missing_nonzero_gradient': missing,
                  'live_tensor_count': len(live), 'trainable_tensor_count': len(named),
                  'last_batch_gradient': gradients, 'original_failed_campaign': str(original),
                  'original_campaign_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                  'entry_sha256': hashlib.sha256(Path(slot.entry.__file__).read_bytes()).hexdigest(),
                  'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'scope': 'Same production M0/seed/data;8 diagnostic optimizer steps in a new directory. No assertion bypass or original receipt update.'}
        target.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in ('status', 'missing_nonzero_gradient', 'live_tensor_count', 'trainable_tensor_count')}), flush=True)
    return capture


def trace(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == 'train' and frame.f_code.co_filename == str(Path(slot.entry.__file__)):
        return capture
    return None


sys.settrace(trace)
slot.main()

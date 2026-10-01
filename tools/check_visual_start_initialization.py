"""Witness exact trainable-state equality before the six visual-start runs."""
import argparse
from datetime import datetime
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_visual_start_roles as run
from tools.queue_correspondence_roles import PROTOCOLS, SOURCE, WEIGHTS
from trifusion.role_global_tokens import GlobalTokenTriFusion


def state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    inputs = json.loads(args.inputs.read_text())
    run.INPUTS_PATH = args.inputs.resolve()
    run.base.TOKEN_MODE = 'static'
    run.base.entry.CONDITION.update(query_mode='context', auxiliary_target='none')
    run.base.entry.runner.CorrespondenceTriFusion = partial(
        GlobalTokenTriFusion, token_mode='static', query_mode='context', auxiliary_target='none')
    rows = []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        protocol = json.loads((PROTOCOLS / f'{dataset}.json').read_text())
        snapshots, bindings = {}, {}
        for variant in ('reid_visual', 'public_visual'):
            run.VISUAL_START = variant
            weight = inputs['datasets'][dataset][variant]
            options = argparse.Namespace(dataset=dataset, signal_source=SOURCE,
                clip_weight=WEIGHTS/'ViT-B-16.pt', baseline_checkpoint=Path(weight['path']),
                baseline_sha256=weight['sha256'], seed=42, width=128, m1=True, m2=True, m3=False,
                pred_weight=0.1, protocol=PROTOCOLS/f'{dataset}.json')
            model, _, _, binding = run.build(options, protocol)
            snapshots[variant] = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
            bindings[variant] = binding
            del model
            gc.collect()
            torch.cuda.empty_cache()
        saved, public = snapshots['reid_visual'], snapshots['public_visual']
        assert set(saved) == set(public)
        visual_prefix = 'backbone.signal.clip_vision_encoder.base.'
        nonvisual = [key for key in saved if not key.startswith(visual_prefix)]
        assert all(torch.equal(saved[key], public[key]) for key in nonvisual)
        changed = [key for key in saved if not torch.equal(saved[key], public[key])]
        assert changed and all(key.startswith(visual_prefix) for key in changed)
        assert bindings['reid_visual']['trainable_parameters'] == bindings['public_visual']['trainable_parameters']
        new_state = {key:value for key,value in saved.items() if not key.startswith('backbone.signal.')}
        rows.append({'dataset':dataset, 'trainable_states_bitwise_equal':True,
                     'all_nonvisual_initial_states_bitwise_equal':True, 'frozen_visual_states_differ':True,
                     'changed_frozen_visual_keys':changed, 'trainable_parameters':bindings['reid_visual']['trainable_parameters'],
                     'new_state_sha256':state_sha(new_state), 'initializers':bindings})
    record = {'status':'MATCHED_TRAINABLE_INITIALIZATION_PASS',
              'completed_at':datetime.now().astimezone().isoformat(), 'inputs_sha256':run.base.entry.sha256(args.inputs),
              'source_sha256':run.base.entry.sha256(Path(__file__)), 'rows':rows,
              'scope':'Actual production model construction on one GPU, no input forward, gradient, optimizer or retrieval evaluation. M0 remains separate.'}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'datasets':[row['dataset'] for row in rows], 'output':str(args.output)}))


if __name__ == '__main__':
    main()

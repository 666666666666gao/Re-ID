"""Classify the registered unscaled auxiliary gate after real production M0."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_role_objective_m0 as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--dataset', choices=original.DATASETS, required=True)
    parser.add_argument('--objective', choices=original.OBJECTIVES, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = ROOT / f'trained-model/{campaign.name}_m0_{args.objective}_{args.dataset}'
    binding = json.loads((campaign / 'initialization' / f'{args.dataset}_{args.objective}.json').read_text())['binding']
    receipt = json.loads((output / 'training.json').read_text())
    assert receipt['schema'] == original.SCHEMA and receipt['status'] == 'M0_PASS' and receipt['initializer'] == binding
    assert len(receipt['history']) == 1 and receipt['history'][0]['steps'] == 8
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters'] == binding['trainable_parameter_tensors']
    assert receipt['m0']['reload_max_abs_difference'] <= 1e-5
    assert receipt['production_m0_diagnostics']['effective_optimizer_updates'] == 8
    assert all(value == 8 for value in receipt['production_m0_diagnostics']['author_bn_batches_tracked'].values())
    assert receipt['frozen_parameters_unchanged'] and receipt['visual_parameters_changed'] and receipt['fresh_camera_parameters_changed']
    steps = [json.loads(line) for line in (output / 'training_steps.jsonl').read_text().splitlines()]
    assert len(steps) == 8 and all(row['incremental_isolated_shared_global_gradient_absent'] for row in steps)
    names = set(steps[0]['incremental_isolated_query_key_gradient_norms'])
    assert len(names) == 6 and all(set(row['incremental_isolated_query_key_gradient_norms']) == names for row in steps)
    norms = [row['incremental_isolated_correction_gradient_norm'] for row in steps]
    all_norms = norms + [value for row in steps for value in row['incremental_isolated_query_key_gradient_norms'].values()]
    assert all(math.isfinite(value) and value >= 0 for value in all_norms)
    sums = {name: sum(row['incremental_isolated_query_key_gradient_norms'][name] for row in steps) for name in names}
    if sum(norms) > 0 and all(value > 0 for value in sums.values()):
        original.accept_m0(campaign, args.dataset, args.objective, output, binding)
        return 0
    batches = [json.loads(line) for line in (output / 'training_batch_order.jsonl').read_text().splitlines()][:8]
    old = ROOT / f'trained-model/global_task_role_v1_20261004_824_m0_semantic_{args.dataset}/training_batch_order.jsonl'
    assert len(batches) == 8 and batches == [json.loads(line) for line in old.read_text().splitlines()][:8]
    probe = output / 'm0_reload_probe.pth'
    assert probe.resolve().is_relative_to((ROOT / 'trained-model').resolve()) and probe.stat().st_nlink == 1
    assert {p.name for p in output.glob('*.pth')} == {'m0_reload_probe.pth'}
    assert original.base.sha(probe) == receipt['m0']['reload_probe_sha256']
    result = dict(status='REGISTERED_UNSCALED_AUXILIARY_GATE_FAIL', dataset=args.dataset, objective=args.objective,
        correction_gradient_norm_sum=sum(norms), query_key_gradient_norm_sums=sums,
        classified_at=original.base.queue.stamp(), source_raw_m0_batch_order_equal=True,
        artifact_sha256={str(p): original.base.sha(p) for p in output.iterdir() if p.is_file() and p != probe},
        probe=dict(path=str(probe), bytes=probe.stat().st_size, sha256=original.base.sha(probe)),
        boundary='Original first-eight unscaled auxiliary criterion unchanged. Production M0 passed; no claim of actual scaled inactivity or formal retrieval failure. Engineering probe has no remaining registered consumer; retain complete receipts/steps/batches before retirement.')
    path = campaign / 'activity_failures' / f'{args.dataset}_{args.objective}.json'
    assert not path.exists()
    path.parent.mkdir(exist_ok=True)
    original.base.queue.write(path, result)
    probe.unlink()
    with (campaign / 'probe_retirement.jsonl').open('a') as log:
        log.write(json.dumps(dict(retired_at=original.base.queue.stamp(), failure_receipt=str(path), **result['probe']))+'\n')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

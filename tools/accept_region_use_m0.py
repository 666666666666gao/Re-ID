"""Prepared text-only gate; no producer launch or result repair."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m0-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training_path = args.m0_dir/'training.json'
    steps_path = args.m0_dir/'training_steps.jsonl'
    training = json.loads(training_path.read_text())
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    assert training['status'] == 'M0_PASS' and len(steps) == 8
    assert training['production_m0_diagnostics']['effective_optimizer_updates'] == 8
    support = [row['region_use_support'] for row in steps]
    activity = [row['region_use_isolated_activity'] for row in steps]
    assert all(row['shared_global_gradient_absent'] for row in activity)
    assert any(row['eligible_queries'] > 0 for row in support)
    assert any(row['actual_region_gradient_norm'] > 0 for row in activity)
    names = set(activity[0]['query_key_value_gradient_norms'])
    assert len(names) == 9 and all(set(row['query_key_value_gradient_norms']) == names for row in activity)
    missing = [name for name in sorted(names) if not any(row['query_key_value_gradient_norms'][name] > 0 for row in activity)]
    assert not missing, missing
    result = dict(status='REGION_USE_M0_SUPPORT_AND_ACTIVITY_PASS',
        eligible_queries=sum(row['eligible_queries'] for row in support),
        legal_triplets=sum(row['legal_triplets'] for row in support),
        active_parameter_names=sorted(names),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (training_path,steps_path)},
        boundary='Original8 finite/update/reload M0 plus actual auxiliary support/activity. Not performance, uniqueness or generalization. Synthetic component checks do not qualify this gate.')
    assert not args.output.exists()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

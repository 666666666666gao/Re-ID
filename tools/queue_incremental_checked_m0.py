"""Fresh six-end M0 with only the auxiliary observation context corrected."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_incremental_role_objective_m0 as original

SCOPE = ROOT / 'refine-logs/incremental_role_objective_v1/CHECKED_VJP_SOURCE_SCOPE.json'
original_command = original.command
original_accept_m0 = original.accept_m0


def source_map():
    source = json.loads(SCOPE.read_text())['source_sha256']
    assert all(original.base.sha(ROOT / name) == digest for name, digest in source.items())
    return source


def command(dataset, objective, mode, campaign, output):
    result = original_command(dataset, objective, mode, campaign, output)
    assert result[2] == str(ROOT / 'tools/run_incremental_role_objective.py')
    result[2] = str(ROOT / 'tools/run_incremental_role_objective_checked.py')
    return result


def accept_m0(campaign, dataset, objective, output, binding):
    steps = [json.loads(line) for line in (output / 'training_steps.jsonl').read_text().splitlines()]
    assert len(steps) == 8 and binding['m0_auxiliary_vjp_policy'] == 'unscaled_autocast_disabled_same_original_targets'
    assert all(row['incremental_isolated_vjp_autocast_enabled'] is False for row in steps)
    assert all(not unused for row in steps for unused in row['incremental_isolated_query_key_unused'].values())
    return original_accept_m0(campaign, dataset, objective, output, binding)


def configure():
    original.source_map = source_map
    original.command = command
    original.accept_m0 = accept_m0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    configure()
    raise SystemExit(original.coordinate(parser.parse_args().campaign.resolve()))

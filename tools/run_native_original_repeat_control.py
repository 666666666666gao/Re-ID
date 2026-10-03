"""Run the single registered native repeat control once and record child exit."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert output.is_dir() and not (output/'JOB.json').exists()
    command = [sys.executable, '-B', str(ROOT/'tools/check_native_original_repeat.py'),
               '--reference-campaign', str(ROOT/'logs/independent_native_evidence_20261003_v5'),
               '--output-dir', str(output)]
    state = {'status': 'RUNNING', 'controller_pid': os.getpid(), 'started_at': stamp(),
             'command': command, 'optimizer_updates': 0, 'weights_generated': 0}
    with (output/'control.log').open('x') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        (output/'JOB.json').write_text(json.dumps(state, indent=2)+'\n')
        code = child.wait()
    state.update(status='FAILED' if code else 'COMPLETE', exit_code=code, completed_at=stamp())
    if code == 0:
        assert json.loads((output/'PASS.json').read_text())['status'] == 'NATIVE_ORIGINAL_REPEAT_CONTROL_PASS'
    (output/'JOB.json').write_text(json.dumps(state, indent=2)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())

"""Post-hoc CPU replay with disclosed sqrt repair; execution bindings stay intact."""
import argparse
import hashlib
import json
from pathlib import Path

from tools import verify_msvr_supported_gradient_balance_stats as stats
from tools import verify_msvr_supported_gradient_balance as verifier


def run(args):
    source = Path(stats.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == '5f8f157b39a9b11dd6ac5447b2a8619915cf20ddb6815fdc4ac107d7a571b327'
    old = "((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))**.5"
    new = "math.sqrt((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))"
    text = source.decode('utf-8')
    assert text.count(old) == 1
    repaired = text.replace(old, new)
    namespace = {}
    exec(compile(repaired, str(Path(stats.__file__)) + ':sqrt_repair', 'exec'), namespace)
    stats.verify_balance = namespace['verify_balance']
    print(json.dumps(dict(repair='verifier-only sqrt arithmetic; exact comparisons unchanged',
        original_source_sha256=hashlib.sha256(source).hexdigest(),
        repaired_source_sha256=hashlib.sha256(repaired.encode()).hexdigest(),
        original_execution_files_unchanged=True)), flush=True)
    verifier.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

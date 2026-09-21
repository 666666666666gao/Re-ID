"""Post-hoc CPU replay with disclosed arithmetic repairs; bindings stay intact."""
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
    old_norm = '                close(actual**2,wr*wr*r*r+wa*wa*a*a+2*wr*wa*ra)'
    new_norm = "                wr32,wa32=(struct.unpack('<f',struct.pack('<f',w))[0] for w in (wr,wa))\n                close(actual**2,wr32*wr32*r*r+wa32*wa32*a*a+2*wr32*wa32*ra)"
    assert repaired.count(old_norm) == repaired.count('import math\n') == 1
    repaired = repaired.replace(old_norm, new_norm).replace('import math\n', 'import math\nimport struct\n')
    assert hashlib.sha256(repaired.encode()).hexdigest() == '4f21a5ec2ace07ed446d56c39881b3e3ce7c3783ff6edfa0bad26bc0b0e85231'
    namespace = {}
    exec(compile(repaired, str(Path(stats.__file__)) + ':sqrt_repair', 'exec'), namespace)
    stats.verify_balance = namespace['verify_balance']
    print(json.dumps(dict(repair='verifier-only sqrt and FP32 scalar coefficients; original thresholds unchanged',
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

import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path('/data/gaob/Re-ID/Trifusion')
EXPECTED = {'tools/report_shared_private_evidence_complete.py': 'ef1aea1ce0d1557fae5bef8c0514b4188450fbf53df9f1d5b50eb818e9fc4781',
            'tools/wait_shared_private_complete_analysis.py': 'dee45aa5a73a8316eed7b266b540f35216231bb4f9e77f01bc542ada1644dcdb'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


proof = ROOT / 'logs/shared_private_report_deployment_20261001.json'
assert not proof.exists()
for name, digest in EXPECTED.items():
    path = ROOT / name
    assert sha(path) == digest
    ast.parse(path.read_text())
for relative, count in (('logs/shared_private_evidence_20261001_v2/manifest.json', 229),
                        ('logs/visual_update_control_20261001_v1/manifest.json', 222)):
    manifest = json.loads((ROOT / relative).read_text())
    assert len(manifest['source_sha256']) == count
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
    assert sha(Path(manifest['preflight_path'])) == manifest['preflight_sha256']
    assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
for name in EXPECTED:
    runpy.run_path(str(ROOT / name), run_name='deployment_import_only')
assert all(sha(ROOT / name) == digest for name, digest in EXPECTED.items())
result = {'checked_at': datetime.now().astimezone().isoformat(), 'status': 'AST_AND_NATIVE_IMPORT_PASS',
          'source_sha256': EXPECTED, 'active_sources_unchanged': 229, 'old_sealed_sources_unchanged': 222,
          'scope': 'AST/import only; no main, neural, report or waiter invocation.'}
proof.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))

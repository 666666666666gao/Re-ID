"""Receive terminal text only; models, distances and gradient tensors stay remote."""
from pathlib import Path
import hashlib
import json
import shlex
import sys

destination = Path(sys.argv[1])
connection_helper = Path(sys.argv[2])
assert not destination.exists(), destination
exec(connection_helper.read_text(encoding="utf-8"), globals())

remote_script = r'''
from pathlib import Path
import datetime, hashlib, json, shutil
root = Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
pipelines = {}
for suffix in ('_pipeline.json', '_verification_pipeline.json', '_analysis_pipeline.json'):
    value = json.loads(Path(str(root) + suffix).read_bytes())
    assert value['status'] == 'COMPLETE' and value['exit_code'] == 0, (suffix, value)
    pipelines[suffix] = value
summary = json.loads((root/'summary.json').read_bytes())
verification = json.loads((root/'source_verification.json').read_bytes())
analysis = json.loads((root/'analysis/analysis.json').read_bytes())
assert summary['status'] == 'COMPLETE_SOURCE_OBJECTIVE_GRADIENTS'
assert verification['status'] == 'PASS_COMPLETE_SOURCE_OBJECTIVE_GRADIENT_LEDGER'
assert analysis['status'] == 'COMPLETE_VERIFIED_SOURCE_GRADIENT_ANALYSIS'
assert verification['summary_sha256'] == analysis['summary_sha256'] == sha(root/'summary.json')
assert analysis['verification_sha256'] == sha(root/'source_verification.json')
assert verification['steps'] == 1560 and verification['role_rows'] == 4680
assert analysis['role_rows'] == 4680 and analysis['groups'] == 72
expected = [f'fold_{f}_{e}' for f in range(3) for e in ('control', 'smooth_ap')]
assert [v['directory'] for v in summary['conditions']] == expected
for name in expected:
    assert len((root/name/'steps.jsonl').read_text().splitlines()) == 260
allowed = {'.json', '.jsonl', '.csv', '.md', '.py', '.log'}
files = [p for p in root.rglob('*') if p.is_file() and p.suffix in allowed]
files += [p for p in root.parent.glob(root.name + '_*') if p.is_file() and p.suffix in allowed]
rows = []
for p in sorted(files):
    data = p.read_bytes()
    data.decode('utf-8')
    name = p.relative_to(root).as_posix() if p.is_relative_to(root) else 'runtime/' + p.name
    rows.append(dict(remote=str(p), name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
    root=str(root), pipelines=pipelines, files=rows, free_bytes=shutil.disk_usage(root).free,
    scope='Terminal text intake only; no independent model or full-vector gradient replay.')))
'''
_, stdout, stderr = c.exec_command(
    "/root/miniconda3/envs/tri_reid/bin/python -B -c " + shlex.quote(remote_script)
)
inventory_bytes = stdout.read()
error = stderr.read()
assert stdout.channel.recv_exit_status() == 0, error.decode()
inventory = json.loads(inventory_bytes)
destination.mkdir(parents=True)
(destination/"inventory.json").write_bytes(inventory_bytes)
sftp = c.open_sftp()
for entry in inventory["files"]:
    with sftp.open(entry["remote"], "rb") as stream:
        stream.prefetch()
        data = stream.read()
    assert len(data) == entry["bytes"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["remote"]
    target = destination/entry["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
sftp.close()
c.close()
receipt = dict(status="COMPLETE_VERIFIED_TEXT_INTAKE", files=len(inventory["files"]),
               bytes=sum(row["bytes"] for row in inventory["files"]),
               inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
               remote_binary_files_downloaded=0)
(destination/"intake.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt))

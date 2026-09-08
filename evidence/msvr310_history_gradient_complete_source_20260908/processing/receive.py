"""Receive this registered diagnostic only after its complete terminal receipt."""
from datetime import datetime
from pathlib import Path
import hashlib
import io
import json
import shlex

tmp = Path('C:/Users/gb/.codex_tmp')
destination = tmp / 'history_gradient_complete_source_20260908'
assert not destination.exists()
exec((tmp / 'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'), globals())
run = '/root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_candidate_gradient_v1_seed42_eebaaa0'
remote = '''
from pathlib import Path
import hashlib,json
root=Path(RUN)
pipeline=json.loads((root/'pipeline.json').read_bytes())
assert pipeline['status']=='COMPLETE_VERIFIED_SOURCE_ONLY'
assert pipeline['code_commit']=='eebaaa07708e8f5e5d3e05d246afe5fe5f7abfc6'
assert pipeline['config_sha256']=='c99ddcf6f6605b2d43725e7fc4f74f0747e774a1110a3d0567c3c42c74a1c0b3'
assert all(s['exit_code']==0 for s in pipeline['stages'])
summary=json.loads((root/'source/summary.json').read_bytes())
cpu=json.loads((root/'source/cpu_verification.json').read_bytes())
assert summary['status']=='PASS_COMPLETE_FIXED_STATE_PROBE' and summary['mode']=='source'
assert cpu['status']=='PASS_COMPLETE_FIXED_STATE_PROBE_CPU' and cpu['batches']==2340
assert pipeline['summary_sha256']==cpu['summary_sha256']==hashlib.sha256((root/'source/summary.json').read_bytes()).hexdigest()
assert pipeline['cpu_sha256']==hashlib.sha256((root/'source/cpu_verification.json').read_bytes()).hexdigest()
names=['pipeline.json','t0.json','t0.log','source.log','source_cpu.log','source/summary.json','source/cpu_verification.json']
for fold in summary['folds']:
 for state in ('initial','control','fresh_memory'):
  receipt=fold['states'][state]
  assert receipt['batches']==260 and receipt['history_batches']==194
  directory=root/'source'/f"fold_{fold['fold']}_{state}"
  assert receipt==json.loads((directory/'receipt.json').read_bytes())
  for name,binding in receipt['files'].items():
   path=directory/name
   assert path.stat().st_size==binding['bytes']
   assert hashlib.sha256(path.read_bytes()).hexdigest()==binding['sha256']
  names.extend(str((directory/n).relative_to(root)) for n in ('receipt.json','steps.jsonl'))
assert len(names)==25
files=[dict(path=n,bytes=(root/n).stat().st_size,sha256=hashlib.sha256((root/n).read_bytes()).hexdigest()) for n in names]
print(json.dumps(dict(status='PASS_COMPLETE_SOURCE_REMOTE_BINDINGS',files=files,source_run=str(root))))
'''.replace('RUN', repr(run))
_, stdout, stderr = c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -c ' + shlex.quote(remote))
data, error = stdout.read(), stderr.read()
assert stdout.channel.recv_exit_status() == 0, error.decode()
manifest = json.loads(data)
destination.mkdir()
sftp = c.open_sftp()
for item in manifest['files']:
    buffer = io.BytesIO()
    sftp.getfo(run + '/' + item['path'], buffer, prefetch=True, max_concurrent_prefetch_requests=64)
    raw = buffer.getvalue()
    assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    local = destination / item['path']
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(raw)
sftp.close()
c.close()
manifest.update(status='PASS_ALL_COMPLETE_SOURCE_TEXT_SIZE_SHA',received_at=datetime.now().astimezone().isoformat())
(destination / 'intake_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=manifest['status'],files=len(manifest['files']),bytes=sum(f['bytes'] for f in manifest['files']),directory=str(destination))))

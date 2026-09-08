"""Read-only first-endpoint artifact and transition check; no retrieval-score selection."""
from pathlib import Path
import json,shlex
tmp=Path('C:/Users/gb/.codex_tmp')
launch=json.loads((tmp/'history_gradient_training_launch_20260908.json').read_bytes())
target=tmp/'history_gradient_first_endpoint_verified_20260908.json';assert not target.exists()
exec((tmp/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
script=r'''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path(__ROOT__);directory=root/'q1/fold_0_control'
pipeline=json.loads((root/'pipeline.json').read_bytes())
assert pipeline['code_commit']==__HEAD__ and pipeline['stages'][-1]['stage']=='q1'
assert Path('/proc',str(pipeline['wrapper_pid'])).exists()
assert Path('/proc',str(pipeline['stages'][-1]['original_pid'])).exists()
r=json.loads((directory/'receipt.json').read_bytes());tr=json.loads((directory/'training.json').read_bytes())
assert tr==r['training'] and tr['optimizer_steps']==260 and tr['epochs']==20
assert tr['trainable_tensors']==tr['nonzero_gradient_tensors']==203 and tr['overflow_events']==0
assert all(r['engineering_checks'].values())
assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
checkpoint=Path(r['checkpoint']);assert checkpoint.parent==directory and checkpoint.name=='roles_epoch20.pth'
checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert checkpoint_sha==r['checkpoint_sha256']
ranking_sha=hashlib.sha256((directory/'rankings.json').read_bytes()).hexdigest();assert ranking_sha==r['retrieval']['rankings_sha256']
assert set(r['retrieval']['outputs'])=={'baseline_only','fused','cnn','transformer','mamba'}
next_log=root/'q1/fold_0_history_gradient/memory_steps.jsonl'
lines=next_log.read_bytes().split(b'\n')[:-1];assert lines
next_row=json.loads(lines[-1])
result=dict(status='PASS_FIRST_ENDPOINT_ARTIFACTS_AND_NEXT_ARM_LIVE',checked_at=datetime.now().astimezone().isoformat(),
 execution_commit=pipeline['code_commit'],wrapper_pid=pipeline['wrapper_pid'],q1_pid=pipeline['stages'][-1]['original_pid'],
 control_steps=tr['optimizer_steps'],control_epochs=tr['epochs'],control_training_seconds=sum(x['elapsed_seconds'] for x in tr['history']),
 nonzero_gradient_tensors=tr['nonzero_gradient_tensors'],overflow=tr['overflow_events'],
 checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=checkpoint_sha,
 rankings_sha256=ranking_sha,receipt_sha256=hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),
 query_count=len(r['retrieval']['query_rows']),gallery_count=len(r['retrieval']['gallery_manifest']),
 fresh_record_forwards=tr['extra_fresh_role_record_forwards'],history_vjp_record_forwards=tr['extra_history_vjp_record_forwards'],
 peak_allocated_mib=tr['peak_allocated_mib'],next_arm='fold_0_history_gradient',next_arm_last_step=next_row['step'],
 free_bytes=shutil.disk_usage(root).free,scope='First endpoint files and recorded runtime checks only; no independent final CPU verification or complete paired scientific conclusion.')
print(json.dumps(result))
'''.replace('__ROOT__',repr(launch['root'])).replace('__HEAD__',repr(launch['code_commit']))
_,out,err=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -c '+shlex.quote(script))
data=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
target.write_bytes(data);print(data.decode());c.close()

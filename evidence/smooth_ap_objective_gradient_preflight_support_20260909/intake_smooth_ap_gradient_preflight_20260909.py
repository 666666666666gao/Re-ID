from pathlib import Path
import hashlib,json,shlex
tmp=Path('C:/Users/gb/.codex_tmp');dest=tmp/'smooth_ap_objective_gradient_preflight_complete_20260909';assert not dest.exists()
exec((tmp/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
root='/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f'
script='''from pathlib import Path
import hashlib,json,shutil
root=Path(ROOT_VALUE)
j=json.loads((root/'summary.json').read_bytes());v=json.loads((root/'preflight_verification.json').read_bytes())
assert j['status']=='PASS_PREFLIGHT' and v['status']=='PASS_COMPLETE_OBJECTIVE_GRADIENT_PREFLIGHT'
assert v['summary_sha256']==hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')]
files += [p for p in root.parent.glob(root.name+'_*') if p.is_file() and p.suffix in ('.json','.log','.py')]
print(json.dumps(dict(files=[dict(remote=str(p),name=p.relative_to(root).as_posix() if p.is_relative_to(root) else 'runtime/'+p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)],free_bytes=shutil.disk_usage(root).free)))
'''.replace('ROOT_VALUE',repr(root))
_,out,err=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -B -c '+shlex.quote(script))
b=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
j=json.loads(b);dest.mkdir();(dest/'inventory.json').write_bytes(b)
sftp=c.open_sftp()
for r in j['files']:
    with sftp.open(r['remote'],'rb') as f:f.prefetch();data=f.read()
    assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
    p=dest/r['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
sftp.close();c.close()
(dest/'intake.json').write_text(json.dumps(dict(status='COMPLETE_TEXT_INTAKE',files=len(j['files']),bytes=sum(r['bytes'] for r in j['files'])),indent=2)+'\n')
summary=json.loads((dest/'summary.json').read_bytes())
rows=[json.loads(line) for p in dest.glob('fold_*/steps.jsonl') for line in p.read_text().splitlines()]
print(json.dumps(dict(files=len(j['files']),steps=len(rows),end_seconds=[r['elapsed_seconds'] for r in summary['conditions']],
    maximum_current_decomposition=max(r['current_decomposition']['relative_to_sum_of_component_norms'] for r in rows),
    maximum_full_decomposition=max(r['full_decomposition']['relative_to_sum_of_component_norms'] for r in rows),
    maximum_direct_vjp=max(r['direct_history_proof']['relative_error'] for r in summary['conditions']),free_bytes=j['free_bytes'])))

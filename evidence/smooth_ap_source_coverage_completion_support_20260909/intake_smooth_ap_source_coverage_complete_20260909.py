from pathlib import Path
import hashlib,json,shlex
tmp=Path('C:/Users/gb/.codex_tmp')
dest=tmp/'smooth_ap_source_coverage_complete_20260909';assert not dest.exists()
exec((tmp/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
root='/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590'
script='''from pathlib import Path
import datetime,hashlib,json
root=Path(ROOT_VALUE)
original=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
verify=json.loads(Path(str(root)+'_verify_pipeline.json').read_bytes())
assert original['status']=='COMPLETE' and verify['status']=='COMPLETE' and verify['exit_code']==0
check=json.loads((root/'coverage_verification.json').read_bytes())
assert check['status']=='PASS_ALL_SOURCE_COVERAGE_ROWS'
assert check['candidate_rows']==1996800 and check['full_rows']==82560
files=[p for p in root.iterdir() if p.is_file() and p.suffix=='.json']
files += [p for d in root.iterdir() if d.is_dir() for p in d.iterdir() if p.name=='receipt.json' or p.name=='inputs.jsonl' or p.name.endswith('_full.json')]
files += [p for p in root.parent.glob(root.name+'_*') if p.is_file() and p.suffix in ('.json','.log','.py')]
rows=[]
for p in sorted(files):
    b=p.read_bytes();rows.append(dict(remote=str(p),name=(p.relative_to(root).as_posix() if p.is_relative_to(root) else 'runtime/'+p.name),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(dict(observed_at=datetime.datetime.now().astimezone().isoformat(),files=rows,
    process_exists={str(p):Path('/proc',str(p)).exists() for p in [original['pid'],verify['pid'],verify['child_pid']]}),indent=2))
'''.replace('ROOT_VALUE',repr(root))
_,out,err=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -B -c '+shlex.quote(script))
b=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error.decode()
inventory=json.loads(b);dest.mkdir();(dest/'inventory.json').write_bytes(b)
sftp=c.open_sftp()
for row in inventory['files']:
    p=dest/row['name'];p.parent.mkdir(parents=True,exist_ok=True)
    with sftp.open(row['remote'],'rb') as f:f.prefetch();data=f.read()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    p.write_bytes(data)
sftp.close();c.close()
receipt=dict(status='COMPLETE_BYTE_VERIFIED_TEXT_INTAKE',files=len(inventory['files']),bytes=sum(r['bytes'] for r in inventory['files']),
             large_candidate_rows_and_features='remote; full CPU verification recorded')
(dest/'intake.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))

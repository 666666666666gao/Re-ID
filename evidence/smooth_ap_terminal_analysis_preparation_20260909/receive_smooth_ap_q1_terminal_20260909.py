"""Receive completed Smooth-AP Q1 text only; do not run before terminal CPU receipt."""
from pathlib import Path
import hashlib,json,sys

destination=Path(sys.argv[1])
assert not destination.exists()
env={}
exec(Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),env)
c=env['c']
remote_script=r'''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
r=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
pipe=json.loads((r/'pipeline.json').read_bytes())
assert pipe['status'] in ('COMPLETE_VERIFIED_Q1_PASS','COMPLETE_VERIFIED_Q1_FAIL'), pipe['status']
assert pipe['code_commit']=='2e947a4325144e37fed638105ac954e7e54b5fe5'
assert [s['stage'] for s in pipe['stages']]==['t0','m0','m0_cpu','q1','q1_cpu']
assert all(s['exit_code']==0 for s in pipe['stages'])
pids=[pipe['wrapper_pid'],*[s['original_pid'] for s in pipe['stages']]]
live={str(pid):Path('/proc',str(pid)).exists() for pid in pids}
assert not any(live.values()),live
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cpu=json.loads((r/'q1_cpu.json').read_bytes())
assert cpu['status']=='PASS_COMPLETE_SMOOTH_AP_Q1'
assert cpu['checked_training_steps']==1560
assert cpu['summary_sha256']==sha(r/'q1/summary.json')==pipe['terminal_summary_sha256']
assert sha(r/'q1_cpu.json')==pipe['terminal_cpu_sha256']
files=[]
for p in sorted(r.rglob('*')):
    if p.is_file() and p.suffix in ('.json','.jsonl','.log'):
        files.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),remote=str(r),pipeline=pipe,original_pid_exists=live,files=files,free_bytes=shutil.disk_usage(r).free,repo_head=subprocess.check_output(['git','-C','/root/autodl-tmp/trifusion-v2/TriFusion-ReID','rev-parse','HEAD'],text=True).strip())))
'''
i,o,e=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -B -',timeout=120)
i.write(remote_script);i.channel.shutdown_write()
raw=o.read();err=e.read();code=o.channel.recv_exit_status()
assert code==0,err.decode('utf-8')
j=json.loads(raw)
destination.mkdir()
(destination/'remote_terminal_inventory.json').write_bytes(raw)
s=c.open_sftp()
for item in j['files']:
    local=destination/item['path'];local.parent.mkdir(parents=True,exist_ok=True)
    s.get(j['remote']+'/'+item['path'],str(local))
    assert local.stat().st_size==item['bytes']
    assert hashlib.sha256(local.read_bytes()).hexdigest()==item['sha256'],item['path']
s.close();c.close()
(destination/'intake_complete.json').write_text(json.dumps(dict(status='RECEIVED_ALL_REGISTERED_TERMINAL_TEXT',files=len(j['files']),bytes=sum(x['bytes'] for x in j['files']),inventory_sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=j['pipeline']['status'],files=len(j['files']),bytes=sum(x['bytes'] for x in j['files']),local=str(destination))))

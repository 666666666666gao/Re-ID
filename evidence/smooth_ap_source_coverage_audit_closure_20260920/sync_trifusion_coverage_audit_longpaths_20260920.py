from pathlib import Path
from datetime import datetime
import hashlib,json,shlex,subprocess,sys
def read_local(path):return Path(chr(92)*2+"?"+chr(92)+str(path)).read_bytes()

p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');tmp=Path('C:/Users/gb/.codex_tmp')
manifest=Path(sys.argv[1]);m=json.loads(manifest.read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()==m['new_head']
assert subprocess.check_output(['git','-c','http.sslBackend=openssl','ls-remote','origin','refs/heads/main'],cwd=p,text=True).split()[0]==m['new_head']
bundle=tmp/(manifest.stem+'_'+m['new_head'][:7]+'.bundle');assert not bundle.exists()
subprocess.run(['git','bundle','create',str(bundle),m['old_head']+'..HEAD'],cwd=p,check=True)
exec((tmp/'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
remote_root='/root/autodl-tmp/trifusion-v2/TriFusion-ReID';remote_bundle='/root/autodl-tmp/trifusion-v2/transport/'+bundle.name
def command(cmd):
    _,out,err=c.exec_command(cmd);data=out.read();error=err.read();assert out.channel.recv_exit_status()==0,error
    return data.decode()
assert command('git -C '+shlex.quote(remote_root)+' rev-parse HEAD').strip()==m['old_head']
sftp=c.open_sftp();sftp.put(str(bundle),remote_bundle)
command('git -C '+shlex.quote(remote_root)+' fetch '+shlex.quote(remote_bundle)+' HEAD')
command('git -C '+shlex.quote(remote_root)+' merge --ff-only FETCH_HEAD')
script="from pathlib import Path\nimport hashlib,json,subprocess,shutil\np=Path("+repr(remote_root)+")\npaths="+repr(m['paths'])+"\nrows=[dict(path=f,bytes=(p/f).stat().st_size,sha256=hashlib.sha256((p/f).read_bytes()).hexdigest()) for f in paths]\nprint(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),files=rows,free_bytes=shutil.disk_usage(p).free)))"
stdin,stdout,stderr=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -B -',timeout=60)
stdin.write(script);stdin.channel.shutdown_write()
raw=stdout.read();error=stderr.read();assert stdout.channel.recv_exit_status()==0,error
remote=json.loads(raw)
assert remote['head']==m['new_head']
for row in remote['files']:
    b=read_local(p/row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
with sftp.open(remote_root+'/'+master,'rb') as stream:
    stream.prefetch();remote_master=stream.read()
local_master=(p/master).read_bytes();assert remote_master==local_master
desktop=Path('C:/Users/gb/Desktop/document')/Path(master).name
assert hashlib.sha256(desktop.read_bytes()).hexdigest()==m['prior_master_sha256']
desktop.write_bytes(local_master);assert desktop.read_bytes()==remote_master
sftp.close();c.close()
proof=dict(checked_at=datetime.now().astimezone().isoformat(),head=m['new_head'],files=remote['files'],master_three_way_sha256=hashlib.sha256(local_master).hexdigest(),free_bytes=remote['free_bytes'])
(tmp/(manifest.stem+'_sync.json')).write_bytes((json.dumps(proof,indent=2)+'\n').encode())
print(json.dumps(dict(head=m['new_head'],files=len(remote['files']),master_sha256=proof['master_three_way_sha256'],free_bytes=remote['free_bytes'])))

from pathlib import Path
import hashlib,json,stat,shlex,datetime,shutil
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
folder=Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_m0_r1_failure_20260921');folder.mkdir()
root='/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_seed42_92a75e4'
exec(Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
sftp=c.open_sftp();rows=[]
def receive(remote,relative):
    with sftp.open(remote,'rb') as f:data=f.read()
    out=folder/'remote'/relative;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    rows.append(dict(path=out.relative_to(folder).as_posix(),remote_path=remote,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
def walk(remote,relative):
    for item in sftp.listdir_attr(remote):
        name=item.filename
        if stat.S_ISDIR(item.st_mode):walk(remote+'/'+name,relative/name)
        elif Path(name).suffix in ('.json','.jsonl','.log','.txt','.csv'):
            receive(remote+'/'+name,relative/name)
walk(root,Path('.'))
receive(root+'_launch.json',Path('launch.json'))
receive(root+'_wrapper.log',Path('wrapper.log'))
script='from pathlib import Path\nimport hashlib,json\npaths='+repr([r['remote_path'] for r in rows])+'\nprint(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}))'
_,stdout,stderr=c.exec_command('/root/miniconda3/envs/tri_reid/bin/python -c '+shlex.quote(script))
proof=json.loads(stdout.read());error=stderr.read();assert stdout.channel.recv_exit_status()==0,error
assert all(proof[r['remote_path']]==r['sha256'] for r in rows)
sftp.close();c.close()
pipeline=json.loads((folder/'remote/pipeline.json').read_bytes())
assert pipeline['status']=='STOPPED_AT_M0' and pipeline['stages'][-1]['exit_code']==1
config=repo/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json';spec=json.loads(config.read_bytes())
for name in [config.relative_to(repo).as_posix(),*spec['project_file_sha256']]:
    old=repo/name;new=folder/'executed_snapshot'/name;new.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,new)
for name in ('trifusion_supported_balance_observation_0539_20260921.json','trifusion_supported_balance_m0_gradients_0539_20260921.json'):
    shutil.copyfile(folder.parent/name,folder/name)
(folder/'inventory.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),
    files=rows,remote_text_sha_verified=True,binary_downloads=0,code_commit=pipeline['code_commit'],
    config_sha256=pipeline['config_sha256'],scope='Original R1 engineering stop; no Q1 or retrieval result'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(folder=str(folder),remote_files=len(rows),bytes=sum(r['bytes'] for r in rows))))

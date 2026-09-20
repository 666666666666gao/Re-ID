from pathlib import Path
import hashlib,json,shlex,datetime

folder=Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_r2_start_20260921')
folder.mkdir()
root='/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639'
names={'t0.json':root+'/t0.json','pipeline_at_intake.json':root+'/pipeline.json','launch.json':root+'_launch.json'}
exec(Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'),globals())
sftp=c.open_sftp();rows=[]
for name,remote in names.items():
    with sftp.open(remote,'rb') as stream:
        stream.prefetch();data=stream.read()
    assert isinstance(json.loads(data),dict)
    (folder/name).write_bytes(data)
    rows.append(dict(path=name,remote_path=remote,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
sftp.close();c.close()
assert json.loads((folder/'t0.json').read_bytes())['status']=='PASS_SUPPORTED_GRADIENT_BALANCE_CPU_CONTRACT'
(folder/'inventory.json').write_text(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(),
    transfer='SFTP text only; pipeline may still change',files=rows,binary_downloads=0),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(folder=str(folder),files=len(rows),bytes=sum(r['bytes'] for r in rows))))

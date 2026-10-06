from pathlib import Path
from datetime import datetime
import hashlib,json,paramiko

base=Path('C:/Users/gb/.codex_tmp')
proof=base/'foundation_recipe_v1_20261002'
published=json.loads((proof/'four_copy872_2025_pending.json').read_bytes())
bundles={hashlib.sha256(p.read_bytes()).hexdigest():dict(local=str(p),size=p.stat().st_size)
         for p in proof.glob('*.bundle')}
packet=base/'independent_evidence_draft/role_head_transport_retirement873'
assert not packet.exists();packet.mkdir()
code='known='+repr(bundles)+'\nexpected_head='+repr(published['head'])+'\n'+r'''from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_independent_role_heads as panel
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==expected_head
sources=panel.source_map();controls=panel.previous.require_controls()
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        assert '/data/gaob/Re-ID/Trifusion/tools/' not in cmd,cmd
qualified=[]
for p in sorted(Path('/tmp').glob('trifusion_*.bundle')):
    if p.is_file() and p.stat().st_size in {r['size'] for r in known.values()}:
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in known:
            assert p.resolve().parent==Path('/tmp').resolve() and p.stat().st_nlink==1
            qualified.append(dict(path=str(p),sha256=digest,bytes=p.stat().st_size,
                                  local_retained=known[digest],same_device=p.stat().st_dev==root.stat().st_dev))
assert qualified
j=root/'logs/independent_role_head_used_transport_retirement_20261006_873';assert not j.exists();j.mkdir()
record=dict(at=datetime.now().astimezone().isoformat(),head=expected_head,
            free_before=shutil.disk_usage(root).free,qualified=qualified,
            boundary='Only used own Git transport bundles byte-matching retained local copies. Repository HEAD and published evidence retained. No NN active.')
(j/'QUALIFIED.json').write_text(json.dumps(record,indent=2)+'\n')
with (j/'RETIREMENT.jsonl').open('x') as log:
    for item in qualified:
        p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];p.unlink()
        log.write(json.dumps(item)+'\n');log.flush()
assert panel.source_map()==sources
assert panel.previous.require_controls()['artifact_sha256']==controls['artifact_sha256']
result=dict(status='USED_OWN_TRANSPORT_RETIRED',at=datetime.now().astimezone().isoformat(),
    retired=len(qualified),retired_bytes=sum(r['bytes'] for r in qualified),
    free_after=shutil.disk_usage(root).free,required_start_bytes=panel.STORAGE_BYTES,
    files={str(p.relative_to(root)):panel.base.sha(p) for p in j.iterdir() if p.is_file()})
(j/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
result['files'][str((j/'COMPLETE.json').relative_to(root))]=panel.base.sha(j/'COMPLETE.json')
print(json.dumps(result))
'''
compile(code,'retire_used_transport873','exec');(packet/'SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status()
(packet/'REMOTE.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=status))+'\n')
assert status==0,error.decode();result=json.loads(data)
s=c.open_sftp()
for n,d in result['files'].items():
    p=packet/'received'/n;p.parent.mkdir(parents=True,exist_ok=True);s.get('/data/gaob/Re-ID/Trifusion/'+n,str(p))
    assert hashlib.sha256(p.read_bytes()).hexdigest()==d
s.close();c.close();print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))

"""Delete own disposable server transport bundles already verified and retained locally."""
from pathlib import Path
from datetime import datetime
import hashlib,json,re
import paramiko

private=Path('C:/Users/gb/.codex_tmp');proof=private/'foundation_recipe_v1_20261002';base=private/'independent_evidence_draft'
approved={}
for p in proof.glob('four_copy*_2025_pending.json'):
 match=re.fullmatch(r'four_copy(\d+)_2025_pending\.json',p.name)
 if match and 739<=int(match[1])<=841:
  n=int(match[1]);b=proof/f'target{n}.bundle'
  assert b.exists()
  v=json.loads(p.read_bytes());assert v['status']=='FOUR_COPIES_VERIFIED_2025_MIRROR_IO_PENDING'
  approved[n]=dict(head=v['head'],bundle_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),bytes=b.stat().st_size,
                   publication_proof_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert len(approved)==20
packet=base/'published_transport_retirement842';assert not packet.exists()
code='''from pathlib import Path
from datetime import datetime
import hashlib,json,re,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion');approved=APPROVED;rows=[]
for path in sorted(Path('/tmp').glob('trifusion_target*_2026100*.bundle')):
 match=re.fullmatch(r'trifusion_target(\\d+)_2026100[2-5]\\.bundle',path.name)
 if match and int(match[1]) in approved:
  p=approved[int(match[1])];assert path.stat().st_size==p['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==p['bundle_sha256']
  subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',p['head'],'HEAD'],check=True,stdout=subprocess.DEVNULL)
  rows.append(dict(path=str(path),section=match[1],**p))
assert rows
archive=root/'logs/published_transport_bundle_retirement_20261005_842';assert not archive.exists();archive.mkdir()
before=shutil.disk_usage(root).free
certificate=dict(status='OWN_COMPLETED_PUBLISHED_TRANSPORT_BUNDLES_VERIFIED',at=datetime.now().astimezone().isoformat(),rows=rows,
 boundary='Temporary /tmp publication transport only;SHA-equal full bundle retained locally,published commit ancestor and four-copy receipt retained. Not model/dataset/distance/gitobject deletion;current842 or pending bundle excluded.')
accept=archive/'ACCEPTANCE.json';accept.write_text(json.dumps(certificate,indent=2)+'\\n')
with (archive/'RETIREMENT.jsonl').open('x') as journal:
 for row in rows:
  Path(row['path']).unlink()
  journal.write(json.dumps(dict(path=row['path'],bundle_sha256=row['bundle_sha256'],bytes=row['bytes'],at=datetime.now().astimezone().isoformat()))+'\\n');journal.flush()
summary=dict(status='OWN_COMPLETED_TRANSPORT_BUNDLES_RETIRED',at=datetime.now().astimezone().isoformat(),count=len(rows),
 retired_bytes=sum(r['bytes'] for r in rows),disk_free_before=before,disk_free_after=shutil.disk_usage(root).free,
 acceptance_sha256=hashlib.sha256(accept.read_bytes()).hexdigest())
(archive/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\\n')
print(json.dumps(dict(summary=summary,files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in archive.iterdir()})))
'''.replace('APPROVED',repr(approved))
compile(code,'remote_published_transport_cleanup842.py','exec');packet.mkdir();(packet/'remote_source.py').write_text(code,encoding='utf-8')
client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -');stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(300)
data,error=out.read(),err.read();rc=out.channel.recv_exit_status();client.close()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=rc,at=datetime.now().astimezone().isoformat()))+'\n')
assert rc==0,error.decode();value=json.loads(data)
for name,row in value['files'].items():
 target=packet/'received'/Path(name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(row['text'].encode('utf-8'))
 assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
(packet/'SUMMARY.json').write_text(json.dumps(value['summary'],indent=2)+'\n',encoding='utf-8')
print(json.dumps(value['summary']))

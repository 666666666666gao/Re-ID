"""Recover exact raw text bytes after pathlib newline normalization was detected."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
OUT=Path(__file__).resolve().parent
manifest=json.loads((OUT/'remote_inventory.json').read_bytes())
needed=[]
for path,row in manifest['files'].items():
    target=OUT/'snapshots/remote'/path.lstrip('/')
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
        needed.append((path,row,target))
namespace={'__name__':'private_audit_transport'}
with open(os.devnull,'w',encoding='utf-8') as sink,contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
    helper=Path('C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py')
    exec(compile(helper.read_text(encoding='utf-8-sig'),str(helper),'exec'),namespace)
client=namespace['c'];sftp=client.open_sftp();results=[]
for path,row,target in needed:
    previous=target.read_bytes()
    preserved=OUT/'attempt_03_normalized_snapshots'/path.lstrip('/')
    preserved.parent.mkdir(parents=True,exist_ok=True);preserved.write_bytes(previous)
    with sftp.open(path,'rb') as stream:raw=stream.read()
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert raw.decode('utf-8-sig').replace('\r\n','\n')==previous.decode('utf-8')
    target.write_bytes(raw)
    results.append(dict(path=path,original_sha256=row['sha256'],original_bytes=len(raw),normalized_snapshot_bytes=len(previous),exact_raw_snapshot_restored=True,scientific_source_changed=False))
sftp.close();client.close()
report={'status':'PASS_RAW_TEXT_SNAPSHOT_RECOVERY','cause':'pathlib.read_text normalized CRLF in 22 remotely hashed text files during snapshot serialization; original remote hashes and scientific data were unaffected.','method':'read-only SFTP binary reads of only the mismatching source text files','recovered':results}
(OUT/'raw_text_snapshot_recovery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'status':report['status'],'files':len(results)},indent=2))

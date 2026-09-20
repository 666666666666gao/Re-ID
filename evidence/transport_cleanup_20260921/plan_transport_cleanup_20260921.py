from pathlib import Path
import hashlib,json,subprocess
t=Path('D:/Program Files/UserCache/gb/codex/tmp')
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
j=json.loads((t/'inventory_trifusion_transport_20260921.json').read_bytes())
assert subprocess.check_output(['git','-c','http.sslBackend=openssl','ls-remote','origin','refs/heads/main'],cwd=p,text=True).split()[0]==j['head']
keep=[];eligible=[]
for x in j['files']:
    copies=[r/x['name'] for r in (t,Path('C:/Users/gb/.codex_tmp')) if (r/x['name']).is_file()]
    identical=[q for q in copies if q.stat().st_size==x['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==x['sha256']]
    if identical and all(r['ancestor_of_current_head'] for r in x['refs']):
        eligible.append(dict(x,local_copy=str(identical[0]),local_copy_identical=True,eligible=True))
    else:keep.append(x)
out=dict(head=j['head'],files=eligible,retained=keep,logical_bytes=sum(x['bytes'] for x in eligible))
(t/'trifusion_transport_retention_20260921.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(eligible=len(eligible),retained=len(keep),bytes=out['logical_bytes'])))

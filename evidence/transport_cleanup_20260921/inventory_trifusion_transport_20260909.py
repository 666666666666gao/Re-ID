from pathlib import Path
import hashlib,json,shutil,subprocess
from datetime import datetime

root=Path('/root/autodl-tmp/trifusion-v2')
repo=root/'TriFusion-ReID';folder=root/'transport'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
rows=[]
for p in sorted(folder.glob('*.bundle')):
    assert p.resolve().parent==folder.resolve() and not p.is_symlink()
    b=p.read_bytes()
    refs=[]
    for line in subprocess.check_output(['git','bundle','list-heads',str(p)],text=True).splitlines():
        oid,name=line.split(' ',1)
        test=subprocess.run(['git','merge-base','--is-ancestor',oid,head],cwd=repo,capture_output=True)
        assert test.returncode in (0,1),test.stderr
        refs.append(dict(oid=oid,name=name,ancestor_of_current_head=test.returncode==0))
    rows.append(dict(path=str(p),name=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),refs=refs))
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),head=head,folder=str(folder),free_bytes=shutil.disk_usage(root).free,files=rows,logical_bytes=sum(x['bytes'] for x in rows))))

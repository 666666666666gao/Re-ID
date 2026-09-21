from pathlib import Path
import subprocess,hashlib,json,shutil
root=Path('/root/autodl-tmp/trifusion-v2');repo=root/'TriFusion-ReID';folder=root/'transport'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
rows=[]
for p in sorted(folder.glob('*.bundle')):
    assert not p.is_symlink() and p.resolve().parent==folder.resolve()
    refs=[]
    for line in subprocess.check_output(['git','bundle','list-heads',str(p)],cwd=repo,text=True).splitlines():
        oid,name=line.split(' ',1)
        refs.append(dict(oid=oid,name=name,ancestor_of_current_head=subprocess.run(['git','merge-base','--is-ancestor',oid,head],cwd=repo,capture_output=True).returncode==0))
    rows.append(dict(path=str(p),name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),refs=refs))
print(json.dumps(dict(head=head,free_bytes=shutil.disk_usage(root).free,output_free_bytes=shutil.disk_usage('/root/trifusion-storage').free,files=rows),indent=2))

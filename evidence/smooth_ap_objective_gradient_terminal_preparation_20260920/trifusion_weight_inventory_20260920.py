from pathlib import Path
import os,json,shutil,datetime,subprocess
roots=[Path('/root/autodl-tmp/trifusion-v2/artifacts'),Path('/root/trifusion-storage/artifacts')]
result={'observed_at':datetime.datetime.now().astimezone().isoformat(),'roots':[],'deleted_files':0}
for root in roots:
 rows=[]
 for base,dirs,files in os.walk(root,followlinks=False):
  for name in files:
   p=Path(base)/name
   if p.suffix in ('.pt','.pth','.ckpt'):
    st=p.stat();rows.append({'path':str(p),'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'links':st.st_nlink,'symlink':p.is_symlink(),'mtime':st.st_mtime})
 result['roots'].append({'root':str(root),'free_bytes':shutil.disk_usage(root).free,'weights':rows,'logical_bytes':sum(x['bytes'] for x in rows)})
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
result['head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
result['transport']=[{'path':str(p),'bytes':p.stat().st_size} for p in Path('/root/autodl-tmp/trifusion-v2/transport').glob('*.bundle')]
print(json.dumps(result))

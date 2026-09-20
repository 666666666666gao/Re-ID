from pathlib import Path
import json,subprocess,shutil,datetime,hashlib
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID');base=Path('/root/trifusion-storage/artifacts')
r=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),user=subprocess.check_output(['whoami'],text=True).strip(),hostname=subprocess.check_output(['hostname'],text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),status=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip(),main_free=shutil.disk_usage(repo).free,output_free=shutil.disk_usage(base).free)
r['gradient_paths']=[str(p) for p in base.glob('msvr310_smooth_ap_objective_gradients*')]
r['audit_paths']=[str(p) for p in base.glob('*smooth*coverage*audit*')]
r['processes']=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit() and (p/'cmdline').exists():
  raw=(p/'cmdline').read_bytes()
  if b'python' in raw and (b'smooth_ap' in raw or b'coverage' in raw):r['processes'].append(dict(pid=int(p.name),command=raw.replace(b'\0',b' ').decode(errors='replace')))
r['master_sha256']=hashlib.sha256((repo/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md').read_bytes()).hexdigest()
print(json.dumps(r,ensure_ascii=False))

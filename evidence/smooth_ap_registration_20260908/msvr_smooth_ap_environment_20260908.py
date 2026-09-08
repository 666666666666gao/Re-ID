from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess,sys
import torch

p=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
output=Path('/root/trifusion-storage/artifacts')
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),user=subprocess.check_output(['whoami'],text=True).strip(),hostname=subprocess.check_output(['hostname'],text=True).strip(),repo=str(p),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),python=sys.version,executable=sys.executable,torch=torch.__version__,cuda_build=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip(),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip(),main_free=shutil.disk_usage(p).free,output_free=shutil.disk_usage(output).free,old_role_set_wrapper_exists=Path('/proc/35302').exists(),old_role_set_q1_exists=Path('/proc/36320').exists(),environment_rebuilt=False,model_forwards=0,optimizer_updates=0)))

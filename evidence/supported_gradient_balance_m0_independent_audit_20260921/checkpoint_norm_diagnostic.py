import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
os.nice(10)
from pathlib import Path
import json,math
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
run=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/m0')
rows=[]
for folder in sorted(run.glob('fold_*')):
    p=torch.load(folder/'roles_m0.pth',map_location='cpu',weights_only=True)
    a=json.loads((folder/'memory_steps.jsonl').read_text().splitlines()[-1])
    for role in ('cnn','transformer','mamba'):
        names=[n for n in p['binding']['trainable_names'] if n.startswith('encoder.'+role+'_')]
        v=math.sqrt(sum(float(p['role_state_dict'][n].double().square().sum()) for n in names))
        saved=a['actual_parameter_updates'][role]['second_norm']
        rows.append(dict(endpoint=folder.name,role=role,cpu=v,gpu_recorded=saved,absolute_difference=abs(v-saved),relative_difference=abs(v-saved)/saved))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(rows=rows,model_forwards=0,optimizer_updates=0,cuda_initialized=False),indent=2))

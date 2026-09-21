from pathlib import Path
import json,hashlib,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/q1/fold_0_control')
r=json.loads((root/'receipt.json').read_bytes());path=Path(r['checkpoint'])
assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha256']
state=torch.load(path,map_location='cpu',weights_only=True)['role_state_dict']
roles={}
for role in ['cnn','transformer','mamba']:
    selected={k:v for k,v in state.items() if k.startswith('encoder.'+role+'_')}
    roles[role]=dict(tensors=len(selected),numel=sum(v.numel() for v in selected.values()),dtypes=sorted({str(v.dtype) for v in selected.values()}))
assert sum(r['tensors'] for r in roles.values())==189
n=sum(r['numel'] for r in roles.values())
print(json.dumps(dict(status='READ_ONLY_CHECKPOINT_STORAGE_ACCOUNTING',checkpoint_sha256=r['checkpoint_sha256'],roles=roles,encoder_elements=n,one_fp32_m_v_pair_bytes=n*8,two_task_fp32_m_v_bytes=n*16,incremental_moments_bytes=n*8,scope='Tensor storage only; excludes gradient buffers, framework allocator, counters, heads and activations. No optimizer implemented or model forward.',model_forwards=0,updates=0,cuda_initialized=torch.cuda.is_initialized()),indent=2))

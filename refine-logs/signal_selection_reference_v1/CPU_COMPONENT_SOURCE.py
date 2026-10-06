expected='1f8fef4dcadb89f5097f83b41a699d9f801a23c88ca3eea73b3ad973a56952a9'
from pathlib import Path
from datetime import datetime
import copy,hashlib,json,os
import torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
torch.set_num_threads(4)
path=Path('/data/gaob/Re-ID/Trifusion/comparators/Signal-cd1b0a6/modeling/AddModule/useA.py')
assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
namespace={};exec(compile(path.read_text(),str(path),'exec'),namespace)
rows=[]
for dataset,k in (('RGBNT201',80),('MSVR310',64),('RGBNT100',112)):
 torch.manual_seed(42)
 model=namespace['Select_Interactive_Module'](512,k=k)
 for name in ('W_q','W_k','W_v'):getattr(model.token_selection,name).requires_grad_(False)
 all_model=copy.deepcopy(model)
 assert all(torch.equal(p,all_model.state_dict()[n]) for n,p in model.state_dict().items())
 patches=[torch.randn(2,128,512,requires_grad=True) for _ in range(3)]
 globals=[torch.randn(2,512,requires_grad=True) for _ in range(3)]
 selected=model.token_selection(*patches,*globals)
 coverage=[float((p.abs().sum(-1)>0).float().mean()) for p in selected]
 assert all(c>=k/128 for c in coverage)
 target=torch.randn(2,1536)
 activities=[]
 for arm,current in (('masked',model),('all_patch',all_model)):
  optimizer=torch.optim.Adam([p for p in current.parameters() if p.requires_grad],lr=0.00035)
  before={n:p.detach().clone() for n,p in current.named_parameters()}
  live=set()
  for step in range(8):
   optimizer.zero_grad(set_to_none=True)
   output=current(*patches,*globals) if arm=='masked' else current.modal_interactive(*patches,*globals)
   assert output.shape==(2,1536) and torch.isfinite(output).all()
   loss=(output-target).square().mean();loss.backward()
   live.update(n for n,p in current.named_parameters() if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
   assert all(p.grad is None for n,p in current.named_parameters() if n.startswith('token_selection.'))
   optimizer.step()
  active={n for n,p in current.named_parameters() if p.requires_grad}
  assert live==active
  changes={n:float((p.detach()-before[n]).abs().max()) for n,p in current.named_parameters()}
  assert all(changes[n]>0 for n in active)
  assert all(changes[n]==0 for n in changes if n.startswith('token_selection.'))
  activities.append(dict(arm=arm,active_tensors=len(active),active_parameters=sum(p.numel() for p in current.parameters() if p.requires_grad),changes=changes))
 rows.append(dict(dataset=dataset,topk=k,selected_nonzero_fraction=coverage,initial_total_parameters=sum(p.numel() for p in model.parameters()),activities=activities))
print(json.dumps(dict(status='CPU_PINNED_SIM_COMPONENT_PASS',at=datetime.now().astimezone().isoformat(),source_sha256=expected,rows=rows,
 boundary='Synthetic B2/128patch/512width CPU component only. No CLIP/images/heads/full model/M0/retrieval result. Six selector tensors unchanged and gradient-free; original interaction trainable in both arms.')))

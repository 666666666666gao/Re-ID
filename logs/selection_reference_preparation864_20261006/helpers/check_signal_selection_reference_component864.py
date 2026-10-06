from pathlib import Path
from datetime import datetime
import hashlib
import json
import paramiko

base=Path('C:/Users/gb/.codex_tmp')
packet=base/'independent_evidence_draft/signal_selection_reference_v1'
assert not (packet/'CPU_COMPONENT.json').exists()
source=base/'rt_complete862/received/comparators/Signal-cd1b0a6/modeling/AddModule/useA.py'
expected=hashlib.sha256(source.read_bytes()).hexdigest()
code='expected='+repr(expected)+'\n'+r'''from pathlib import Path
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
'''
compile(code,'CPU_SIM_COMPONENT864','exec')
(packet/'CPU_COMPONENT_SOURCE.py').write_text(code,encoding='utf-8')
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('CUDA_VISIBLE_DEVICES= /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B -');i.write(code);i.channel.shutdown_write();o.channel.settimeout(180)
data,error=o.read(),e.read();status=o.channel.recv_exit_status();c.close()
(packet/'CPU_COMPONENT.json').write_bytes(data);(packet/'CPU_COMPONENT_STDERR.txt').write_bytes(error)
(packet/'CPU_COMPONENT_EXIT.json').write_text(json.dumps(dict(exit_code=status,at=datetime.now().astimezone().isoformat()))+'\n')
assert status==0,error.decode()
result=json.loads(data)
print(json.dumps(dict(status=result['status'],at=result['at'],rows=[dict(dataset=r['dataset'],topk=r['topk'],coverage=r['selected_nonzero_fraction'],activities=[dict(arm=a['arm'],tensors=a['active_tensors'],parameters=a['active_parameters']) for a in r['activities']]) for r in result['rows']]),indent=2))

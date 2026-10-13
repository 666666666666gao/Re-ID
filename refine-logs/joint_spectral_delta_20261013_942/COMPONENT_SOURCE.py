"""First CPU arithmetic/autograd witness for the new readout component."""
import base64
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

w=Path(__file__).resolve().parent
r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
path=r/'modeling/trifusion/joint_spectral_delta.py'
source=path.read_bytes();source_sha=hashlib.sha256(source).hexdigest()
assert not (w/'COMPONENT_REQUEST.json').exists()
(w/'COMPONENT_REQUEST.json').write_bytes((json.dumps(dict(at=datetime.now().astimezone().isoformat(),source_sha256=source_sha,
    mode='CPU synthetic component only; no CLIP/Mamba/full model/optimizer/retrieval',CUDA_VISIBLE_DEVICES=''),indent=2)+'\n').encode())
code='source='+repr(source.decode())+'\nexpected='+repr(source_sha)+'\n'+r'''
import hashlib,json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from datetime import datetime
torch.set_num_threads(1)
assert hashlib.sha256(source.encode()).hexdigest()==expected
scope={};exec(compile(source,'joint_spectral_delta.py','exec'),scope)
Module=scope['JointSpectralDelta'];torch.manual_seed(42)
joint=Module(mode='joint_delta');add=Module(mode='add_delta');absolute=Module(mode='joint_absolute')
add.load_state_dict(joint.state_dict(),strict=True);absolute.load_state_dict(joint.state_dict(),strict=True)
assert all(torch.equal(joint.state_dict()[k],add.state_dict()[k]) and torch.equal(joint.state_dict()[k],absolute.state_dict()[k]) for k in joint.state_dict())
reference=[torch.randn(4,3,16,128) for _ in range(3)]
processed=[value.clone() for value in reference]
for value in processed:value[:,0]+=torch.randn_like(value[:,0])*.1
readout=torch.randn(1536,384)*.001
calls=0
def run(module,actual,base,weight):
    global calls
    calls+=1
    return module(actual,base,weight)[0]
zero_joint=run(joint,reference,reference,readout);zero_add=run(add,reference,reference,readout)
assert torch.equal(zero_joint,torch.zeros_like(zero_joint)) and torch.equal(zero_add,torch.zeros_like(zero_add))
first_joint=run(joint,processed,reference,readout);first_add=run(add,processed,reference,readout)
flipped_actual=[value.clone() for value in processed];flipped_base=[value.clone() for value in reference]
for value in flipped_actual+flipped_base:value[:,1].neg_()
second_joint=run(joint,flipped_actual,flipped_base,readout);second_add=run(add,flipped_actual,flipped_base,readout)
assert torch.allclose(first_joint,-second_joint,atol=1e-5,rtol=1e-5)
assert torch.allclose(first_add,second_add,atol=1e-5,rtol=1e-5)
actual=[torch.randn(4,3,16,128,requires_grad=True) for _ in range(3)]
base=[torch.randn(4,3,16,128,requires_grad=True) for _ in range(3)]
weight=readout.clone().requires_grad_();global_feature=torch.randn(4,1536,requires_grad=True)
correction=run(joint,actual,base,weight)
fused=torch.nn.functional.normalize(global_feature.detach()+.1*correction,dim=1)
(fused*torch.randn_like(fused)).sum().backward()
assert global_feature.grad is None
assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0 for value in actual+base)
assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0 for value in joint.parameters())
assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum()>0
assert correction.shape==(4,1536)
absolute_value=run(absolute,reference,reference,readout)
assert absolute_value.norm()>0
print(json.dumps(dict(status='CPU_COMPONENT_ARITHMETIC_AUTOGRAD_PASS',at=datetime.now().astimezone().isoformat(),source_sha256=expected,
    component_forward_calls=calls,backward_calls=1,optimizer_updates=0,added_parameters=sum(p.numel() for p in joint.parameters()),
    same_parameter_state_three_modes=True,matched_reference_zero=True,
    product_cross_modal_sign_maxdiff=float((first_joint+second_joint).abs().max()),
    additive_other_modal_sign_maxdiff=float((first_add-second_add).abs().max()),
    all_factor_tensor_gradients_finite_nonzero=True,processed_and_reference_input_gradients_present=True,shared_global_gradient_absent=True,
    full_model_calls=0,clip_or_mamba_calls=0,real_images=0,retrieval_calls=0,
    boundary='Synthetic readout only. semantic_reference helper and real model integration not exercised; no ReID gain, non-redundancy, cause or novelty evidence.')))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','env','CUDA_VISIBLE_DEVICES=','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code,text=True,capture_output=True)
(w/'COMPONENT.stdout').write_bytes(run.stdout.encode());(w/'COMPONENT.stderr').write_bytes(run.stderr.encode())
(w/'COMPONENT_EXIT.json').write_bytes((json.dumps(dict(exit_code=run.returncode,at=datetime.now().astimezone().isoformat()),indent=2)+'\n').encode())
assert run.returncode==0,run.stderr
value=json.loads(run.stdout)
(w/'COMPONENT_RESULT.json').write_bytes((json.dumps(value,indent=2)+'\n').encode())
print(json.dumps(value))

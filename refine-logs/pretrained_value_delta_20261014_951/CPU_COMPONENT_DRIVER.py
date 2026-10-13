"""CPU component and source binding checks, no GPU/model/optimizer run."""
import ast,base64,hashlib,json,subprocess
from pathlib import Path

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
names=['modeling/trifusion/pretrained_value_delta.py','tools/run_pretrained_value_delta.py','tools/queue_pretrained_value_delta.py']
assert not (w/'VALUE_CPU_CHECK_951.json').exists()
data={name:(r/name).read_bytes() for name in names}
for name,value in data.items():ast.parse(value.decode(),filename=name)
source=base64.b64encode(data[names[0]]).decode()
code='source='+repr(source)+'\n'+r'''
import ast,base64,hashlib,json,os
from types import SimpleNamespace
assert os.environ['CUDA_VISIBLE_DEVICES']==''
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(2);torch.manual_seed(42)
tree=ast.parse(base64.b64decode(source).decode());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='value_delta_readout')
module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
scope={'torch':torch,'F':F};exec(compile(module,'value_delta_readout','exec'),scope)
roles=SimpleNamespace(query_projections=nn.ModuleList(nn.Linear(128,128) for _ in range(3)),
    key_projections=nn.ModuleList(nn.Linear(128,128) for _ in range(3)))
processed=[torch.randn(2,3,16,128,requires_grad=True) for _ in range(3)]
reference=[torch.randn(2,3,16,128,requires_grad=True) for _ in range(3)]
keys=[torch.randn(2,3,128,128,requires_grad=True) for _ in range(3)]
values=torch.randn(2,3,128,512)
rows=[]
for mode in ('semantic_reference','uniform_reference'):
    result,details=scope['value_delta_readout'](roles,processed,reference,keys,values,mode)
    assert result.shape==(2,1536) and details['attention_change'].shape==(2,3,3,16,128)
    shift=torch.randn(2,3,1,512)
    shifted,_=scope['value_delta_readout'](roles,processed,reference,keys,values+shift,mode)
    difference=float((result-shifted).abs().max());assert difference<3e-6
    gradient=torch.autograd.grad(result.square().mean(),details['attention_change'],retain_graph=True)[0]
    assert bool(torch.isfinite(gradient).all()) and bool((gradient!=0).any())
    common=[*processed,*keys,*roles.query_projections.parameters(),*roles.key_projections.parameters()]
    grads=torch.autograd.grad(result.square().mean(),common,retain_graph=True)
    assert all(bool(torch.isfinite(g).all()) and bool((g!=0).any()) for g in grads)
    if mode=='semantic_reference':
        refs=torch.autograd.grad(result.square().mean(),reference,retain_graph=True)
        assert all(bool(torch.isfinite(g).all()) and bool((g!=0).any()) for g in refs)
        zero,_=scope['value_delta_readout'](roles,processed,processed,keys,values,mode)
        assert torch.count_nonzero(zero)==0
    rows.append(dict(mode=mode,conditional_common_value_shift_max_difference=difference,
        attention_row_sum_max_abs=float(details['attention_change'].sum(-1).abs().max()),actual_change_loss_gradient='finite_nonzero'))
print(json.dumps(dict(status='CPU_VALUE_CHANGE_COMPONENT_PASS',rows=rows,full_model_forwards=0,optimizer_updates=0,cuda_visible_devices='')))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','env','CUDA_VISIBLE_DEVICES=',
    'PYTHONDONTWRITEBYTECODE=1','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code,text=True,capture_output=True)
(w/'VALUE_CPU_CHECK_951.stdout').write_bytes(run.stdout.encode());(w/'VALUE_CPU_CHECK_951.stderr').write_bytes(run.stderr.encode())
(w/'VALUE_CPU_CHECK_951_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout);value['source_sha256']={name:hashlib.sha256(content).hexdigest() for name,content in data.items()}
(w/'VALUE_CPU_CHECK_951.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

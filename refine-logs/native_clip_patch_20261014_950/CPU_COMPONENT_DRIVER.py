"""CPU mathematics/source check in the existing remote environment, no CUDA."""
import ast, base64, hashlib, json, subprocess
from pathlib import Path

w = Path(__file__).resolve().parent
r = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
assert not (w / 'NATIVE_PATCH_CPU_950.json').exists()
names = ['modeling/trifusion/native_clip_patch_evidence.py', 'tools/diagnose_native_clip_patch.py']
data = {name: (r / name).read_bytes() for name in names}
for name, value in data.items():
    ast.parse(value.decode(), filename=name)
reg = json.loads((w / 'REGISTRATION_FINAL.json').read_bytes())['source_sha256']
wrappers = {name: digest for name, digest in reg.items()
            if name.startswith('comparators/Signal-cd1b0a6/modeling/')
            and name.endswith(('/meta_arch.py', '/make_model.py'))}
assert wrappers
request = dict(sources={name: base64.b64encode(value).decode() for name, value in data.items()},
    official=base64.b64encode((w / 'clearclip_source950_gitpin/open_clip/transformer.py').read_bytes()).decode(),
    wrappers=wrappers)
code = 'request=' + repr(request) + '\n' + r'''
import ast,base64,hashlib,json,math,os
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import torch
import torch.nn.functional as F
torch.set_num_threads(2);torch.manual_seed(42)
source=base64.b64decode(request['sources']['modeling/trifusion/native_clip_patch_evidence.py']).decode()
scope={};exec(compile(source,'native_clip_patch_evidence.py','exec'),scope)
official=ast.parse(base64.b64decode(request['official']).decode())
method=next(node for node in ast.walk(official) if isinstance(node,ast.FunctionDef) and node.name=='custom_attn')
module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module)
reference={'F':F,'torch':torch};exec(compile(module,'pinned_ClearCLIP_custom_attn','exec'),reference)
rows=[]
for length,batch,width,heads in ((17,2,16,4),(129,1,768,12)):
    attention=torch.nn.MultiheadAttention(width,heads,dropout=0)
    x=torch.randn(length,batch,width)
    before={name:value.clone() for name,value in attention.state_dict().items()}
    ours=scope['query_self_attention'](attention,x)
    expected=reference['custom_attn'](None,attention,x,model_type='ClearCLIP')
    diff=float((ours-expected).abs().max())
    assert torch.allclose(ours,expected,atol=2e-6,rtol=2e-6)
    assert all(torch.equal(value,attention.state_dict()[name]) for name,value in before.items())
    assert all(parameter.grad is None for parameter in attention.parameters())
    rows.append(dict(shape=list(x.shape),max_abs_difference=diff))
root=Path('/data/gaob/Re-ID/Trifusion');wrappers={}
for name,digest in request['wrappers'].items():
    value=(root/name).read_bytes();assert hashlib.sha256(value).hexdigest()==digest
    wrappers[name]=dict(bytes=len(value),sha256=digest,data=base64.b64encode(value).decode())
print(json.dumps(dict(status='CPU_QQ_MATH_MATCHES_PINNED_AUTHOR_FUNCTION',fixtures=rows,
    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],optimizer_updates=0,
    full_model_forwards=0,source_sha256={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in request['sources'].items()},
    wrapper_sources=wrappers)))
'''
options = ['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
command = ['ssh',*options,'-p','2026','gaob@172.19.12.138',
    'env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2',
    '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-']
run = subprocess.run(command,input=code,text=True,capture_output=True)
(w / 'NATIVE_PATCH_CPU_950.stdout').write_bytes(run.stdout.encode())
(w / 'NATIVE_PATCH_CPU_950.stderr').write_bytes(run.stderr.encode())
(w / 'NATIVE_PATCH_CPU_950_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n')
assert run.returncode == 0, run.stderr
value = json.loads(run.stdout)
out = w / 'clip_wrapper950';assert not out.exists();out.mkdir()
for name,item in value['wrapper_sources'].items():
    content=base64.b64decode(item['data']);assert hashlib.sha256(content).hexdigest()==item['sha256']
    (out / Path(name).name).write_bytes(content)
value['wrapper_sources']={name:{key:item[key] for key in ('bytes','sha256')} for name,item in value['wrapper_sources'].items()}
(w / 'NATIVE_PATCH_CPU_950.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))

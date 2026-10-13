"""Real CPU component hook plus independent two-vector comparison fixture."""
import ast,base64,hashlib,json,subprocess
from pathlib import Path
w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
names=['tools/diagnose_joint_spectral_delta_best.py','tools/compare_saved_region_reads.py']
files={name:base64.b64encode((r/name).read_bytes()).decode() for name in names}
assert not (w/'REGION_CHECK_947_REQUEST.json').exists()
(w/'REGION_CHECK_947_REQUEST.json').write_text(json.dumps({'sources':{name:hashlib.sha256((r/name).read_bytes()).hexdigest() for name in names}},indent=2)+'\n')
for name in names:ast.parse((r/name).read_text(encoding='utf-8'),filename=name)
code='files='+repr(files)+'\n'+r'''
import ast,base64,json,sys,types
from pathlib import Path
import torch
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'modeling'))
torch.set_num_threads(2)
from trifusion.joint_spectral_delta import JointSpectralDelta
source=base64.b64decode(files['tools/diagnose_joint_spectral_delta_best.py']).decode()
extract=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='extract')
hook=next(node for node in extract.body if isinstance(node,ast.FunctionDef) and node.name=='region_hook')
hook_module=ast.fix_missing_locations(ast.Module(body=[hook],type_ignores=[]))
torch.manual_seed(42)
processed=tuple(torch.randn(3,3,16,128) for _ in range(3));reference=tuple(torch.randn_like(value) for value in processed)
weight=torch.randn(1536,384)*.001
captures=[]
for mode in JointSpectralDelta.MODES:
    component=JointSpectralDelta(mode=mode);initial={name:value.clone() for name,value in component.state_dict().items()}
    regions={name:[] for name in ('processed_regions','reference_regions')};namespace=dict(torch=torch,regions=regions)
    exec(compile(hook_module,'<actual_extract_region_hook>','exec'),namespace)
    with torch.inference_mode():
        plain=component(processed,reference,weight)
        handle=component.register_forward_pre_hook(namespace['region_hook'])
        observed=component(processed,reference,weight);handle.remove()
    assert torch.equal(plain[0],observed[0])
    assert all(torch.equal(plain[1][name],observed[1][name]) for name in plain[1])
    assert all(torch.equal(initial[name],value) for name,value in component.state_dict().items())
    assert all(parameter.grad is None for parameter in component.parameters())
    assert len(regions['processed_regions'])==len(regions['reference_regions'])==1
    assert torch.equal(regions['processed_regions'][0],torch.stack(processed,dim=1))
    assert torch.equal(regions['reference_regions'][0],torch.stack(reference,dim=1))
    captures.append(dict(mode=mode,region_shape=list(regions['processed_regions'][0].shape),output_difference=0))
module=types.ModuleType('tools.compare_saved_region_reads');module.__file__=str(root/'tools/compare_saved_region_reads.py');module.__package__='tools'
exec(compile(base64.b64decode(files['tools/compare_saved_region_reads.py']),module.__file__,'exec'),module.__dict__)
q=torch.zeros(1,3,3,16,128);q[:,:,:,:8,0]=1;q[:,:,:,8:,1]=1
g=torch.cat((q.clone(),torch.zeros_like(q)),dim=0);g[1,:,:,:,0]=1
scores=module.region_scores(q,g)
assert torch.allclose(scores['mean_pairs'],torch.tensor([[.5,.5]]),atol=1e-6,rtol=0)
assert torch.allclose(scores['symmetric_maxsim'],torch.tensor([[1.,.75]]),atol=1e-6,rtol=0)
perm=torch.randperm(16);reordered=module.region_scores(q[:,:,:,perm],g[:,:,:,perm])
assert all(torch.allclose(scores[name],reordered[name],atol=1e-6,rtol=0) for name in scores)
one=torch.zeros_like(q);one[:,:,:,:,0]=1;equal=module.region_scores(one,one)
assert all(torch.allclose(value,torch.ones(1,1),atol=1e-6,rtol=0) for value in equal.values())
print(json.dumps(dict(status='CPU_REGION_HOOK_AND_COMPARATOR_FIXTURE_PASS',real_component_forwards=6,hooked_forwards=3,
    comparison_fixture_calls=3,optimizer_updates=0,full_clip_mamba_calls=0,real_images=0,
    captures=captures,analytical_mean_scores=scores['mean_pairs'].tolist(),analytical_max_scores=scores['symmetric_maxsim'].tolist(),
    boundary='Actual AST hook registered on real readout components; unchanged outputs/state, ordered processed/reference tensors. Independent two-basis fixture and permutation check only; no production loader, full extractor/model, selected checkpoint or retrieval qualification.')))
'''
opt=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*opt,'-p','2026','gaob@172.19.12.138','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1',
    '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code,text=True,capture_output=True)
(w/'REGION_CHECK_947.stdout').write_bytes(run.stdout.encode());(w/'REGION_CHECK_947.stderr').write_bytes(run.stderr.encode())
(w/'REGION_CHECK_947_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout.splitlines()[-1]);(w/'REGION_CHECK_947_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))

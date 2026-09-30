"""CPU-only reconstruction of production factory and independent strict loads; no forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path('/data/gaob/Re-ID/Trifusion')
OUT=ROOT/'.codex_tmp/integrity678_factory'
if len(sys.argv)==1:
    OUT.mkdir(exist_ok=False)
    jobs=[]
    for ds in ('RGBNT201','RGBNT100','MSVR310'):
        with (OUT/(ds+'.log')).open('x') as f:
            child=subprocess.Popen([sys.executable,'-u','-B',__file__,ds],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            code=child.wait()
        jobs.append({'dataset':ds,'pid':child.pid,'exit_code':code})
        (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
        print(json.dumps(jobs[-1]),flush=True)
    raise SystemExit(0 if all(j['exit_code']==0 for j in jobs) else 1)
import torch
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
# Only device moves are redirected in this disposable audit process. Sources unchanged.
device_moves=[]
def cpu_cuda(self,*args,**kwargs):
    device_moves.append(type(self).__name__+'.cuda')
    return self
original_to=torch.nn.Module.to
def cpu_to(self,*args,**kwargs):
    if args and args[0]=='cuda':
        device_moves.append(type(self).__name__+'.to(cuda)')
        args=('cpu',*args[1:])
    return original_to(self,*args,**kwargs)
torch.nn.Module.cuda=cpu_cuda
torch.Tensor.cuda=cpu_cuda
torch.nn.Module.to=cpu_to
sys.path.insert(0,str(ROOT))
from functools import partial
from types import SimpleNamespace
from tools import run_patch_memory_roles as patch
from tools.queue_correspondence_roles import BASELINES,PROTOCOLS,SOURCE,WEIGHTS
import inspect
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
ds=sys.argv[1]
matrix_path=ROOT/'logs/patch_memory_roles_recovery_20261001/accepted_matrix.json'
matrix=json.loads(matrix_path.read_text())
protocol_path=PROTOCOLS/(ds+'.json')
protocol=json.loads(protocol_path.read_text())
result={'dataset':ds,'started_at':datetime.now(timezone.utc).isoformat(),'matrix_sha256':sha(matrix_path),
        'script_sha256':sha(__file__),'neural_forward_executed':False,'gpu_use':False,'device_redirection':'process-only Module.cuda, Tensor.cuda, Module.to(cuda) to CPU','endpoints':[]}
for row in [r for r in matrix['rows'] if r['dataset']==ds]:
    entry=patch.entry
    patch.MEMORY_MODE=row['memory_mode']
    entry.CONDITION.update(query_mode='context',auxiliary_target='none')
    entry.runner.CorrespondenceTriFusion=partial(patch.PatchMemoryTriFusion,memory_mode=patch.MEMORY_MODE,**entry.CONDITION)
    weight,digest=BASELINES[ds]
    args=SimpleNamespace(dataset=ds,seed=42,epochs=50,width=128,m1=True,m2=True,m3=False,pred_weight=.1,
                         protocol=protocol_path,signal_source=SOURCE,clip_weight=WEIGHTS/'ViT-B-16.pt',baseline_checkpoint=WEIGHTS/weight,
                         baseline_sha256=digest,output_dir=Path(row['run_dir']),mode='evaluate')
    model,_,_,binding=patch.build(args,protocol)
    training=json.loads((Path(row['run_dir'])/'training.json').read_text())
    assert binding==training['initializer']
    init_hash=state_hash(model.state_dict())
    assert init_hash==row['initial_model_state_sha256']
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==row['trainable_parameters']
    assert all(p.device.type=='cpu' for p in model.parameters())
    assert all(not p.requires_grad for p in model.backbone.signal.parameters())
    baseline_hash=state_hash(model.backbone.signal.state_dict())
    assert baseline_hash==binding['signal_state_sha256']
    assert model.roles.mamba.__class__.__module__=='mamba_ssm.modules.mamba_simple'
    assert model.roles.mamba.__class__.__name__=='Mamba'
    assert model.roles.local_support.sum(1).tolist()==[9]*16
    mamba_source=Path(inspect.getfile(model.roles.mamba.__class__))
    assert not torch.cuda.is_initialized()
    evidence={'variant':row['variant'],'initial_model_state_sha256':init_hash,'initialization_binding_exact':True,
              'trainable_parameters':row['trainable_parameters'],'trainable_parameter_tensors':sum(p.requires_grad for p in model.parameters()),
              'signal_state_sha256':baseline_hash,'signal_frozen':True,'model_state_tensors':len(model.state_dict()),
              'mamba_class':model.roles.mamba.__class__.__module__+'.'+model.roles.mamba.__class__.__name__,
              'mamba_source':str(mamba_source),'mamba_source_sha256':sha(mamba_source),'loads':[]}
    expected_keys={n for n in model.state_dict() if not n.startswith(('backbone.signal.','teacher.'))}
    for kind,path,epoch in [('best',Path(row['run_dir'])/'best_map.pth',row['best_epoch']),('m0',Path(row['m0_run_dir'])/'m0_reload_probe.pth',0)]:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['schema']=='trifusion-patch-memory-roles-v1'
        assert payload['dataset']==ds and payload['seed']==42 and payload['epoch']==epoch
        assert payload['protocol_sha256']==sha(protocol_path) and payload['baseline_sha256']==digest
        assert payload['condition']==entry.CONDITION and payload['memory_mode']==row['memory_mode']
        assert payload['variants']=={'m1':True,'m2':True,'m3':False}
        assert set(payload['state'])==expected_keys
        merged=model.state_dict();merged.update(payload['state'])
        outcome=model.load_state_dict(merged,strict=True)
        assert not outcome.missing_keys and not outcome.unexpected_keys
        assert all(torch.equal(model.state_dict()[n],v) for n,v in payload['state'].items())
        assert state_hash(model.backbone.signal.state_dict())==baseline_hash
        evidence['loads'].append({'kind':kind,'path':str(path),'sha256':sha(path),'strict_load':'PASS','loaded_state_keys':len(expected_keys),
                                  'full_model_after_load_sha256':state_hash(model.state_dict()),'frozen_baseline_unchanged':True})
    result['endpoints'].append(evidence)
    del model
result['device_moves']=device_moves
result['cuda_initialized']=torch.cuda.is_initialized()
assert result['cuda_initialized'] is False
result['status']='CPU_PRODUCTION_FACTORY_INIT_AND_STRICT_LOAD_COMPLETE'
result['completed_at']=datetime.now(timezone.utc).isoformat()
(OUT/(ds+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'dataset':ds,'status':result['status'],'init_hashes':[x['initial_model_state_sha256'] for x in result['endpoints']]}),flush=True)

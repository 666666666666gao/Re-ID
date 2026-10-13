"""Sync new owned sources and check their real warm-environment bindings only."""
import ast,base64,hashlib,json,subprocess
from pathlib import Path

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((w/'COMPLETE_FINAL_950.json').read_bytes());reg=json.loads((w/'REGISTRATION_FINAL.json').read_bytes())
cpu=json.loads((w/'VALUE_CPU_CHECK_951.json').read_bytes())
assert all(sha(r/name)==digest for name,digest in cpu['source_sha256'].items())
assert not (w/'VALUE_BINDING_CHECK_951.json').exists()
names=[*cpu['source_sha256'],'refine-logs/pretrained_value_delta_20261014_951/EXPERIMENT_PLAN.md']
sources=dict(reg['source_sha256'],**{name:sha(r/name) for name in ('modeling/trifusion/native_clip_patch_evidence.py',
    'tools/diagnose_native_clip_patch.py','refine-logs/native_clip_patch_20261014_950/DIAGNOSIS_PLAN.md',*names)})
request=dict(sources=sources,files={name:base64.b64encode((r/name).read_bytes()).decode() for name in names})
code='request='+repr(request)+'\n'+r'''
import ast,base64,hashlib,json,os,sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in request['sources'].items():
    if name not in request['files']:assert sha(root/name)==digest,name
for name,encoded in request['files'].items():
    path=root/name;assert not path.exists() and path.resolve().is_relative_to(root.resolve())
    data=base64.b64decode(encoded);assert hashlib.sha256(data).hexdigest()==request['sources'][name]
    if name.endswith('.py'):ast.parse(data.decode(),filename=name)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
sys.path.insert(0,str(root))
from tools import run_pretrained_value_delta as entry
from tools import queue_pretrained_value_delta as panel
from trifusion.pretrained_value_delta import PretrainedValueDeltaTriFusion
from trifusion.global_task_role_heads import GlobalTaskRoleHeads
entry.configure()
assert entry.inner.RawFeatureSemanticTriFusion is PretrainedValueDeltaTriFusion
assert entry.inner.AuthorHeadEvidence is GlobalTaskRoleHeads
assert entry.inner.loss_values is entry.loss_values and entry.inner.M0Diagnostics is entry.ValueM0Diagnostics
assert entry.inner.foundation.build_core is entry.build_core
assert entry.inner.foundation.condition is entry.condition
assert panel.MODES==('semantic_reference','uniform_reference')
assert all(sha(root/name)==digest for name,digest in request['sources'].items())
print(json.dumps(dict(status='WARM_CPU_BINDINGS_PASS',source_sha256=request['sources'],cuda_visible_devices='',
    full_model_forwards=0,model_initializations=0,optimizer_updates=0,scope='Actual imports/configuration identities only; no model construction/M0 or runtime source qualification.')))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','env','CUDA_VISIBLE_DEVICES=',
    'PYTHONDONTWRITEBYTECODE=1','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code,text=True,capture_output=True)
(w/'VALUE_BINDING_CHECK_951.stdout').write_bytes(run.stdout.encode());(w/'VALUE_BINDING_CHECK_951.stderr').write_bytes(run.stderr.encode())
(w/'VALUE_BINDING_CHECK_951_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n');assert run.returncode==0,run.stderr
value=json.loads(run.stdout);(w/'VALUE_BINDING_CHECK_951.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({key:value[key] for key in ('status','model_initializations','full_model_forwards','optimizer_updates')}))

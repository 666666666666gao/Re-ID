"""Compile/import the future diagnosis in the warm CPU runtime, no NN."""
import ast,base64,hashlib,json,subprocess
from pathlib import Path
w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
name='tools/diagnose_joint_spectral_delta_best.py';data=(r/name).read_bytes()
ast.parse(data.decode(),filename=name)
code='source='+repr(base64.b64encode(data).decode())+'\n'+r'''
import base64,json,sys,types
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
module=types.ModuleType('tools.diagnose_joint_spectral_delta_best');module.__file__=str(root/'tools/diagnose_joint_spectral_delta_best.py');module.__package__='tools'
exec(compile(base64.b64decode(source),module.__file__,'exec'),module.__dict__)
assert module.runner is module.producer.inner.runner
assert callable(module.producer.inner.foundation.build) and callable(module.producer.inner.foundation.load)
assert callable(module.base.metadata) and callable(module.base.score) and callable(module.base.paired)
assert module.panel.MODES==('joint_delta','add_delta','joint_absolute')
assert module.KEYS==('shared_global','correction','raw_fused','fused')
print(json.dumps(dict(status='CPU_DIAGNOSTIC_IMPORT_PASS',model_constructions=0,neural_forwards=0,optimizer_updates=0,
    invocation_of_diagnose=0,boundary='Imports/linked APIs only; no real seal, selected model, extraction, parity or retrieval check.')))
'''
assert not (w/'REGION_CAPTURE_IMPORT_947_REQUEST.json').exists()
(w/'REGION_CAPTURE_IMPORT_947_REQUEST.json').write_text(json.dumps(dict(source=name,sha256=hashlib.sha256(data).hexdigest()),indent=2)+'\n')
opt=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*opt,'-p','2026','gaob@172.19.12.138','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1',
    '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code,text=True,capture_output=True)
(w/'REGION_CAPTURE_IMPORT_947.stdout').write_bytes(run.stdout.encode());(w/'REGION_CAPTURE_IMPORT_947.stderr').write_bytes(run.stderr.encode())
(w/'REGION_CAPTURE_IMPORT_947_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n')
assert run.returncode==0,run.stderr
value=json.loads(run.stdout.splitlines()[-1]);(w/'REGION_CAPTURE_IMPORT_947_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))

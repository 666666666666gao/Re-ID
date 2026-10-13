"""Warm CPU import/configuration check; no model construction or neural call."""
import ast, base64, hashlib, json, subprocess
from pathlib import Path
w = Path(__file__).resolve().parent
r = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
names = ['modeling/trifusion/joint_spectral_roles.py', 'tools/run_joint_spectral_delta.py',
         'tools/queue_joint_spectral_delta.py']
files = {name: base64.b64encode((r / name).read_bytes()).decode() for name in names}
for name in names:
    ast.parse((r / name).read_text(encoding='utf-8'), filename=name)
code = 'files=' + repr(files) + '\n' + r'''
import ast,base64,json,sys,types
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'modeling'))
for name, module_name in [('modeling/trifusion/joint_spectral_roles.py','trifusion.joint_spectral_roles'),
                          ('tools/run_joint_spectral_delta.py','tools.run_joint_spectral_delta'),
                          ('tools/queue_joint_spectral_delta.py','tools.queue_joint_spectral_delta')]:
    data=base64.b64decode(files[name]);ast.parse(data.decode(),filename=name)
    module=types.ModuleType(module_name);module.__file__=str(root/name)
    module.__package__=module_name.rpartition('.')[0];sys.modules[module_name]=module
    exec(compile(data,module.__file__,'exec'),module.__dict__)
driver=sys.modules['tools.run_joint_spectral_delta'];driver.configure();driver.inner.configure()
assert driver.inner.build_core is driver.build_core
assert driver.inner.foundation.build_core is driver.build_core
assert driver.inner.RawFeatureSemanticTriFusion is driver.JointSpectralSemanticTriFusion
assert driver.inner.clean.control.GlobalTokenTriFusion is driver.JointSpectralSemanticTriFusion
assert driver.inner.AuthorHeadEvidence.__name__=='GlobalTaskRoleHeads'
assert driver.inner.foundation.loss_values is driver.loss_values
assert driver.inner.M0Diagnostics is driver.SpectralM0Diagnostics
assert driver.inner.SCHEMA==driver.inner.foundation.SCHEMA==driver.SCHEMA
print(json.dumps(dict(status='CPU_IMPORT_CONFIGURATION_PASS',model_constructions=0,neural_forwards=0,
    optimizer_updates=0,builder='new_spectral_subclass',head='GlobalTaskRoleHeads',loss='raw_author_split',
    schema=driver.SCHEMA,new_source_files=len(files),boundary='No real initializer/M0/gradient qualification.')))
'''
(w / 'CONFIGURATION_REQUEST.json').write_text(json.dumps({'source_sha256': {
    name: hashlib.sha256((r/name).read_bytes()).hexdigest() for name in names}}, indent=2)+'\n')
options = ['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run = subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138',
    'env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1',
    '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'], input=code, text=True, capture_output=True)
(w/'CONFIGURATION.stdout').write_bytes(run.stdout.encode());(w/'CONFIGURATION.stderr').write_bytes(run.stderr.encode())
(w/'CONFIGURATION_EXIT.json').write_text(json.dumps({'exit_code':run.returncode})+'\n')
assert run.returncode==0, run.stderr
value=json.loads(run.stdout.splitlines()[-1]);(w/'CONFIGURATION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))

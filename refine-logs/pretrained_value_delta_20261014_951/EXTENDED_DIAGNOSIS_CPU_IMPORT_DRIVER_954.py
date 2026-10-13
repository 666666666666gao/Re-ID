"""Import pending diagnosis/report sources with CUDA hidden, no NN execution."""
import ast,base64,hashlib,json,subprocess
from pathlib import Path

w=Path(__file__).resolve().parent;r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
names=['tools/diagnose_pretrained_value_delta_best.py','tools/report_pretrained_value_delta.py']
data={name:(r/name).read_bytes() for name in names}
for name,value in data.items():ast.parse(value.decode(),filename=name)
assert not (w/'VALUE_DIAGNOSIS_CPU_CHECK_956.json').exists()
request=dict(files={name:base64.b64encode(value).decode() for name,value in data.items()},
    sources=json.loads((w/'VALUE_REGISTRATION_FINAL_951.json').read_bytes())['source_sha256'])
code='request='+repr(request)+'\n'+r'''
import ast,base64,hashlib,json,os,sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name)==digest for name,digest in request['sources'].items())
sys.path.insert(0,str(root));scopes={}
for name,encoded in request['files'].items():
    data=base64.b64decode(encoded);ast.parse(data.decode(),filename=name)
    scope={'__name__':'source_only_'+Path(name).stem,'__file__':str(root/name)}
    exec(compile(data.decode(),str(root/name),'exec'),scope);scopes[name]=scope
assert scopes['tools/diagnose_pretrained_value_delta_best.py']['SCHEMA']=='trifusion-pretrained-value-fixed-best-diagnosis-v1'
assert callable(scopes['tools/diagnose_pretrained_value_delta_best.py']['diagnose'])
assert scopes['tools/report_pretrained_value_delta.py']['panel'].MODES==('semantic_reference','uniform_reference')
assert all(sha(root/name)==digest for name,digest in request['sources'].items())
print(json.dumps(dict(status='WARM_CPU_POST_TERMINAL_SOURCE_IMPORT_PASS',model_initializations=0,neural_forwards=0,
    optimizer_updates=0,report_invocations=0,diagnosis_invocations=0,
    source_sha256={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in request['files'].items()},
    boundary='Import/name check only, not actual controller, report or frozen full-model diagnosis qualification. Live496 sources unchanged.')))
'''
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','env','CUDA_VISIBLE_DEVICES=',
    'PYTHONDONTWRITEBYTECODE=1','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code,text=True,capture_output=True)
(w/'VALUE_DIAGNOSIS_CPU_CHECK_956.stdout').write_bytes(run.stdout.encode());(w/'VALUE_DIAGNOSIS_CPU_CHECK_956.stderr').write_bytes(run.stderr.encode())
(w/'VALUE_DIAGNOSIS_CPU_CHECK_956_EXIT.json').write_text(json.dumps(dict(exit_code=run.returncode))+'\n');assert run.returncode==0,run.stderr
value=json.loads(run.stdout);(w/'VALUE_DIAGNOSIS_CPU_CHECK_956.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

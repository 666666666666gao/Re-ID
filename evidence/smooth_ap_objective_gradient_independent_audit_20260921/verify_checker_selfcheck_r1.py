"""Closed-form synthetic fixtures for the independent auditor, not experiment results."""
from pathlib import Path
import ast,hashlib,json,math
import numpy as np

path=Path(__file__).with_name('remote_terminal_verification.py')
module=ast.parse(path.read_text(encoding='utf-8'))
definitions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('smooth_loss','distribution')]
namespace=dict(np=np)
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),namespace)
ids=np.repeat(np.arange(8),8)
equal=np.zeros((64,64),dtype=np.float64)
easy=np.where(ids[:,None]==ids[None,:],.1,1.5)
hard=np.where(ids[:,None]==ids[None,:],1.5,.1)
assert abs(namespace['smooth_loss'](equal,ids)-.875)<1e-14
assert abs(namespace['smooth_loss'](easy,ids))<1e-14
assert abs(namespace['smooth_loss'](hard,ids)-57/61)<1e-14
extended_ids=np.r_[ids,[0,1,2,8]]
expected=1.-np.mean([(int((extended_ids==i).sum()))/len(extended_ids) for i in ids])
assert abs(namespace['smooth_loss'](np.zeros((64,68)),extended_ids)-expected)<1e-14
dist=namespace['distribution']([None,0.,0.,10.])
assert dist==dict(count=3,undefined=1,zeros=2,mean=10/3,median=0.,p10=0.,p90=8.,minimum=0.,maximum=10.)
assert namespace['distribution']([None,None])==dict(count=0,undefined=2,zeros=0,mean=None,median=None,p10=None,p90=None,minimum=None,maximum=None)
result=dict(status='PASS_INDEPENDENT_VERIFIER_CLOSED_FORM_FIXTURES',fixture_count=6,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),numpy_version=np.__version__,
            evaluation_type='synthetic_proxy_math_fixture',experiment_performance_claim=False)
Path(__file__).with_name('verifier_selfcheck.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))

from pathlib import Path
import json,math,inspect,hashlib
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/q1')
source=(repo/'tools/verify_msvr_supported_gradient_balance_stats.py').read_text()
old="((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))**.5"
assert source.count(old)==1
ns={};exec(compile(source.replace(old,"math.sqrt((states[role]['auxiliary']+1e-12)/(states[role]['rank']+1e-12))"),'scalar_diagnostic','exec'),ns)
failures=[];calls=0;current=None
def collect(a,b):
    global calls
    calls+=1
    assert math.isfinite(a) and math.isfinite(b)
    threshold=1e-7*max(1,abs(a),abs(b))
    if abs(a-b)>threshold:
        frame=inspect.currentframe().f_back
        f=frame
        while f and f.f_code.co_name!='verify_balance':f=f.f_back
        failures.append(dict(endpoint=current,line=frame.f_lineno,step=f.f_locals['index']+1,role=f.f_locals['role'],left=a,right=b,absolute_error=abs(a-b),original_threshold=threshold,ratio=abs(a-b)/threshold))
ns['close']=collect
rows=[]
for local in sorted(root.glob('fold_*')):
    current=local.name
    audits=[json.loads(s) for s in (local/'memory_steps.jsonl').read_text().splitlines()]
    training=json.loads((local/'training.json').read_text())
    result=ns['verify_balance'](audits,training,current.rsplit('_',1)[1],65)
    rows.append(dict(endpoint=current,controller_replay=result))
print(json.dumps(dict(status='DIAGNOSTIC_ONLY_ORIGINAL_FAILURES_RETAINED_NOT_VERIFICATION_PASS',close_comparisons=calls,violations=failures,endpoints=rows,source_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2))

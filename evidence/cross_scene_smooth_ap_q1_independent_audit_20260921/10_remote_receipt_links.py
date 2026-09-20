"""Read saved phase links and initial bindings only; no M0 audit replay."""
from pathlib import Path
import hashlib
import json
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=read(RUN/'q1/summary.json');m=read(RUN/'m0/summary.json');cpu=read(RUN/'m0_cpu.json')
assert q['m0_receipt_sha256']==sha(RUN/'m0/summary.json')==cpu['summary_sha256']
assert q['m0_verification_sha256']==sha(RUN/'m0_cpu.json')
assert m['status']=='PASS_ENGINEERING_ONLY' and cpu['status']=='PASS_COMPLETE_CROSS_SCENE_SMOOTH_AP_M0'
assert q['config_sha256']==m['config_sha256']
links=[]
for f,mf in zip(q['folds'],m['folds']):
    for e,row in f['endpoints'].items():
        old=mf['endpoints'][e]
        assert row['initialization']==old['initialization']
        assert row['training']['initial_state_sha256']==old['training']['initial_state_sha256']
        assert row['training']['initial_state_sha256']!=old['training']['final_state_sha256']
        links.append(dict(fold=f['fold'],endpoint=e,initial_state_sha256=row['training']['initial_state_sha256'],
                          initial_binding_matches_m0=True,m0_terminal_state_not_used=True))
print(json.dumps(dict(status='PASS_Q1_M0_RECEIPT_AND_INITIALIZATION_LINKS',m0_summary_sha256=sha(RUN/'m0/summary.json'),
    m0_cpu_sha256=sha(RUN/'m0_cpu.json'),q1_project_commit=q['project_commit'],links=links,
    m0_reexecuted=False,model_forwards=0,optimizer_updates=0)))

"""Close the exact verifier/analyzer execution and original-process provenance scope."""
from pathlib import Path
import hashlib,json
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
def js(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
vp=js(Path(str(root)+'_verification_pipeline.json'));ap=js(Path(str(root)+'_analysis_pipeline.json'));sp=js(Path(str(root)+'_pipeline.json'))
check=repo/'evidence/smooth_ap_objective_gradient_source_launch_20260920/verify_smooth_ap_objective_gradient_source_20260920.py'
analyze=repo/'evidence/smooth_ap_objective_gradient_analysis_preparation_20260920/analyze_smooth_ap_objective_gradients_20260920.py'
assert sha(check)==vp['checker_sha256']=='79802db8ef471e8fc495f625aa15321f2d9c1034b13afd82a8f08b1541fb2ef4'
assert sha(analyze)==ap['analyzer_sha256']=='a4337f88e93200d7738018a4288022a3a98f10a34c724fdda13a33d6849a0653'
contract=repo/'evidence/smooth_ap_objective_gradient_analysis_preparation_20260920/ANALYSIS_CONTRACT.json'
assert js(contract)['analyzer_sha256']==sha(analyze)
assert js(contract)['expected_steps']==1560 and js(contract)['expected_role_rows']==4680 and js(contract)['expected_groups']==72
files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in (check,analyze,contract)]
processes=[]
for label,pipeline in [('source',sp),('verification',vp),('analysis',ap)]:
    assert pipeline['status']=='COMPLETE' and pipeline['exit_code']==0
    for key in ('pid','child_pid'):
        pid=pipeline[key];processes.append(dict(stage=label,handle=key,pid=pid,proc_exists=Path('/proc',str(pid)).exists()))
print(json.dumps(dict(status='PASS_EXECUTED_SCRIPT_BINDINGS',files=files,process_handles=processes,all_original_pid_paths_absent=all(not r['proc_exists'] for r in processes),
                      scope='Exact verifier/analyzer files and the original registered pipeline handles; no whole-worktree immutability claim.')))

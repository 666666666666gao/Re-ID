from pathlib import Path
import hashlib,json,importlib.util

temp=Path('D:/Program Files/UserCache/gb/codex/tmp')
base=temp/'analyze_cross_scene_smooth_ap_q1_20260921.py'
target=temp/'analyze_supported_gradient_balance_q1_20260921.py'
assert not target.exists()
code=base.read_text(encoding='utf-8')
code=code.replace('PASS_COMPLETE_CROSS_SCENE_SMOOTH_AP_Q1','PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_Q1')
code=code.replace("('control','cross_scene')","('control','balanced')")
code=code.replace("('smooth_ap' if end=='control' else 'cross_scene_smooth_ap')","'cross_scene_smooth_ap'")
code=code.replace("groups['cross_scene']","groups['balanced']")
code=code.replace('Actual totals use different active objectives, while hard and AP columns preserve common definitions.','Both endpoints use the same scalar objectives. Saved original total/history norms describe the pre-controller path, not the applied candidate gradient; role-specific balancing is summarized separately.')
target.write_text(code,encoding='utf-8')
spec=importlib.util.spec_from_file_location('supported_terminal',target)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source=temp/'trifusion_supported_balance_m0_complete_20260921'
groups=[]
for path in sorted((source/'m0').glob('*/training.json')):
    tr=json.loads(path.read_bytes())
    rows=[json.loads(line) for line in (path.parent/'memory_steps.jsonl').read_text().splitlines()]
    result=module.analyze(rows,tr['steps'])
    assert result['counts']['steps']==tr['optimizer_steps']
    groups.append(dict(group=path.parent.name,result=result))
assert len(groups)==8 and sum(g['result']['counts']['steps'] for g in groups)==248
receipt=dict(status='PASS_DESCRIPTIVE_FUNCTION_ON_COMPLETE_M0_ONLY',groups=groups,
    script_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),base_script_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
    input_inventory_sha256=hashlib.sha256((source/'inventory.json').read_bytes()).hexdigest(),
    model_forwards=0,optimizer_updates=0,
    scope='Only reusable descriptive function checked on all completed M0 text. Complete Q1 main and terminal assertions have NOT run. No partial Q1 metrics used. Candidate applied gradient and AdamW updates require separate controller summary.')
out=temp/'supported_balance_terminal_analysis_m0_check_20260921.json'
assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],groups=len(groups),steps=248,output=str(out))))

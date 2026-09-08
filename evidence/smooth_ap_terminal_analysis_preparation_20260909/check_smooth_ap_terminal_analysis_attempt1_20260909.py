from pathlib import Path
import importlib.util,json,hashlib,ast
p=Path('C:/Users/gb/.codex_tmp');script=p/'analyze_smooth_ap_q1_terminal_20260909.py'
ast.parse(script.read_bytes());spec=importlib.util.spec_from_file_location('descriptive',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
root=p/'smooth_ap_m0_complete_20260908/m0';checked=[]
for f in [root/f'fold_{i}_{e}' for i in range(3) for e in ('control','smooth_ap')]+[root/f'overfit_{e}' for e in ('control','smooth_ap')]:
    tr=json.loads((f/'training.json').read_bytes());rows=list(map(json.loads,(f/'memory_steps.jsonl').read_text().splitlines()));a=mod.analyze(rows,tr['steps'])
    assert a['counts']['steps']==tr['optimizer_steps']==len(rows)
    assert a['counts']['anchor_exposures']==64*len(rows)
    assert a['counts']['history_vjp_record_forwards']==tr['extra_history_vjp_record_forwards']
    assert a['counts']['fresh_role_record_forwards']==tr['extra_fresh_role_record_forwards']
    assert sum(a['active_metric_step_counts'].values())==len(rows)
    for k,v in a['mean_components'].items():assert v==sum(s['components'][k] for s in tr['steps'])/len(rows)
    checked.append(dict(endpoint=f.name,steps=len(rows)))
a=json.loads((root/'fold_0_control/training.json').read_bytes())['steps'];b=json.loads((root/'fold_0_smooth_ap/training.json').read_bytes())['steps'];d=mod.paired_phase(a,b)
assert d['max_abs_total_loss_difference']==max(abs(x['loss']-y['loss']) for x,y in zip(a,b,strict=True))
r=dict(status='PASS_REAL_M0_LOG_FUNCTION_CHECK_ONLY',script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),checked=checked,steps=sum(x['steps'] for x in checked),scope='Full completed M0 source logs only; Q1 terminal intake,1560-step orchestration and scientific conclusions not executed.',q1_terminal_execution=False,model_forwards=0,optimizer_updates=0)
out=p/'smooth_ap_terminal_analysis_function_check_20260909.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))

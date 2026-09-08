from pathlib import Path
from collections import Counter
import ast,hashlib,json
t=Path('C:/Users/gb/.codex_tmp')
script=t/'analyze_role_set_q1_terminal_v2_20260908.py';tree=ast.parse(script.read_bytes())
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
assert [node.name for node in functions]==['analyze','paired_phase']
ns={'Counter':Counter};exec(compile(ast.Module(body=functions,type_ignores=[]),str(script),'exec'),ns)
root=t/'role_set_m0_complete_20260908/m0';checked=[]
folders=[root/f'fold_{fold}_{end}' for fold in range(3) for end in ('control','role_set')]+[root/f'overfit_{end}' for end in ('control','role_set')]
for p in folders:
    tr=json.loads((p/'training.json').read_bytes())
    rows=[json.loads(line) for line in (p/'memory_steps.jsonl').read_text().splitlines()]
    result=ns['analyze'](rows,tr['steps'])
    assert result['counts']['steps']==len(rows)==tr['optimizer_steps']
    assert result['counts']['anchor_exposures']==64*len(rows)
    assert len(result['mean_components'])==14
    for key,value in result['mean_components'].items():
        assert value==sum(s['components'][key] for s in tr['steps'])/len(rows)
    assert result['mean_losses']['actual_total']==sum(s['loss'] for s in tr['steps'])/len(rows)
    for key in ('unique_positions_histogram','unique_records_histogram','unique_negative_identities_histogram'):
        assert sum(result[key].values())==64*len(rows)
    checked.append(dict(endpoint=p.name,steps=len(rows),components=14))
a=json.loads((root/'fold_0_control/training.json').read_bytes())['steps']
b=json.loads((root/'fold_0_role_set/training.json').read_bytes())['steps']
phase=ns['paired_phase'](a,b)
assert phase['steps']==8 and len(phase['max_abs_component_difference'])==14
assert phase['max_abs_total_loss_difference']==max(abs(x['loss']-y['loss']) for x,y in zip(a,b,strict=True))
result=dict(status='PASS_REAL_M0_LOG_FUNCTION_CHECK_ONLY',script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),endpoints=checked,checked_steps=sum(x['steps'] for x in checked),q1_terminal_execution=False,scope='Only the descriptive functions on all eight completed M0 logs plus AST. No Q1 text consumed; terminal intake/1560-step orchestration not executed.',model_forwards=0,optimizer_updates=0)
out=t/'role_set_terminal_analysis_v2_check_20260908.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

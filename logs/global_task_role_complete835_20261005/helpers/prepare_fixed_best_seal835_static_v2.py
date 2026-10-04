"""Preserve a timed-out read and persist the next static seal verification."""
import ast
from pathlib import Path
import json

private=Path('C:/Users/gb/.codex_tmp')
original=private/'seal_global_task_role_fixed_best_after_complete.py'
target=private/'seal_global_task_role_fixed_best_after_complete_v2.py'
assert not target.exists()
source=original.read_text(encoding='utf-8')
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
code=ast.literal_eval(node.value)
remote_tree=ast.parse(code)
last=remote_tree.body[-1]
assert isinstance(last,ast.Expr) and isinstance(last.value,ast.Call) and last.value.func.id=='print'
record=ast.get_source_segment(code,last.value.args[0].args[0])
prefix=code[:last.col_offset+sum(len(l) for l in code.splitlines(keepends=True)[:last.lineno-1])]
code=prefix+"result="+record+"\njournal=root/'logs/global_task_role_fixed_best_seal835_20261005'\nassert not journal.exists()\njournal.mkdir()\n(journal/'RESULT.json').write_text(json.dumps(result,indent=2)+'\\n')\nprint(json.dumps(result))\n"
compile(code,'persist_static_seal.py','exec')
old=ast.get_source_segment(source,node.value)
source=source.replace(old,repr(code),1)
source=source.replace('assert not packet.exists() and not destination.exists()',"assert (packet/'LOCAL_TIMEOUT.json').exists() and not (packet/'stdout.json').exists() and not destination.exists()",1)
source=source.replace('packet.mkdir()\n','',1)
source=source.replace("remote_seal_inputs.py","remote_seal_inputs_v2.py")
source=source.replace('stdout.channel.settimeout(60)','stdout.channel.settimeout(300)',1)
compile(source,str(target),'exec')
target.write_text(source,encoding='utf-8')
packet=private/'independent_evidence_draft/global_task_role_fixed_best_seal'
(packet/'OBSERVER_SYNTAX_FAILURE.json').write_text(json.dumps(dict(status='PRE_OBSERVATION_SYNTAX_ERROR',error='Outer string expanded null bytes; corrected observer executed once successfully.',failed_helper='observe_fixed_best_seal_timeout835_failed.py'))+'\n',encoding='utf-8')
(packet/'STATIC_V2_BOUNDARY.json').write_text(json.dumps(dict(status='PREPARED_STATIC_SEAL_V2',reason='Original SSH read timed out; original process absent at01:20:33. No model ever started. New verification persists its completed result remotely and uses300s read deadline. Original failure retained; no tolerance/model/source330 change.'))+'\n',encoding='utf-8')
print('STATIC_SEAL_V2_PREPARED')

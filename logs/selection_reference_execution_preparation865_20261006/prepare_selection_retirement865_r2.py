from pathlib import Path
import ast,json
base=Path('C:/Users/gb/.codex_tmp')
failed=base/'independent_evidence_draft/selection_storage865_retirement'
assert failed.exists() and not (failed/'FAILURE.json').exists()
(failed/'SOURCE_LOCAL.py').write_bytes((base/'retire_selection_storage865.py').read_bytes())
(failed/'FAILURE.json').write_text(json.dumps(dict(stage='local_embedded_code_compile',ssh_connected=False,deletions=0,error='Non-raw embedded source turned literal newline escapes into quoted newlines.'),indent=2)+'\n')
source=(base/'retire_selection_storage865.py').read_text()
source=source.replace("packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865_retirement')", "packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865_retirement_r2')")
source=source.replace("code='payload='+repr(qualified)+'\\n'+'''", "code='payload='+repr(qualified)+'\\n'+r'''")
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
tail=ast.literal_eval(node.value.right)
compile('payload={}\n'+tail,'actual_embedded_retirement865','exec')
destination=base/'retire_selection_storage865_r2.py';assert not destination.exists()
destination.write_text(source)
print('Actual embedded code compiled before SSH; original pre-SSH failure preserved')

"""Use the actual terminal schema: active_command is absent after child exit."""
from pathlib import Path
import ast,json
private=Path('C:/Users/gb/.codex_tmp');packet=private/'independent_evidence_draft/deployment_metric_original_terminal842'
assert json.loads((packet/'EXIT.json').read_bytes())['exit_code']==1
source=(private/'collect_deployment_metric_terminal842.py').read_text(encoding='utf-8')
assert "state['active_command'] is None" in source
source=source.replace("state['active_command'] is None","'active_command' not in state")
source=source.replace("packet=base/'deployment_metric_original_terminal842'","packet=base/'deployment_metric_original_terminal842b'")
ast.parse(source);target=private/'collect_deployment_metric_terminal842b.py';assert not target.exists();target.write_text(source,encoding='utf-8')
print('Prepared read-only terminal intake for original absent-active-command schema.')

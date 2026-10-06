from pathlib import Path
import json

base=Path('C:/Users/gb/.codex_tmp')
source=(base/'launch_additive_geometry_fixed_arrays864.py').read_text(encoding='utf-8')
packet=base/'independent_evidence_draft/additive_geometry_fixed_arrays864'
assert not (packet/'PRE_SSH_FAILURE.json').exists()
(packet/'PRE_SSH_FAILURE.json').write_text(json.dumps(dict(exit_code=1,stage='local_supervisor_ast_compile',
    ssh_connected=False,remote_created=False,diagnostic_launched=False,
    error='Embedded newline literal in supervisor string was decoded before AST parse.'),indent=2)+'\n')
source=source.replace("supervisor='''", "supervisor=r'''")
exec(compile(source,'launch_additive_geometry_fixed_arrays864_r2','exec'))

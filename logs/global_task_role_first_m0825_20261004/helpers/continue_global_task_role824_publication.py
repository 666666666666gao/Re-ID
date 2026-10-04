"""Complete already-pushed824 deployment; do not rerun the failed publisher."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

private=Path('C:/Users/gb/.codex_tmp')
proof=private/'foundation_recipe_v1_20261002'
packet=private/'independent_evidence_draft/global_task_role_publication824_failure'
assert not packet.exists()
source=(private/'publish_global_task_role824.py').read_text(encoding='utf-8')
assert "destination='/tmp/trifusion_target823_20261004.bundle'" in source
assert (proof/'commit824.json').exists() and not (proof/'four_copy824_2025_pending.json').exists()
packet.mkdir()
(packet/'FAILED_PUBLISHER.py').write_text(source,encoding='utf-8')
(packet/'EXIT.json').write_text(json.dumps(dict(exit_code=1,at=datetime.now().astimezone().isoformat(),
 source_sha256=hashlib.sha256(source.encode()).hexdigest(),observed='Native exec session22778 exit1 at remote assertion Path(/tmp/trifusion_target823_20261004.bundle) must not exist;GitHub commit/push complete,2026 merge not yet executed. Original bundle823 preserved. No training or restart.'),indent=2)+'\n',encoding='utf-8')
continuation='''"""Continue only the remaining sync after exact original path collision."""
from pathlib import Path
import hashlib,json,shlex,subprocess,urllib.request
import paramiko
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
publication=json.loads((proof/'publication824_local.json').read_bytes())
previous=json.loads((proof/'four_copy823_2025_pending.json').read_bytes())
committed=json.loads((proof/'commit824.json').read_bytes())
head=committed['head']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==head
assert not (proof/'four_copy824_2025_pending.json').exists()
'''
continuation+=source[source.index('files=set()'):]
continuation=continuation.replace("destination='/tmp/trifusion_target823_20261004.bundle'", "destination='/tmp/trifusion_target824_20261004.bundle'")
ast.parse(continuation)
target=private/'sync_pushed_global_task_role824.py'
assert not target.exists()
target.write_text(continuation,encoding='utf-8')
print(json.dumps(dict(status='PUSHED824_SYNC_CONTINUATION_PREPARED',head=json.loads((proof/'commit824.json').read_bytes())['head'])))

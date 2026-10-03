from pathlib import Path
from datetime import datetime
import json,subprocess,ast

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
publication=json.loads((proof/'publication767_local.json').read_bytes())
actual=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=repo).decode().rstrip('\0').split('\0')
assert set(actual)==set(publication['files'])
failure=proof/'publication767_git_format_failure.json';assert not failure.exists()
failure.write_bytes((json.dumps({'status':'ORIGINAL_PUBLISHER_FAILED_BEFORE_COMMIT',
 'recorded_at':datetime.now().astimezone().isoformat(),'exit_code':2,
 'evidence_file':'logs/metric_scale_storage_finish767_20261003/launcher_revision/SOURCE_REVIEW.md',
 'line':58,'finding':'original reviewer blank line contains one trailing space; preserve primary bytes'},indent=2)+'\n').encode())
name='logs/metric_scale_storage_finish767_20261003/.gitattributes';path=repo/name;assert not path.exists()
path.write_bytes(b'launcher_revision/SOURCE_REVIEW.md -whitespace\n')
publication['files'].append(name);publication['files'].sort()
(proof/'publication767_local.json').write_bytes((json.dumps(publication,indent=2)+'\n').encode())
subprocess.run(['git','-c','core.autocrlf=false','add','-f','--',name],cwd=repo,check=True)
source=Path('C:/Users/gb/.codex_tmp/publish_metric_finish_launch767.py').read_text(encoding='utf-8')
source=source.replace("assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).strip()\n",'')
source=source.replace("subprocess.run(['git','-c','core.autocrlf=false','add','-f','--',*changed],cwd=repo,check=True)\n",'')
source=source.replace("subprocess.run(['git','-c','core.autocrlf=false','add','--renormalize','--',*changed],cwd=repo,check=True)\n",'')
out=Path('C:/Users/gb/.codex_tmp/resume_metric_finish_publication767.py');assert not out.exists()
out.write_bytes(source.encode());ast.parse(source)
print('Prepared publication resume from existing exact owned index; original evidence bytes preserved.')

from pathlib import Path
from datetime import datetime
import ast,hashlib,json,subprocess,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
old=json.loads((root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(old)==339
assert all(sha(root/n)==d for n,d in old.items())
controls=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
assert len(controls['artifact_sha256'])==187 and len(controls['rows'])==9
assert all(sha(Path(n))==d for n,d in controls['artifact_sha256'].items())
for n,d in {'modeling/trifusion/semantic_capacity_evidence.py': 'e80e825bf080e7fd2517df7a9f918707d6de37eedcbe3c0af26ffa0aff4e126e', 'tools/run_semantic_capacity.py': '828e9f279a1934d742d7247c6c136e6b2dfacc450ee5a0b5472696b9bcb47bf5', 'tools/check_semantic_capacity.py': '0c18f5cce1ee3f3c8affffd28e349c2673cebf3ff73038223e886ed1d94663eb', 'tools/queue_semantic_capacity.py': 'd4c2551bc1fced4ea0f3310a820c88f33430b98c0ba330fce7c8387dced6fe21', 'tools/report_semantic_capacity.py': '863b3761c83ca8f0a23f395e8e0431383b8d356d3d2f1ffffb8746a018ed17e1'}.items():
 assert sha(root/n)==d;ast.parse((root/n).read_text())
directory=root/'logs/semantic_capacity_validation_20261005_856'
assert not directory.exists();directory.mkdir()
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/check_semantic_capacity.py'),'--output',str(directory/'CPU_COMPONENT.json')]
with (directory/'component.log').open('x') as log:
 result=subprocess.run(command,cwd=root,env=dict(__import__('os').environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
value=dict(status='CPU_COMPONENT_PASS' if result.returncode==0 else 'CPU_COMPONENT_FAILED',at=datetime.now().astimezone().isoformat(),
 old_sources=339,controls=187,new_source_sha256={'modeling/trifusion/semantic_capacity_evidence.py': 'e80e825bf080e7fd2517df7a9f918707d6de37eedcbe3c0af26ffa0aff4e126e', 'tools/run_semantic_capacity.py': '828e9f279a1934d742d7247c6c136e6b2dfacc450ee5a0b5472696b9bcb47bf5', 'tools/check_semantic_capacity.py': '0c18f5cce1ee3f3c8affffd28e349c2673cebf3ff73038223e886ed1d94663eb', 'tools/queue_semantic_capacity.py': 'd4c2551bc1fced4ea0f3310a820c88f33430b98c0ba330fce7c8387dced6fe21', 'tools/report_semantic_capacity.py': '863b3761c83ca8f0a23f395e8e0431383b8d356d3d2f1ffffb8746a018ed17e1'},component_exit_code=result.returncode,disk_free_bytes=shutil.disk_usage(root).free)
(directory/'VALIDATION.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
print((directory/'component.log').read_text())
assert result.returncode==0


from datetime import datetime
from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
output=root/'results/deployment_metric_storage_five_20261005_848'
launch=root/'logs/deployment_metric_storage_five_report_launch_20261005_850'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==0
report=json.loads((output/'SUMMARY.json').read_text())
expected=[('RGBNT201','semantic'),('RGBNT201','native'),('MSVR310','semantic'),('RGBNT100','semantic'),('RGBNT100','native')]
assert report['status']=='PARTIAL_FIVE_FORMAL_ENDPOINTS_AND_ONE_ORIGINAL_M0_FAILURE'
assert report['report_work_status']=='COMPLETE'
assert report['expected_formal_endpoints']==6 and report['accepted_formal_endpoints']==5
assert report['formal_epochs']==250 and report['formal_steps']==12262
assert [(r['dataset'],r['variant']) for r in report['rows']]==expected
assert len(report['pairs'])==12 and len(report['unavailable_pairs'])==3
assert all(p['status']=='UNAVAILABLE_ORIGINAL_M0_FAILED' and p['dataset']=='MSVR310' for p in report['unavailable_pairs'])
source=root/'refine-logs/deployment_metric_role_v1/REPORT_STORAGE_FIVE_848.py'
assert sha(source)==report['producer_sha256']=='545c115b3402337285c88e8870e961f7fba36eeae7055fba4c92d00e169755b2'
scope=root/'refine-logs/deployment_metric_role_v1/SOURCE_SCOPE.json'
assert sha(scope)=='76987d2ca7d6dc7a9aeac34000d37ebe5e7d1d8ead6d52b1832f0b1557523a3d'
sources=json.loads(scope.read_text())['source_sha256']
assert len(sources)==339 and all(sha(root/name)==digest for name,digest in sources.items())
seal=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
controls=json.loads(seal.read_text())
assert len(controls['artifact_sha256'])==187 and all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())
origin=root/'logs/deployment_metric_role_v1_20261005_837/campaign.json'
pending=root/'logs/deployment_metric_role_pending100_20261005_842/campaign.json'
assert sha(origin)=='88e2575a7d3693afc0e9e02d0ed3352e61e0eb02447263b8a5c4ff1612db62f9'
assert sha(pending)=='4d5b9e00e2b3739a59c0ae411c89b28b4955701b5bb16780eaab84e54130223d'
files=[output/'SUMMARY.json',output/'REPORT.md',launch/'LAUNCH.json',launch/'EXIT.json',launch/'report.log',launch/'supervisor.log']
artifact_sha256={str(p.relative_to(root)):sha(p) for p in files}
result=dict(status='STORAGE_FIVE_SAVED_ARRAY_CPU_REPORT_COLLECTED',collected_at=datetime.now().astimezone().isoformat(),
 files={str(p.relative_to(root)):p.read_text(encoding='utf-8') for p in files},artifact_sha256=artifact_sha256,
 formal_endpoints=5,formal_epochs=250,formal_steps=12262,available_pairs=12,unavailable_pairs=3,
 matched_role_progress_count=report['primary_matched_role_progress_count'],independent_global_progress_count=report['independent_global_progress_count'],
 scientific_source_count=339,control_artifact_count=187,boundary='Only original report output and launch receipts collected. Five accepted endpoints and one original M0 failure; no report/model replay, six-endpoint completion, SOTA or training-seed significance.')
print(json.dumps(result))

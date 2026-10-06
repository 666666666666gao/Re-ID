from pathlib import Path
from datetime import datetime
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/signal_selection_reference_v1_20261006_868'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=json.loads((campaign/'campaign.json').read_text());assert state['status']=='RUNNING' and state['report_invocations']==0
assert sum(r['phase']=='full' and r['status']=='COMPLETE' for r in state['jobs'])==5
parent=Path('/proc/262930');assert parent.exists() and int((parent/'stat').read_text().split()[21])==52446692
steps=[s for r in state['jobs'] for s in r.get('steps',[]) if s['status']=='RUNNING']
assert len(steps)==1 and steps[0]['mode']=='train'
step=steps[0];assert step['command'][step['command'].index('--dataset')+1]=='MSVR310' and step['command'][step['command'].index('--selection')+1]=='all_patch'
process=Path('/proc')/str(step['pid']);assert process.exists() and int((process/'stat').read_text().split()[21])==step['start_ticks']
scope=root/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json';scope_sha=sha(scope)
sources=json.loads(scope.read_text())['source_sha256'];assert len(sources)==374 and all(sha(root/n)==v for n,v in sources.items())
old=root/'logs/selection_counter_record_normalization_20261006_868/MSVR310_optimizer_batch_metadata.jsonl'
provenance=old.parent/'PROVENANCE.json';provenance_sha=sha(provenance)
assert sha(old)=='48ed83d5118036bd0ad322389a249be5efb028daed074c68c629e72270b37472'
raw=root/'trained-model/signal_selection_reference_v1_20261006_867_full_global_only_MSVR310/training_batch_metadata.jsonl'
assert sha(raw)=='e45690800f383d3b266d7949f5bd19a0b8fc3f55d99baa1bf9977e61ea988241'
masked=root/'trained-model/signal_selection_reference_v1_20261006_868_full_masked_MSVR310/training_batch_metadata.jsonl'
old_rows=list(map(json.loads,old.read_text().splitlines()));new_rows=list(map(json.loads,masked.read_text().splitlines()))
assert old_rows==new_rows and len(old_rows)==706
fields=('global_step','labels','cameras','view_ids','rgb_basenames')
assert all(set(r)==set(fields) for r in old_rows)
canonical=''.join(json.dumps(dict(global_step=r['global_step'],labels=r['labels'],cameras=r['cameras'],view_ids=r['view_ids'],rgb_basenames=r['rgb_basenames']))+'\n' for r in old_rows).encode()
assert canonical==masked.read_bytes()
assert hashlib.sha256(canonical).hexdigest()==sha(masked)=='7b16ed8e2ad4cd50c50decb49d952851729fdabf65fad6257e9f3803279807b4'
paths=campaign/'batch_metadata_paths.json';old_paths_bytes=paths.read_bytes();old_paths=json.loads(old_paths_bytes)
assert len(old_paths)==9 and old_paths['MSVR310:global_only']==str(old)
assert json.loads((campaign/'manifest.json').read_text())['batch_metadata_paths']==old_paths
out=root/'logs/selection_metadata_key_order_revision_20261006_868';assert not out.exists();out.mkdir()
target=out/'MSVR310_optimizer_batch_metadata_producer_order.jsonl'
updated=dict(old_paths);updated['MSVR310:global_only']=str(target)
qualification=dict(status='QUALIFIED_SERIALIZATION_ONLY_BEFORE_PATH_REVISION',at=datetime.now().astimezone().isoformat(),old_batch_paths_sha256=hashlib.sha256(old_paths_bytes).hexdigest(),
    old_metadata_path=str(old),old_metadata_sha256=sha(old),new_metadata_path=str(target),new_metadata_sha256=hashlib.sha256(canonical).hexdigest(),
    ordered_fields_equal=True,rows=706,old_and_new_key_order=dict(old=list(old_rows[0]),new=list(new_rows[0])),
    old_original_mixed_raw_sha256=sha(raw),old_normalization_provenance_sha256=provenance_sha,source_scope_sha256=scope_sha,
    old_manifest_sha256=sha(campaign/'manifest.json'),old_report_path_map=old_paths,new_report_path_map=updated,
    same_original_controller_pid=262930,active_allpatch_step=step,
    boundary='Before mutation registration: only change one explicit finalCPUreport path to same706fieldvalues serializedin actualproducer keyorder. Originalcanonical48ed/originalraw2056/PROVENANCE/manifest/source374/traininputs/NN/optimizer/metrics/threshold remain untouched. Existing reportbyte guard unchanged. Originalcollector bytefailure preserved. No newNN/eval/report orscientificclaim.')
(out/'QUALIFIED.json').write_text(json.dumps(qualification,indent=2)+'\n')
(out/'ORIGINAL_BATCH_METADATA_PATHS.json').write_bytes(old_paths_bytes)
target.write_bytes(canonical)
assert json.loads((out/'QUALIFIED.json').read_text())['new_metadata_sha256']==sha(target)
assert list(map(json.loads,target.read_text().splitlines()))==old_rows
paths.write_text(json.dumps(updated,indent=2)+'\n')
assert json.loads(paths.read_text())==updated
assert sha(scope)==scope_sha and all(sha(root/n)==v for n,v in sources.items())
assert sha(old)==qualification['old_metadata_sha256'] and sha(raw)==qualification['old_original_mixed_raw_sha256'] and sha(provenance)==provenance_sha
result=dict(status='COMPLETE_SERIALIZATION_ONLY_CPU_REPORT_PATH_REVISION',at=datetime.now().astimezone().isoformat(),qualified_sha256=sha(out/'QUALIFIED.json'),updated_report_path_map_sha256=sha(paths),
    new_metadata_sha256=sha(target),rows=706,old_canonical_unchanged=True,old_original_raw_unchanged=True,all374sources_unchanged=True,
    actual_allpatch_train_step=step,out_dir=str(out),original_manifest_sha256=qualification['old_manifest_sha256'])
(out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
files=list(out.iterdir())+[paths]
print(json.dumps(dict(result=result,files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})))

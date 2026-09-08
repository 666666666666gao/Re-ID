"""Verify the final audit delivery and create an actual-byte artifact manifest."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
def hashed(path):
    h=hashlib.sha256()
    size=0
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):
            h.update(block)
            size+=len(block)
    return dict(bytes=size,sha256=h.hexdigest())

request_path=OUT/'trace/001-role-relation-coverage.request.json'
request=json.loads(request_path.read_bytes())
request['dispatch_prompt']=('Perform the fresh read-only audit specified in C:/Users/gb/.codex_tmp/msvr_role_relation_coverage_audit_request_20260908.md. '
    'Read that request and listed primary files directly, independently verify completion and numerical evidence yourself, and follow the cited experiment-audit/local-policy instructions. '
    'Write all outputs only to C:/Users/gb/.codex_tmp/msvr_role_relation_coverage_independent_audit_20260908. '
    'No scientific edits, remote writes, model runs, package installs, official-image reads or training. '
    'Preserve the full substantive response in reviewer_full_response.md plus audit.json and your independent scripts/output hashes. '
    'Requested routing is gpt-6-astra/max, fresh context; record same-family/provisional and do not invent a backend attestation or agent UUID.')
request['prompt_file_contents']=(OUT/'reviewer_request.md').read_text(encoding='utf-8')
request['native_dispatch_metadata_observed_by_reviewer']='canonical task path and received dispatch message only; no backend attestation or agent UUID exposed'
request_path.write_text(json.dumps(request,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
audit=json.loads((OUT/'audit.json').read_bytes())
det=json.loads((OUT/'deterministic_verification.json').read_bytes())
response=(OUT/'reviewer_full_response.md').read_bytes()
assert response==(OUT/'trace/001-role-relation-coverage.response.md').read_bytes()
assert hashlib.sha256(response).hexdigest()==audit['reviewer_full_response_sha256']
assert audit['verdict']=='WARN' and audit['review_independence']=='same-family' and audit['acceptance_status']=='provisional'
assert audit['agent_uuid'] is None and audit['backend_attestation'] is None
assert det['status']=='PASS_DETERMINISTIC_FULL_FIELD_REPLAY' and det['rows']==37152 and det['condition_results']==54
for tag in ['inspect_remote_inputs','independent_replay','inspect_code_provenance']:
    execution=json.loads((OUT/(tag+'.execution.json')).read_bytes())
    assert execution['exit_code']==0
    assert hashed(OUT/(tag+'.py'))['sha256']==execution['remote_script_sha256']
    assert hashed(OUT/(tag+'.stdout.jsonl'))==execution['stdout']
    assert hashed(OUT/(tag+'.stderr.txt'))==execution['stderr'] and execution['stderr']['bytes']==0
for name,expected in audit['key_output_hashes'].items():
    assert hashed(OUT/name)==expected
local_inputs=json.loads((OUT/'local_input_hashes.json').read_bytes())
local_now=[]
for entry in local_inputs:
    expected={k:entry[k] for k in ['bytes','sha256']}
    assert hashed(OUT/entry['snapshot'])==expected
    actual=hashed(Path(entry['path']))
    local_now.append(dict(path=entry['path'],unchanged=actual==expected,**actual))
assert all(row['unchanged'] for row in local_now)
rows=sum(1 for _ in (OUT/'independently_recreated_query_rows.jsonl').open(encoding='utf-8'))
witnesses=sum(1 for _ in (OUT/'independent_extremum_witnesses.jsonl').open(encoding='utf-8'))
assert rows==37152 and witnesses==29376
validation=dict(status='PASS_FINAL_AUDIT_ARTIFACT_VALIDATION',checked_at=datetime.now(timezone.utc).isoformat(),
    query_rows=rows,extremum_witness_rows=witnesses,full_response_trace_byte_equal=True,all_three_remote_scripts_exit_zero=True,
    all_remote_script_and_stdout_hashes_match=True,local_audited_inputs_unchanged=local_now,
    report=hashed(OUT/'reviewer_full_response.md'),audit=hashed(OUT/'audit.json'))
(OUT/'final_validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
excluded={'artifact_hashes.json','artifact_hashes.json.sha256'}
manifest={p.relative_to(OUT).as_posix():hashed(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in excluded}
(OUT/'artifact_hashes.json').write_text(json.dumps(dict(schema='sha256-audit-artifact-manifest-v1',
    root=str(OUT),excluded_self_referential_files=sorted(excluded),files=manifest),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
mh=hashed(OUT/'artifact_hashes.json')
(OUT/'artifact_hashes.json.sha256').write_text(mh['sha256']+'  artifact_hashes.json\n',encoding='ascii')
print(json.dumps(dict(status=validation['status'],files=len(manifest),total_bytes=sum(r['bytes'] for r in manifest.values()),
    report=validation['report'],audit=validation['audit'],manifest=mh,deterministic=hashed(OUT/'deterministic_verification.json'),
    recreated_rows=hashed(OUT/'independently_recreated_query_rows.jsonl')),indent=2))

"""Finalize receipt, UTF-8 file manifest and manifest digest without altering evidence."""
from pathlib import Path
import hashlib,json,datetime,ast
root=Path(__file__).parent
def proof(name):
    data=(root/name).read_bytes()
    return dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
report=json.loads((root/'EXPERIMENT_AUDIT.json').read_bytes())
assert report['verdict']=='WARN' and report['closure_status']=='CLOSED_WITH_LIMITS'
assert report['audited_input_hashes']['m0_cpu_receipt']=='910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1'
for p in root.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
artifacts=['EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','AUDIT_DETAILS.md','final_response.md',
    '001-request.md','independent_recompute_02.py','independent_recompute_02.json','remote_intake_02.json',
    'text_copy_provenance.json','verify_cpu_receipt_bytes.json','claim_correction_check.json',
    'audit_failures.json','audit_tool_trace.json','recovered_failure_tool_trace.json']
receipt=dict(status='COMPLETE_FRESH_M0_AUDIT_ARCHIVE',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    reviewer_agent='/root/audit_supported_gradient_balance_m0_20260921',
    requested_route=dict(model='gpt-6-astra',reasoning_effort='max',fork_turns='none'),
    review_independence='same-family',acceptance_status='provisional',backend_attestation='unavailable',
    verdict='WARN',closure_status='CLOSED_WITH_LIMITS',deterministic_checks_status='pass',
    engineering_status='PASS_ENGINEERING_ONLY',remaining_engineering_blockers=[],remaining_audit_blockers=[],
    execution_commit=report['execution_commit'],input_hashes=report['audited_input_hashes'],
    primary_artifacts=[proof(name) for name in artifacts],
    remote_requests=[p.name for p in sorted(root.glob('*.request.json'))],
    remote_terminal_status='PASS_INDEPENDENT_COMPLETE_M0_SAVED_EVIDENCE',
    independent_cpu_seconds=report['reviewer_cpu']['seconds'],
    raw_original_cpu_receipt=proof('snapshots/intake/m0_cpu.json'),
    rendered_cpu_receipt=proof('remote_text/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/m0_cpu.json'),
    text_copy_provenance='text_copy_provenance.json',failure_ledger='audit_failures.json',
    full_manifest='artifact_manifest.json',manifest_digest_file='artifact_manifest.sha256',
    manifest_self_reference_policy='Manifest inventories every retained file except itself and its separate SHA-256 file. Receipt is included in the manifest; no circular hash claim.',
    boundary=report['no_new_work'])
(root/'audit_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
excluded={'artifact_manifest.json','artifact_manifest.sha256'}
entries=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.relative_to(root).as_posix() in excluded:continue
    data=p.read_bytes();data.decode('utf-8-sig')
    entries.append(dict(relative_path=p.relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),encoding='utf-8',line_count=len(data.splitlines())))
manifest=dict(status='COMPLETE_UTF8_AUDIT_ARTIFACT_MANIFEST',root=str(root),
    generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    file_count=len(entries),total_bytes=sum(x['bytes'] for x in entries),
    excluded_self_reference_files=sorted(excluded),files=entries)
(root/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest_hash=proof('artifact_manifest.json')['sha256']
(root/'artifact_manifest.sha256').write_text(manifest_hash+'  artifact_manifest.json\n',encoding='utf-8')
for item in entries:
    p=root/item['relative_path'];assert p.resolve().is_relative_to(root.resolve())
    actual=proof(item['relative_path']);assert actual['bytes']==item['bytes'] and actual['sha256']==item['sha256']
assert all((root/name).exists() for name in artifacts)
print(json.dumps(dict(status='PASS_ARCHIVE_COMPLETE_AND_REHASHED',files=len(entries),bytes=manifest['total_bytes'],
    manifest_sha256=manifest_hash,report_markdown=proof('EXPERIMENT_AUDIT.md'),
    report_json=proof('EXPERIMENT_AUDIT.json'),final_response=proof('final_response.md'),receipt=proof('audit_receipt.json')),indent=2))

from pathlib import Path
import json,hashlib
root=Path(__file__).parent
def sha(b):return hashlib.sha256(b).hexdigest()
raw=json.loads((root/'remote_intake_02.json').read_bytes())
remote=json.loads((root/'verify_cpu_receipt_bytes.json').read_bytes())
path=remote['path']
orig=(root/'snapshots/intake/m0_cpu.json').read_bytes()
copy=(root/'remote_text'/path.lstrip('/')).read_bytes()
embedded=raw['text_inputs'][path].encode('utf-8')
inventory=json.loads((root/'snapshots/intake/inventory.json').read_bytes())
entry=next(x for x in inventory['files'] if x['path']=='m0_cpu.json')
assert len(orig)==remote['bytes']==entry['bytes']==16016
assert sha(orig)==remote['sha256']==entry['sha256']==sha(embedded)
assert orig==embedded
assert copy==orig.replace(b'\n',b'\r\n')
assert json.loads(copy)==json.loads(orig)
assert (orig.count(b'\n'),orig.count(b'\r'),len(copy)-len(orig))==(343,0,343)
copies=[]
for name,text in raw['text_inputs'].items():
    p=root/'remote_text'/name.lstrip('/')
    value=text.encode('utf-8');saved=p.read_bytes()
    assert saved==value.replace(b'\n',b'\r\n')
    copies.append(dict(remote_path=name,rendered_path=p.relative_to(root).as_posix(),
        embedded_text_bytes=len(value),embedded_text_sha256=sha(value),
        rendered_bytes=len(saved),rendered_sha256=sha(saved),
        added_carriage_returns=len(saved)-len(value)))
archive=root/'draft_failures';archive.mkdir(exist_ok=True)
drafts=[]
for name in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','AUDIT_DETAILS.md'):
    data=(root/name).read_bytes();dest=archive/(name+'.before_raw_receipt_hash_correction')
    assert not dest.exists();dest.write_bytes(data)
    drafts.append(dict(path=dest.relative_to(root).as_posix(),bytes=len(data),sha256=sha(data)))
result=dict(status='PASS_RAW_RECEIPT_AND_TEXT_RENDERING_DISTINCTION',
    raw_remote_path=path,raw_receipt_bytes=len(orig),raw_receipt_sha256=sha(orig),
    byte_exact_local_receipt='snapshots/intake/m0_cpu.json',
    local_rendered_copy='remote_text/'+path.lstrip('/'),rendered_bytes=len(copy),rendered_sha256=sha(copy),
    lf_count=343,added_carriage_returns=343,parsed_json_equal=True,
    raw_bytes_equal_embedded_utf8_text=True,raw_bytes_match_live_remote_hash=True,
    explanation='process_remote_intake.py:8 used Windows Path.write_text without newline control. The text payloads were rendered with LF -> CRLF. These are text copies with their own hashes, not byte-exact remote files. Registered remote/source hashes were computed on the remote files, independently of those renderings.',
    all_rendered_copies_checked=len(copies),rendered_copy_mapping=copies,retained_wrong_drafts=drafts)
(root/'text_copy_provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
fail=json.loads((root/'audit_failures.json').read_bytes())
for row in fail['attempts']:
    if row['id'] in ('AF3','AF4'):
        row['recovered_original_trace']='recovered_failure_tool_trace.json'
        row['retention_resolution']='Exact original request/output event pair subsequently recovered from this reviewer session transcript; the later AF3 reproduction remains labeled separately.'
    if row['id']=='AF5':row['recovered_original_trace']='recovered_failure_tool_trace.json contains the missing protocol-builder lookup; other exploratory lookup context remains in the session.'
fail['attempts'].append(dict(id='AF6',state='RESOLVED_AUDITOR_REPORT_HASH_LABEL_ERROR',
    finding='Draft labeled the locally rendered m0_cpu.json SHA as the raw remote receipt SHA.',
    raw_remote_sha256=sha(orig),rendered_copy_sha256=sha(copy),
    correction='Report now uses the independently confirmed raw receipt hash and labels the normalized local text copy separately. Experimental receipts and data were not modified.',
    proof='text_copy_provenance.json',remote_byte_witness='verify_cpu_receipt_bytes.json',retained_wrong_drafts=drafts))
(root/'audit_failures.json').write_text(json.dumps(fail,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('rendered_copy_mapping','retained_wrong_drafts')},ensure_ascii=False,indent=2))

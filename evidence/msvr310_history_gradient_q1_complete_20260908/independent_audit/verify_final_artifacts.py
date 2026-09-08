"""Final report/JSON/reference and output hash verification, no scientific writes."""
import hashlib,json,re
from pathlib import Path
from inspect_inputs import OUT,REPO

def get(p):return json.loads(Path(p).read_bytes())
report=(OUT/'reviewer_full_response.md').read_text(encoding='utf-8')
audit=get(OUT/'audit.json');replay=get(OUT/'remote_replay.stdout.json');manifest=get(OUT/'audit_artifact_manifest.json')
roots={'P':REPO,'I':Path('C:/Users/gb/.codex_tmp/history_gradient_q1_complete_20260908'),'T':Path('C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908'),'A':OUT,'S':OUT/'remote_sources/comparators/Signal-cd1b0a6'}
refs=[]
for root,name,line in re.findall(r'\b([PITAS])/([^`\s;|]+?):(\d+)',report):
 p=roots[root]/name;lines=p.read_text(encoding='utf-8-sig').splitlines();n=int(line)
 assert 1<=n<=len(lines),(p,n,len(lines));assert lines[n-1].strip(),(p,n,'blank')
 refs.append(dict(path=p.as_posix(),line=n))
hashes={v['sha256'] for v in audit['audited_input_details'].values()}
hashes.update([audit['execution_provenance']['actual_signal_diff_sha256']])
for digest in re.findall(r'\b[0-9a-f]{64}\b',report):assert digest in hashes,digest
assert sum(x['legal_distance_tie_queries']>0 for x in replay['retrieval'])==29
assert audit['costs']['control']['vjp_record_forwards']==263040
assert audit['costs']['history_gradient']['vjp_record_forwards']==265472
assert all(audit['costs'][x]['fresh_record_forwards']==292800 for x in ('control','history_gradient'))
assert max(x['peak_reserved_mib'] for x in audit['costs'].values())==7086
for name,info in manifest.items():
 data=(OUT/name).read_bytes();assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256'],name
assert audit['full_response_sha256']==hashlib.sha256((OUT/'reviewer_full_response.md').read_bytes()).hexdigest()
output=dict(status='PASS_FINAL_ARTIFACT_AND_REFERENCE_VERIFICATION',references_checked=len(refs),manifest_files_verified=len(manifest),report_sha256=audit['full_response_sha256'],audit_json_sha256=hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest(),all_full_hash_literals_bound=True,all_full_file_line_references_exist_and_nonblank=True,report_json_agree=True)
(OUT/'final_verification.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output,indent=2))

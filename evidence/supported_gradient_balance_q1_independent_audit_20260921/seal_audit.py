from pathlib import Path
import hashlib,json,re
OUT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_bytes())
manifest=js(OUT/'input_manifest.json');remote=js(OUT/'remote_bindings.json');checks=[]
for row in manifest['bindings']:
    if row['actual']!=row['expected']:
        p=OUT/'snapshots/repo'/row['path'];normalized=p.read_bytes().replace(b'\r\n',b'\n')
        h=hashlib.sha256(normalized).hexdigest();assert h==row['expected']
        checks.append(dict(path=row['path'],local_raw=row['actual'],local_lf=h,remote_expected=row['expected'],difference='CRLF only'))
(OUT/'local_line_ending_provenance.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
audit=js(OUT/'EXPERIMENT_AUDIT.json')
assert audit['verdict']=='WARN' and audit['scientific_status']=='Q1_FAIL' and not audit['next_phase_qualified'] and not audit['remaining_engineering_audit_blockers']
assert audit['review_independence']=='same-family' and audit['acceptance_status']=='provisional'
for check in audit['checks'].values():
    for evidence in check['evidence']:
        match=re.fullmatch(r'(.+?):(\d+)(?:-(\d+))?',evidence)
        if match:
            p=OUT/match[1];lines=p.read_text(encoding='utf-8').splitlines();assert 1<=int(match[2])<=len(lines);assert int(match[3] or match[2])<=len(lines),(evidence,len(lines))
        else:assert (OUT/evidence).is_file(),evidence
response=OUT/'final_response.md';s=response.read_text(encoding='utf-8');s=s.replace('</D:/','<D:/');response.write_text(s,encoding='utf-8')
assert s.rstrip().endswith('</oai-mem-citation>') and s.count('<oai-mem-citation>')==1
paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','artifact_manifest.sha256','audit_receipt.json'))
entries=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
mp=OUT/'artifact_manifest.json';mp.write_text(json.dumps(entries,indent=2)+'\n',encoding='utf-8');(OUT/'artifact_manifest.sha256').write_text(sha(mp)+'  artifact_manifest.json\n',encoding='utf-8')
receipt=dict(status='PASS_AUDIT_ARTIFACT_VALIDATION',artifact_count=len(entries),artifact_manifest_sha256=sha(mp),md_sha256=sha(OUT/'EXPERIMENT_AUDIT.md'),json_sha256=sha(OUT/'EXPERIMENT_AUDIT.json'),final_response_sha256=sha(response),evidence_references_exist=True,large_arrays_checkpoints_images_copied=False)
(OUT/'audit_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))

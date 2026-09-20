from pathlib import Path
import hashlib,json
out=Path(__file__).parent
original=(out/'independent_remote_cpu_verify.py').read_bytes()
source=original.decode('utf-8')
edits={
    'all_identity_eligible=sum(x>0 for x in all_n)':'all_identity_eligible=int(sum(x>0 for x in all_n))',
    'cross_scene_eligible=sum(x>0 for x in cross_n)':'cross_scene_eligible=int(sum(x>0 for x in cross_n))',
    'single_scene_records=sum(x==0 for x in cross_n)':'single_scene_records=int(sum(x==0 for x in cross_n))',
}
for old,new in edits.items():
    assert source.count(old)==1
    source=source.replace(old,new)
compile(source,'independent_remote_cpu_verify_r2.py','exec')
(out/'independent_remote_cpu_verify_r2.py').write_text(source,encoding='utf-8')
payload=(out/'launch_remote_verification.py').read_text()
assert repr(original.decode('utf-8')) in payload
payload=payload.replace(repr(original.decode('utf-8')),repr(source))
payload=payload.replace(hashlib.sha256(original).hexdigest(),hashlib.sha256(source.encode()).hexdigest())
payload=payload.replace("root=Path('/root/trifusion-storage/artifacts/smooth_ap_source_coverage_independent_audit_20260920')","root=Path('/root/trifusion-storage/artifacts/smooth_ap_source_coverage_independent_audit_20260920/attempt02')")
(out/'launch_remote_verification_r2.py').write_text(payload,encoding='utf-8')
(out/'auditor_script_r2_change.json').write_text(json.dumps(dict(reason='Observed NumPy int64 serialization error in protocol summary before array checks; both R1 processes ended.',edits=edits,old_sha256=hashlib.sha256(original).hexdigest(),new_sha256=hashlib.sha256(source.encode()).hexdigest(),science_source_changed=False),indent=2),encoding='utf-8')
observe=(out/'observe_remote_verification.py').read_text().replace("root=Path('/root/trifusion-storage/artifacts/smooth_ap_source_coverage_independent_audit_20260920')","root=Path('/root/trifusion-storage/artifacts/smooth_ap_source_coverage_independent_audit_20260920/attempt02')")
(out/'observe_remote_verification_r2.py').write_text(observe,encoding='utf-8')
print(json.dumps(dict(new_sha256=hashlib.sha256(source.encode()).hexdigest())))

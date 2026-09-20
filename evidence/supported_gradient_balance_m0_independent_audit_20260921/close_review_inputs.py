from pathlib import Path
import json,hashlib,subprocess,sys,datetime
root=Path(__file__).parent
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
name='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
receipt_path=Path('D:/Program Files/UserCache/gb/codex/tmp/supported_balance_m0_claim_qualification_20260921.json')
receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
before=(root/'latest_claims'/name).read_bytes();after=(repo/name).read_bytes()
assert hashlib.sha256(before).hexdigest()==receipt['before_sha256']
assert hashlib.sha256(after).hexdigest()==receipt['after_sha256']
expected=before
for key in ('reload_qualifier','current_overview'):
    a,b=receipt[key+'_before'].encode(),receipt[key+'_after'].encode()
    assert expected.count(a)==1
    expected=expected.replace(a,b)
assert expected==after
dest=root/'corrected_claims'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(after)
(root/'corrected_claims/author_correction_receipt.json').write_bytes(receipt_path.read_bytes())
result=dict(status='PASS_RELOAD_SCOPE_CORRECTION_INDEPENDENTLY_REREAD',
    old_snapshot='latest_claims/'+name,corrected_snapshot='corrected_claims/'+name,
    before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),
    exactly_two_text_replacements=True,reload_claim_line=6754,
    reviewed_correction='Six capacity checkpoints only; two overfit runs have no final checkpoint or strict reload.',
    other_replacement='Opening overview corrects previously sealed experiment context; not a new retrieval claim audited here.')
(root/'claim_correction_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Repeat the untouched failing local checker solely to retain an exact reproducible stderr.
# It stops on the original line 19 before writing r1_r2.diff or any claim snapshot.
command=[sys.executable,'-X','utf8',str(root/'check_local_bindings_claims.py')]
repro=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
assert repro.returncode!=0 and "KeyError: 'local_relative_path'" in repro.stderr
(root/'local_checker_failure_reproduction.stdout.txt').write_text(repro.stdout,encoding='utf-8')
(root/'local_checker_failure_reproduction.stderr.txt').write_text(repro.stderr,encoding='utf-8')
(root/'local_checker_failure_reproduction.request.json').write_text(json.dumps(dict(command=command,
    returncode=repro.returncode,purpose='Explicit later reproduction of original local schema-key failure; original script retained untouched.',
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

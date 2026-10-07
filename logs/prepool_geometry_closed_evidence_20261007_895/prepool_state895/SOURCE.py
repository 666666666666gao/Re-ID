from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert head=='24291e5d523751c654b19d3232e3e6de82727247',head
scope=json.loads((root/'refine-logs/incremental_role_objective_v1/FIXED_FIVE_DIAGNOSIS_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(scope)==424
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in scope.items())
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        words=(p/'cmdline').read_bytes().split(bytes([0]))
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in words)
exit=json.loads((root/'logs/fixed_five_incremental_best_launch_20261007_892/EXIT.json').read_text())
assert exit['exit_code']==0
retired=json.loads((root/'logs/fixed_five_closed_binary_retirement_20261007_894/RETIRED.json').read_text())
print(json.dumps(dict(status='CLOSED_OWN_NN_FROZEN424_UNCHANGED',at=datetime.now().astimezone().isoformat(),head=head,source_count=len(scope),free_bytes=shutil.disk_usage(root).free,original_exit=exit,retirement_receipt_sha256=hashlib.sha256((root/'logs/fixed_five_closed_binary_retirement_20261007_894/RETIRED.json').read_bytes()).hexdigest(),boundary='read-only26; no NN/GPU/temp/power/25/old75seal actions')))

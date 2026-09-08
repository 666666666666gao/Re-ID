import base64
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
PROJECT=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
hashes=[]
for line in (OUT/'inspect_code_provenance.stdout.jsonl').read_text(encoding='utf-8').splitlines():
    entry=json.loads(line)
    raw=base64.b64decode(entry.pop('contents_b64'))
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    assert entry['git_returncode']==0 and entry['sha256']==entry['committed_sha256']
    assert raw==(PROJECT/entry['path']).read_bytes()
    path=OUT/'remote_snapshots/project'/entry['path']
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    hashes.append(dict(**entry,snapshot=str(path.relative_to(OUT))))
(OUT/'supplementary_input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
print(json.dumps(hashes,indent=2))

"""Repair demonstrated CRLF normalization in audit snapshots using equal-hash bytes.
Only audit files are changed. Original normalized snapshots and failed validation
are retained. Remote expected hashes were calculated from original raw bytes.
"""
import hashlib,json,shutil
from pathlib import Path
O=Path(__file__).resolve().parent
proof=json.loads((O/'snapshot_validation.json').read_bytes())
shutil.copyfile(O/'snapshot_validation.json',O/'snapshot_validation_attempt1.json')
lookup={}
for path,item in json.loads((O/'local_input_manifest.json').read_bytes()).items():lookup[item['sha256']]=O/item['snapshot']
repairs=[]
for row in proof:
    if row['byte_exact']:continue
    dest=O/row['snapshot'];raw=lookup[row['source_sha256']].read_bytes()
    assert hashlib.sha256(raw).hexdigest()==row['source_sha256']
    old=dest.read_bytes();assert raw.replace(b'\r\n',b'\n')==old
    preserved=O/'attempt_02_normalized_text_snapshots'/dest.relative_to(O/'snapshots')
    preserved.parent.mkdir(parents=True,exist_ok=True);preserved.write_bytes(old)
    dest.write_bytes(raw)
    repairs.append({'source':row['source'],'original_expected_sha256':row['source_sha256'],'normalized_snapshot_sha256':row['snapshot_sha256'],'exact_bytes_from':str(lookup[row['source_sha256']].relative_to(O)),'remote_source_and_scientific_files_changed':False})
(O/'snapshot_crlf_repairs.json').write_text(json.dumps(repairs,indent=2),encoding='utf-8')
print('restored_exact_snapshots',len(repairs))

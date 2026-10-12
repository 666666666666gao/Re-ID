import hashlib,json,zipfile
from pathlib import Path
work=Path(__file__).resolve().parent
meta=json.loads((work/'SIXTH_RESULTS_912.meta.json').read_text())
archive=work/'SIXTH_RESULTS_912.zip'
assert archive.stat().st_size==meta['bytes']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
target=work/'results912'
assert not target.exists()
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('INTAKE_MANIFEST_912.json'))
 assert len(manifest['files'])==meta['files']
 data={name:z.read(name) for name in manifest['files']}
 assert all(hashlib.sha256(data[n]).hexdigest()==h for n,h in manifest['files'].items())
 assert all((target/n).resolve().is_relative_to(target.resolve()) for n in data)
 target.mkdir()
 for name,value in data.items():
  p=target/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
 (target/'INTAKE_MANIFEST_912.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
summary=json.loads((target/'logs/region_use_remaining_endpoint_20261013_912/RGBNT100_ambiguity/SUMMARY.json').read_text())
print(json.dumps(dict(status='SIXTH_ORIGINAL_RESULT_INTAKE_COMPLETE',files=len(data),metrics=summary['metrics'])))

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from pypdf import PdfReader

ROOT = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/rodi_primary_pdf_20261004')
ROOT.mkdir(exist_ok=True)
URL = 'https://raw.githubusercontent.com/lsh-ahu/RoDI/2f38911c49d42d4ca259d440a851b8d77dddccbe/assets/RoDI.pdf'
data = urlopen(URL, timeout=60).read()
assert len(data) == 5636720
blob_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
assert blob_sha == 'dcfca845758fdb072688c5077679f10ee6f92cc3'
pdf = ROOT / 'RoDI.pdf'
pdf.write_bytes(data)
reader = PdfReader(pdf)
pages = [page.extract_text() for page in reader.pages]
(ROOT / 'pages.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'text.txt').write_text('\n\n'.join(f'--- PDF PAGE {i + 1} ---\n{text}' for i, text in enumerate(pages)), encoding='utf-8')
receipt = {'url': URL, 'commit': '2f38911c49d42d4ca259d440a851b8d77dddccbe', 'bytes': len(data), 'git_blob_sha1': blob_sha, 'sha256': hashlib.sha256(data).hexdigest(), 'pages': len(pages)}
(ROOT / 'DOWNLOAD.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
for i, page in enumerate(pages):
    if any(phrase.lower() in page.lower() for phrase in ('implementation details', 'evaluation protocols', 'table 1', 'dataset')):
        print(f'PAGE {i + 1}: {page[:1200]}')

"""Fetch one already-observed primary URL without retries or model execution."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('url')
parser.add_argument('name')
args = parser.parse_args()
folder = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/strong_reference_primary_20261004/downloads')
folder.mkdir(parents=True, exist_ok=True)
target = folder / args.name
assert not target.exists(), target
request = urllib.request.Request(args.url, headers={'User-Agent': 'TriFusion-primary-source-check'})
with urllib.request.urlopen(request, timeout=45) as response:
    payload = response.read()
    final_url = response.url
    content_type = response.headers.get('Content-Type')
target.write_bytes(payload)
receipt = dict(url=args.url, final_url=final_url, filename=args.name,
               at=datetime.now().astimezone().isoformat(), bytes=len(payload),
               sha256=hashlib.sha256(payload).hexdigest(), content_type=content_type,
               boundary='Public original source download only; no SSH/GPU/model/code execution.')
(folder / (args.name + '.receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))

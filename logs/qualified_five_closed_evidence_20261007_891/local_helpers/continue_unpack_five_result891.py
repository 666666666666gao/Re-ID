from pathlib import Path
from datetime import datetime
import hashlib
import json

b = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = b / 'qualified_five_full_complete889'
r = json.loads((packet / 'REMOTE.json').read_bytes())
mapping = json.loads((packet / 'FILE_MAP.json').read_bytes())
assert all(hashlib.sha256(Path(v['local_file']).read_bytes()).hexdigest() == v['sha256'] for v in mapping.values())
source = Path('C:/Users/gb/.codex_tmp/unpack_five_result891.py')
(packet / 'UNPACK_FIRST_FAILURE.json').write_bytes((json.dumps({
    'status': 'LOCAL_TEXT_UNPACK_FIRST_INVOCATION_FAILED_AFTER_VERIFIED_EXTRACTION',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'error': 'StopIteration at summary_path selector: collected paths are root-relative results/... without leading slash',
    'remote_collection_exit': 0, 'remote_training_and_report_exit': 0,
    'continuation_scope': 'Read existing exact FILE_MAP and root-relative results SUMMARY; no recollection, model execution or source changes'
}, indent=2) + '\n').encode('utf-8'))
tail = 'summary_path = ' + source.read_text(encoding='utf-8').split('summary_path = ', 1)[1]
tail = tail.replace("and '/results/' in n", "and n.startswith('results/')")
exec(compile(tail, 'continue_unpack_five_result891_tail', 'exec'))

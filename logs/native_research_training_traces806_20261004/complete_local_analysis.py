from datetime import datetime
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path('C:/Users/gb/.codex_tmp')
original = root / 'analyze_closed_native_training_traces.py'
receipt = root / 'independent_evidence_draft/native_research_trace_analysis_attempt1.json'
assert not receipt.exists()
receipt.write_text(json.dumps({
    'recorded_at': datetime.now().astimezone().isoformat(), 'exit_code': 1,
    'script_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
    'failed_assertion': "training['status'] == official['status'] == 'COMPLETE'",
    'actual_training_status': 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE',
    'actual_official_status': 'COMPLETE',
    'boundary': 'Local post-selection text-analysis assertion used the wrong training receipt status. Stopped before output creation. No training, inference, source model or formal result changed.'
}, indent=2) + '\n', encoding='utf-8')
source = original.read_text(encoding='utf-8')
old = "assert training['status'] == official['status'] == 'COMPLETE'"
new = "assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status'] == 'COMPLETE'"
assert source.count(old) == 1
source = source.replace(old, new)
target = root / 'analyze_closed_native_training_traces_v2.py'
assert not target.exists()
compile(source, str(target), 'exec')
target.write_text(source, encoding='utf-8')
raise SystemExit(subprocess.run([sys.executable, '-X', 'utf8', str(target)]).returncode)

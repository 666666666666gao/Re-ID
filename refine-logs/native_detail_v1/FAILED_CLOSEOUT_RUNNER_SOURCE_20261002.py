"""Record the exit of the once-only saved-distance CPU closeout."""
from datetime import datetime
import json
from pathlib import Path
import subprocess

root = Path('/data/gaob/Re-ID/Trifusion')
command = ['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', '-B',
    str(root / 'tools/analyze_native_detail_failed_campaign.py'), '--campaign',
    str(root / 'logs/native_detail_20261002_v1'), '--output-dir',
    str(root / 'results/native_detail_failed_closeout_20261002')]
started = datetime.now().astimezone().isoformat()
result = subprocess.run(command, cwd=root)
(root / 'logs/native_detail_failed_closeout_execution_20261002.json').write_text(
    json.dumps({'started_at': started, 'completed_at': datetime.now().astimezone().isoformat(),
        'exit_code': result.returncode, 'command': command}, indent=2) + '\n')
raise SystemExit(result.returncode)

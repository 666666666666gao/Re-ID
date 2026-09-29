from datetime import datetime
import json
from pathlib import Path
import time

deadline = datetime.fromisoformat('2026-09-29T21:22:40+08:00').timestamp()
while time.time() < deadline:
    time.sleep(min(30, deadline-time.time()))
source = Path('/data/gaob/Re-ID/Trifusion/.git/check_cross_depth_progress_659.py').read_text()
source = source.replace('cross_depth_progress_659_20260929.json','cross_depth_progress_660_20260929.json')
source = source.replace("phase['recorded_epochs'] = len(training['history'])",
                        "phase['recorded_epochs'] = len(training['history'])\n"
                        "                phase['training_started_at'] = training['started_at']\n"
                        "                phase['epoch_history'] = training['history']")
exec(compile(source, 'cross_depth_progress_660_observer', 'exec'))

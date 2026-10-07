from pathlib import Path
from datetime import datetime
import json
campaign=Path('/data/gaob/Re-ID/Trifusion/logs/qualified_five_incremental_full_v1_20261007_888');journal=Path('/data/gaob/Re-ID/Trifusion/logs/qualified_five_incremental_full_launch_20261007_888')
state=json.loads((campaign/'campaign.json').read_text()) if (campaign/'campaign.json').exists() else None
exit=json.loads((journal/'EXIT.json').read_text()) if (journal/'EXIT.json').exists() else None
active=state.get('active_command') if state else None
training=None
if active and '--output-dir' in active['command']:
    p=Path(active['command'][active['command'].index('--output-dir')+1])/'training.json'
    if p.is_file():
        value=json.loads(p.read_text());training={k:value.get(k) for k in ('status','dataset','history','started_at','completed_at')}
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),state=state,exit=exit,active_training=training,
    supervisor_exists=(Path('/proc')/str(2503715)).exists(),child=json.loads((journal/'CHILD.json').read_text()) if (journal/'CHILD.json').exists() else None)))

from pathlib import Path
from datetime import datetime
import json
campaign=Path('/data/gaob/Re-ID/Trifusion/logs/fixed_five_incremental_best_diagnosis_20261007_892'); journal=Path('/data/gaob/Re-ID/Trifusion/logs/fixed_five_incremental_best_launch_20261007_892')
state=json.loads((campaign/'campaign.json').read_text()) if (campaign/'campaign.json').exists() else None
exit=json.loads((journal/'EXIT.json').read_text()) if (journal/'EXIT.json').exists() else None
logs={}
if state:
    for job in state['jobs']:
        p=campaign/(job['dataset']+'_'+job['objective']+'.log')
        logs[p.name]=p.read_text()[-6000:]
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),state=state,exit=exit,logs=logs,
    supervisor_exists=(Path('/proc')/str(3192933)).exists(),
    child=json.loads((journal/'CHILD.json').read_text()) if (journal/'CHILD.json').exists() else None)))

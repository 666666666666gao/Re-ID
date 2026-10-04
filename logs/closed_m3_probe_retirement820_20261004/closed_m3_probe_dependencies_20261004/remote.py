
from datetime import datetime
from pathlib import Path
import json
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/correspondence_m3_prediction_20260929'
manifest=json.loads((campaign/'manifest.json').read_text())
state=json.loads((campaign/'campaign.json').read_text())
candidates=[]
for folder in (root/'logs',root/'refine-logs',root/'results'):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix=='.json' and ('m3' in path.name or 'role_prediction' in path.name):
            candidates.append({'path':str(path),'bytes':path.stat().st_size})
print(json.dumps({'status':'READ_ONLY_CLOSED_M3_DEPENDENCY_INSPECTION','at':datetime.now().astimezone().isoformat(),
    'campaign':str(campaign),'campaign_state':state,'manifest':manifest,'candidate_reports':candidates}))

from pathlib import Path
from datetime import datetime
import json
journal = Path('/data/gaob/Re-ID/Trifusion/logs/incremental_pending_m0_launch_20261007_880')
campaign = Path('/data/gaob/Re-ID/Trifusion/logs/incremental_pending_m0_v1_20261007_880')
exit_receipt = json.loads((journal / 'EXIT.json').read_text()) if (journal / 'EXIT.json').exists() else None
state = json.loads((campaign / 'campaign.json').read_text()) if (campaign / 'campaign.json').exists() else None
log = (journal / 'controller.log').read_text() if exit_receipt is not None else None
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(), exit=exit_receipt, state=state, log=log)))

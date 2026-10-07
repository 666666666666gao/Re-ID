from pathlib import Path
from datetime import datetime
import json,hashlib
journal=Path('/data/gaob/Re-ID/Trifusion/logs/initial_aux_autocast_context_launch_20261007_884');output=Path('/data/gaob/Re-ID/Trifusion/results/initial_aux_autocast_context_20261007_884/DIAGNOSIS.json')
exit=json.loads((journal/'EXIT.json').read_text()) if (journal/'EXIT.json').exists() else None
result=json.loads(output.read_text()) if output.exists() else None
log=(journal/'controller.log').read_text() if exit is not None else None
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),exit=exit,result=result,log=log,
    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None,
    supervisor_exists=(Path('/proc')/str(2401510)).exists(),
    child=json.loads((journal/'CHILD.json').read_text()) if (journal/'CHILD.json').exists() else None)))

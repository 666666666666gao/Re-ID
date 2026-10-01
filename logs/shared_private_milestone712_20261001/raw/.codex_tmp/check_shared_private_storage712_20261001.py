"""One read-only disk/process milestone; no training or score invocation."""
from datetime import datetime
import json
from pathlib import Path
import shutil

ROOT = Path('/data/gaob/Re-ID/Trifusion')
proof = ROOT / 'logs/shared_private_storage712_20261001.json'
assert not proof.exists()
pids = {}
for name, pid in (('wrapper', 2840894), ('controller', 2846591), ('analysis_waiter', 2910065)):
    proc = Path('/proc') / str(pid)
    pids[name] = {'pid': pid, 'live': proc.exists(),
                  'command': (proc / 'cmdline').read_bytes().replace(b'\x00', b' ').decode() if proc.exists() else None}
disk = shutil.disk_usage(ROOT)
result = {'observed_at': datetime.now().astimezone().isoformat(),
          'processes': pids, 'disk': dict(zip(('total', 'used', 'free'), disk)),
          'scope': 'One read-only process/storage check, no scoring, tensor read, neural call, weight retirement, or environment change.'}
proof.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))

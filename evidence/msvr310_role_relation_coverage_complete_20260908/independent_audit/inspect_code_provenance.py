"""Read supplementary original feature mapping source; no model imports."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
PROJECT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
COMMIT='4e57e542e22c895ac7af96c01357b6a49e4739e4'
for name in ['tools/msvr310_exact_signal_inference.py','tools/train_msvr310_trifusion_oof.py',
             'tools/train_msvr310_source_style.py','modeling/trifusion/signal_preserving_v8.py',
             'modeling/trifusion/source_style_v27.py']:
    raw=(PROJECT/name).read_bytes()
    committed=subprocess.run(['git','show',f'{COMMIT}:{name}'],cwd=PROJECT,capture_output=True,check=False)
    print(json.dumps(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        contents_b64=base64.b64encode(raw).decode(),commit=COMMIT,git_returncode=committed.returncode,
        committed_sha256=hashlib.sha256(committed.stdout).hexdigest())),flush=True)

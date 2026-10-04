
from datetime import datetime
from pathlib import Path
import hashlib,json,os
root=Path('/data2/gb/Re-ID/Trifusion')
directory=os.stat(root)
doc=root/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
data=doc.read_bytes()
print(json.dumps(dict(status='TEXT_DIRECTORY_AND_HANDOFF_READABLE',at=datetime.now().astimezone().isoformat(),root=str(root),directory_inode=directory.st_ino,
 doc_bytes=len(data),doc_sha256=hashlib.sha256(data).hexdigest(),gpu_accessed=False,writes=0)))

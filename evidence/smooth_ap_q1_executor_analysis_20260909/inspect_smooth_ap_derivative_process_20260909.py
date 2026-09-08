from pathlib import Path
import json, time
rows=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or not (p/'cmdline').exists():
        continue
    raw=(p/'cmdline').read_bytes().split(b'\0')
    if raw and raw[0]==b'/root/miniconda3/envs/tri_reid/bin/python':
        rows.append(dict(pid=int(p.name), args=[x.decode() for x in raw if x], stat=(p/'stat').read_text(), stdout_target=str((p/'fd/1').readlink())))
print(json.dumps(dict(observed_at=time.time(), processes=rows)))

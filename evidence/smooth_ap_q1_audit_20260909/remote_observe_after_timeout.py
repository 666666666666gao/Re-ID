from pathlib import Path
import os,json,datetime
rows=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit() and (p/'cmdline').is_file():
  cmd=(p/'cmdline').read_bytes().split(b'\0')
  if len(cmd)>=3 and cmd[0]==b'/root/miniconda3/envs/tri_reid/bin/python' and cmd[1:3]==[b'-B',b'-']:
   rows.append(dict(pid=int(p.name),this_observer=int(p.name)==os.getpid()))
print(json.dumps(dict(observed_at=datetime.datetime.now().astimezone().isoformat(),read_only_stdin_python_processes=rows)))

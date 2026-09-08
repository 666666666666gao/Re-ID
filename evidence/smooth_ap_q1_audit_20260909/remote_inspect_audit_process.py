from pathlib import Path
import json,time
p=Path('/proc/59782')
r={'pid':59782,'exists':p.exists()}
if p.exists():
 cmd=(p/'cmdline').read_bytes().split(b'\0');r['expected_command']=cmd[:3]==[b'/root/miniconda3/envs/tri_reid/bin/python',b'-B',b'-']
 r['status']={l.split(':',1)[0]:l.split(':',1)[1].strip() for l in (p/'status').read_text().splitlines() if l.split(':',1)[0] in ['State','VmRSS','VmSize','Threads']}
 r['fds']=[str(f.readlink()) for f in (p/'fd').iterdir() if f.exists() and '/msvr310_smooth_ap_v1_seed42_2e947a4/' in str(f.readlink())]
print(json.dumps(r))

from datetime import datetime
from pathlib import Path
import json
import paramiko

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'role_input_detach_m0_retirement_observations' / datetime.now().strftime('%H%M%S_%f')
packet.mkdir(parents=True)
code = '''
from datetime import datetime
from pathlib import Path
import json,os
root=Path('/data/gaob/Re-ID/Trifusion')
p=root/'logs/role_input_detach_m0_retirement_20261004/RETIREMENT.json'
record=json.loads(p.read_text()) if p.exists() else None
probes=[str(root/('trained-model/role_input_detach_v1_20261004_813_m0_'+variant+'_'+dataset+'/m0_reload_probe.pth')) for dataset in ('RGBNT201','MSVR310','RGBNT100') for variant in ('semantic','native')]
processes=[]
for entry in Path('/proc').iterdir():
 if entry.name.isdigit() and entry.stat().st_uid==os.getuid() and (entry/'cmdline').is_file():
  cmd=(entry/'cmdline').read_bytes().decode().replace('\\0',' ')
  if cmd=='/usr/bin/python3 -B - ':
   stat=(entry/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
   processes.append({'pid':int(entry.name),'start_ticks':int(fields[19]),'state':fields[0],'cwd':str((entry/'cwd').resolve()),'cmdline':cmd})
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'retirement':record,'probe_exists':{name:Path(name).exists() for name in probes},'python_stdin_processes':processes}))
'''
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,out,err=client.exec_command('/usr/bin/python3 -B -')
stdin.write(code);stdin.channel.shutdown_write();out.channel.settimeout(60)
data,error=out.read(),err.read();status=out.channel.recv_exit_status()
(packet/'stdout.json').write_bytes(data);(packet/'stderr.txt').write_bytes(error)
(packet/'EXIT.json').write_text(json.dumps({'exit_code':status})+'\n',encoding='utf-8')
client.close();assert status==0,error.decode()
print(json.dumps({'packet':str(packet),**json.loads(data)}))

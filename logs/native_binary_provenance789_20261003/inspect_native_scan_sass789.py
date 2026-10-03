from datetime import datetime
from pathlib import Path
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_scan_sass789')
assert not target.exists()
target.mkdir()
code = '''from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess
library=Path('/data/gaob/Re-ID/conda-envs/tri_reid/lib/python3.10/site-packages/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so')
tool='/usr/local/cuda-13.2/bin/cuobjdump'
nm=subprocess.run(['/usr/bin/nm','-D','--defined-only',str(library)],capture_output=True,text=True,check=True)
names=[line.split()[-1] for line in nm.stdout.splitlines() if 'selective_scan_bwd_kernel' in line]
demangled=subprocess.run(['/usr/bin/c++filt'],input='\\n'.join(names)+'\\n',capture_output=True,text=True,check=True).stdout.splitlines()
assert len(names)==len(demangled)
wanted=[(name,pretty) for name,pretty in zip(names,demangled) if any(term in pretty for term in ('Selective_Scan_bwd_kernel_traits<32, 4, false, true, true, true, true, c10::Half, float>','Selective_Scan_bwd_kernel_traits<32, 4, false, true, true, true, true, float, float>'))]
assert len(wanted)==2
version=subprocess.run([tool,'--version'],capture_output=True,text=True,check=True)
elf=subprocess.run([tool,'--list-elf',str(library)],capture_output=True,text=True,check=True)
command=[tool,'--dump-sass','--gpu-architecture','sm_86','--function',','.join(name for name,_ in wanted),str(library)]
dump=subprocess.run(command,capture_output=True,text=True)
sections=[]
for line in dump.stdout.splitlines():
 if 'Function :' in line:
  sections.append({'function':line.split('Function :',1)[1].strip(),'atomic_add_lines':[]})
 if sections and ('RED.' in line or 'ATOM.' in line) and 'ADD' in line:
  sections[-1]['atomic_add_lines'].append(line.strip())
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'binary_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'tool':tool,'tool_version':version.stdout,'elf_list':elf.stdout,'selected_functions':[{'mangled':name,'demangled':pretty} for name,pretty in wanted],'command':command,'exit_code':dump.returncode,'stderr':dump.stderr,'sass':dump.stdout,'sections':sections,'boundary':'Static disassembly of two compiled sm86 selective-scan backward variants only. Does not load the extension or run kernels; no Torch/model/CUDA execution, dependency change, weights or training. Presence of an instruction is not proof it executed or caused historical gradient differences.'}))
'''
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
data, error = stdout.read(), stderr.read()
status = stdout.channel.recv_exit_status()
(target/'stdout.json').write_bytes(data)
(target/'stderr.txt').write_bytes(error)
(target/'EXIT.json').write_text(json.dumps({'exit_code':status,'at':datetime.now().astimezone().isoformat()})+'\n',encoding='utf-8')
client.close()
assert status == 0, error.decode()
record=json.loads(data)
(target/'INTAKE.json').write_bytes(data)
(target/'SELECTIVE_SCAN_SM86.sass.txt').write_text(record['sass'],encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ('sass','command','selected_functions')},indent=2))
assert record['exit_code']==0 and len(record['sections'])==2

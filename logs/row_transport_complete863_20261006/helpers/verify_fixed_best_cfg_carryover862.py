"""CPU-only source configuration replay, never a neural forward or weight change."""
import json
from pathlib import Path
import paramiko

output=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/fixed_best_cfg_carryover862.json')
assert not output.exists()
source='''
from datetime import datetime
import hashlib,json,os,sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
root=Path('/data/gaob/Re-ID/Trifusion')
vendor=root/'comparators/Signal-cd1b0a6'
sys.path.insert(0,str(vendor))
from config import cfg
fresh,carried=cfg.clone(),cfg.clone()
fresh.merge_from_file(str(vendor/'configs/RGBNT100/Signal.yml'))
carried.merge_from_file(str(vendor/'configs/MSVR310/Signal.yml'))
carried.merge_from_file(str(vendor/'configs/RGBNT100/Signal.yml'))
for c in (fresh,carried):
 c.MODEL.PRETRAIN_PATH_T=str(root/'pertrained-model/ViT-B-16.pt')
 c.MODEL.USE_A=False;c.MODEL.USE_B=False
 c.SOLVER.SEED=42;c.SOLVER.MAX_EPOCHS=50;c.DATALOADER.NUM_WORKERS=4
fresh.freeze();carried.freeze()
path=root/'logs/row_mass_role_transport_v2_20261006_860/initialization/RGBNT100_semantic.json'
witness=json.loads(path.read_text())
assert fresh.dump()==witness['binding']['cfg_yaml']
def flatten(c,prefix=''):
 result={}
 for key,value in c.items():
  name=prefix+key
  if hasattr(value,'items'):result.update(flatten(value,name+'.'))
  else:result[name]=value
 return result
f,c=flatten(fresh),flatten(carried)
diff={k:dict(expected=f[k],carried=c[k]) for k in f if f[k]!=c[k]}
assert diff=={'SOLVER.STEPS':dict(expected=[40,70],carried=[20,40])}
print(json.dumps(dict(status='SOURCE_CONFIGURATION_CARRYOVER_REPRODUCED',
 at=datetime.now().astimezone().isoformat(),fresh_cfg_exactly_matches_initializer=True,
 differences=diff,initialization_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 neural_forwards=0,parameter_updates=0,
 boundary='CPU YACS replay shows singleton merge carries MSVR-onlySTEPS intoRGBNT100. '
          'Original formal jobs each ran fresh processes; their sources and accepted results remain unchanged.')))
'''
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',
 key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
stdin,stdout,stderr=client.exec_command("CUDA_VISIBLE_DEVICES='' /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B -")
stdin.write(source)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data,error=stdout.read(),stderr.read()
code=stdout.channel.recv_exit_status()
client.close()
output.write_text(json.dumps(dict(exit_code=code,stdout=data.decode(),stderr=error.decode(),source=source),indent=2)+'\n')
assert code==0,error.decode()
print(data.decode())

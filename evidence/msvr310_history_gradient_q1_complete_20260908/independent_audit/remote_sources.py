import hashlib,json,subprocess
from pathlib import Path
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
run=Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_gradient_v1_seed42_a1b4777')
config=json.loads((repo/'configs/MSVR310/Signal-source-oof-v1.json').read_bytes())
source=Path(config['signal_source'])
paths=[source/n for n in config['signal_source_file_sha256']]
paths += [run/'t0.json',run/'m0_cpu.json',run/'m0/summary.json']
paths += [repo/'evidence/msvr310_history_gradient_complete_source_20260908/source/summary.json',repo/'evidence/msvr310_history_gradient_complete_source_20260908/source/cpu_verification.json']
docs={}
for path in paths:
 b=path.read_bytes();docs[str(path)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode('utf-8')}
train=Path(config['dataset_root'])/'bounding_box_train'
names=sorted(str(p.relative_to(Path(config['dataset_root']))) for p in train.glob('*/*/*') if p.is_file())
print(json.dumps({'documents':docs,'train_file_paths_only':names,'signal_head':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'signal_diff_sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(source),'diff','--binary'])).hexdigest()}))

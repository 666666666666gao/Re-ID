"""Close source-checkpoint config/protocol metadata binding on CPU only."""
import hashlib,json
from pathlib import Path
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
R=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
base_path=R/'configs/MSVR310/Signal-source-oof-v1.json'
base=json.loads(base_path.read_bytes());protocol_path=R/base['protocol'];protocol=json.loads(protocol_path.read_bytes())
config_hash=hashlib.sha256(base_path.read_bytes()).hexdigest();protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
style=json.loads((R/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json').read_bytes())
summary=json.loads(Path(style['BASELINE']['SUMMARY']).read_bytes())
assert summary['config_sha256']==config_hash and summary['protocol_sha256']==protocol_hash
out=[]
for fold,row in zip(protocol['folds'],summary['folds'],strict=True):
    p=torch.load(row['checkpoint'],map_location='cpu',weights_only=True)
    assert p['config_sha256']==config_hash and p['protocol_sha256']==protocol_hash and p['fold']==fold['fold']
    assert p['source_ids']==row['source_ids']==fold['source_ids'] and p['heldout_ids']==row['heldout_ids']==fold['heldout_ids']
    out.append({'fold':fold['fold'],'path':row['checkpoint'],'config_sha256':p['config_sha256'],'protocol_sha256':p['protocol_sha256'],'source_identity_binding_exact':True})
    del p
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'PASS_SOURCE_CHECKPOINT_CONFIG_PROTOCOL_BINDING','folds':out,'model_forwards':0,'optimizer_steps':0}))

import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
assert not (w/'MODULE_CONTRACT_AUDIT_OBSERVED.json').exists()
baseline=json.loads((w/'MODULE_FREE_BASELINE_AUDIT.json').read_bytes())['rows']
expected={row['dataset']:dict(path=row['weight_path'],sha256=row['weight_sha256'],metadata_path=row['metadata'][0]['path'],metadata_sha256=row['metadata'][0]['sha256']) for row in baseline}
pins=json.loads((w/'seed43_m0/logs/near_signal_seed43_launch970/SOURCE_SEAL.json').read_bytes())
code='expected='+repr(expected)+'\npins='+repr(pins)+'\n'+r'''
from datetime import datetime
import hashlib,json
from pathlib import Path
import torch
root=Path('/data/gaob/Re-ID/Trifusion');torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name)==digest for name,digest in pins.items())
configs={'RGBNT201':'configs/RGBNT201/TriFusion-signal-preserving-v27-source-style-rtx3090.json',
'RGBNT100':'configs/RGBNT100/TriFusion-main-v1.json','MSVR310':'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json'}
keys=('FEATURE_WIDTH','SEMANTIC_WIDTH','BRANCH_AFTER_BLOCK','ADAPTER_WIDTH','EXPERT_MODAL_WIDTH','SCALE_INIT','GRADIENT_CHECKPOINTING')
rows=[];shared=[]
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
    source=root/configs[dataset];cfg=json.loads(source.read_text());model=cfg['MODEL']
    contract={key:model[key] for key in keys};shared.append(contract)
    record=expected[dataset];path=Path(record['path']);assert sha(path)==record['sha256']
    metadata_path=Path(record['metadata_path']);assert sha(metadata_path)==record['metadata_sha256']
    receipt=json.loads(metadata_path.read_text())
    assert receipt['status']=='COMPLETE' and not receipt['signal_sim_gam_lam'] and not receipt['trifusion_modules'] and receipt['feature_dim']==1536
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items())
    module_keys=[key for key in state if {'SIM','AlignM'}&set(key.split('.'))];assert not module_keys
    rows.append(dict(dataset=dataset,effective_builder='build_signal_preserving_trifusion_v8_expert_formation',
        configured_architecture=model['ARCHITECTURE'],r2_effective_architecture='signal_preserving_collaborative_v8_expert_formation',
        architecture_override=dataset=='RGBNT201',shared_contract=contract,grid_size=model['GRID_SIZE'],
        loss=cfg['LOSS'],lr=cfg['OPTIMIZATION']['NEW_MODULE_LR'],weight_decay=cfg['OPTIMIZATION']['WEIGHT_DECAY'],
        plain_checkpoint_sha256=record['sha256'],plain_state_key_count=len(state),plain_state_tensor_elements=sum(v.numel() for v in state.values()),
        sim_or_alignm_state_keys=module_keys,plain_metrics=receipt['metrics'],plain_query_count=receipt['query_count'],plain_gallery_count=receipt['gallery_count']))
    del state
assert all(contract==shared[0] for contract in shared)
assert all(row['loss']==rows[0]['loss'] and row['lr']==rows[0]['lr'] and row['weight_decay']==rows[0]['weight_decay'] for row in rows)
assert all(sha(root/name)==digest for name,digest in pins.items())
paths=['tools/official_three_dataset_model.py','tools/build_v12_complete_path_oof_targets.py','modeling/trifusion/signal_preserving_v8_builder.py',
    'modeling/trifusion/signal_preserving_v8.py','tools/train_official_three_dataset_roles.py','tools/run_official_three_dataset_roles.py',
    'tools/queue_signal_plain_baseline.py','tools/evaluate_signal_plain_baseline.py']
print(json.dumps(dict(status='COMPLETE_CURRENT_SOURCE_AND_PLAIN_STATE_CONTRACT_AUDIT',rows=rows,
    shared_contract_equal=True,shared_loss_lr_weight_decay_equal=True,baseline_module_keys_absent=True,
    source_sha256={name:sha(root/name) for name in paths},nn_calls=0,optimizer_updates=0,
    completed_at=datetime.now().astimezone().isoformat(),
    boundary='Current source/config and actual plain-checkpoint key evidence only. No fresh model construction, full inference, complete-goal proof or budget-matched causal attribution. Strong R2 freezes full Signal author weights including SIM and adds own-role training; plain source queue uses separate public CLIP training, seed1234, final epoch50/30/50. Dataset classes/camera count and patch-grid orientation differ legitimately; shared expert family/width/split/loss fields are equal. Existing actual endpoint parity supplies retrieval evidence separately.')))
'''
proc=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B','-'],input=code.encode(),capture_output=True)
(w/'MODULE_CONTRACT_AUDIT.stdout').write_bytes(proc.stdout);(w/'MODULE_CONTRACT_AUDIT.stderr').write_bytes(proc.stderr)
assert proc.returncode==0,proc.stderr.decode()
value=json.loads(proc.stdout);(w/'MODULE_CONTRACT_AUDIT_OBSERVED.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=value['status'],rows=[dict(dataset=r['dataset'],shared_contract=r['shared_contract'],grid_size=r['grid_size'],plain_state_key_count=r['plain_state_key_count'],sim_or_alignm_state_keys=r['sim_or_alignm_state_keys']) for r in value['rows']])))

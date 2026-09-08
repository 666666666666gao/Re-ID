"""Independent CPU-only tensor inventory and state hashing. No model is built."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
START=time.time()
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def state_sha(state):
    h=hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode('utf-8'));h.update(state[name].detach().contiguous().numpy().tobytes())
    return h.hexdigest()
config=read(ROOT/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
protocol=read(ROOT/config['DATA']['PROTOCOL'])
baseline=read(config['BASELINE']['SUMMARY'])
summary=read(RUN/'m0/summary.json')
cpu=read(RUN/'m0_cpu.json')
contract_sha=sha(ROOT/'configs/MSVR310/TriFusion-role-set-paired-v1.json')
assert summary['config_sha256']==contract_sha
checks=[];source_checks=[]
for fold,b0,current in zip(protocol['folds'],baseline['folds'],summary['folds'],strict=True):
    assert fold['fold']==b0['fold']==current['fold']
    path=Path(b0['checkpoint']);actual_sha=sha(path)
    assert actual_sha==b0['checkpoint_sha256']
    payload=torch.load(path,map_location='cpu',weights_only=True)
    assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids'] and payload['fold']==fold['fold']
    original=payload['model_state_dict'];original_sha=state_sha(original)
    assert original_sha==b0['training']['final_state_sha256']
    rows=b0['training']['steps']
    assert len(rows)==b0['training']['optimizer_steps']==650
    for index,row in enumerate(rows):
        assert row['step']==index+1
        assert set(row['sampled_record_indices'])<=set(fold['source_record_indices'])
    source_checks.append(dict(fold=fold['fold'],checkpoint=str(path),file_sha256=actual_sha,state_sha256=original_sha,checkpoint_keys=sorted(payload),state_tensors=len(original),source_training_steps=len(rows),source_ids=len(payload['source_ids']),heldout_ids=len(payload['heldout_ids']),all_650_saved_source_batches_isolated=True))
    for endpoint in ('control','role_set'):
        row=current['endpoints'][endpoint];path=Path(row['checkpoint']);actual_sha=sha(path)
        assert actual_sha==row['checkpoint_sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert set(payload)=={'role_state_dict','baseline_aliases','binding','fold','source_ids','heldout_ids','config_sha256'}
        assert payload['config_sha256']==contract_sha and payload['binding']==row['initialization']
        assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids'] and payload['fold']==fold['fold']
        roles=payload['role_state_dict'];aliases=payload['baseline_aliases']
        assert not set(roles)&set(aliases) and all(x.startswith('baseline.') for x in aliases) and all(not x.startswith('baseline.') for x in roles)
        assert set(aliases.values())<=set(original)
        state={k:original[v] for k,v in aliases.items()};state.update(roles)
        final=state_sha(state)
        assert final==row['training']['final_state_sha256']==row['strict_reload_state_sha256']
        names=payload['binding']['trainable_names']
        assert len(names)==len(set(names))==203 and set(names)<=set(roles)
        assert len([n for n in names if n.startswith('encoder.')])==189
        assert sum(state[n].numel() for n in names)==payload['binding']['trainable_parameters']
        assert all(torch.isfinite(t).all().item() for t in state.values())
        nontrain_roles=sorted(set(roles)-set(names))
        buffers=[n for n in nontrain_roles if n.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
        frozen_role_parameters=sorted(set(nontrain_roles)-set(buffers))
        assert len(frozen_role_parameters)==7 and all(n.endswith('_neck.bias') or '_necks.' in n and n.endswith('.bias') for n in frozen_role_parameters)
        frozen={k:v for k,v in state.items() if k in aliases or k in frozen_role_parameters}
        frozen_sha=state_sha(frozen)
        assert frozen_sha==row['training']['frozen_state_after_sha256']==row['training']['frozen_state_before_sha256']
        assert original_sha==row['training']['signal_state_before_sha256']==row['training']['signal_state_after_sha256']==row['initialization']['signal_state_sha256']
        checks.append(dict(endpoint=f"fold_{fold['fold']}_{endpoint}",checkpoint=str(path),bytes=path.stat().st_size,checkpoint_sha256=actual_sha,full_state_sha256=final,frozen_state_sha256=frozen_sha,signal_state_sha256=original_sha,baseline_alias_entries=len(aliases),role_state_entries=len(roles),trainable_tensors=len(names),encoder_tensors=189,trainable_parameters=payload['binding']['trainable_parameters'],claimed_total_parameters=payload['binding']['total_parameters'],frozen_role_parameter_names=frozen_role_parameters,role_buffer_names=buffers,finite_all_state=True,strict_reload_state_hash_match=True,output_bitwise_equality_independently_rerun=False))
        del payload,state,roles,frozen
    del original
for f in summary['folds']:
    assert f['endpoints']['control']['initialization']==f['endpoints']['role_set']['initialization']
assert not torch.cuda.is_initialized()
receipt_files=[]
for path,expected in cpu['files'].items():
    actual={'bytes':Path(path).stat().st_size,'sha256':sha(path)}
    assert actual==expected,(path,actual,expected)
    receipt_files.append(dict(path=path,**actual))
result=dict(status='PASS_INDEPENDENT_M0_CHECKPOINT_STATE_HASHES',torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),cuda_initialized=torch.cuda.is_initialized(),source_checkpoint_checks=source_checks,m0_checkpoint_checks=checks,cpu_receipt_file_hash_checks=receipt_files,paired_initialization_bindings_equal=True,model_constructions=0,model_forwards=0,model_backwards=0,image_reads=0,q1_artifacts_read=0,elapsed_seconds=time.time()-START,limits=['No initial role-state checkpoint is saved; deterministic construction and paired initial hashes are run evidence.','Strict model.load_state_dict and output bitwise equality are execution assertions; this audit rebuilds saved tensor maps and hashes only.'])
print(json.dumps(result,indent=2))

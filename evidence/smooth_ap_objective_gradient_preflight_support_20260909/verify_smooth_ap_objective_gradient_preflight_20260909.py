from pathlib import Path
import hashlib,json,sys
import torch

repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID');sys.path.insert(0,str(repo))
from tools.msvr_smooth_ap import paired_objectives
from tools.run_signal_preserving_v5 import weighted_training_loss
from tools.probe_msvr_history_candidate_gradients import compare
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f')
pipeline=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
assert pipeline['status']=='COMPLETE' and pipeline['exit_code']==0
summary=json.loads((root/'summary.json').read_bytes());assert summary['status']=='PASS_PREFLIGHT'
assert len(summary['conditions'])==6
config=json.loads((repo/'configs/MSVR310/TriFusion-source-style-paired-v1.json').read_bytes())
torch.set_num_threads(4)
result=dict(status='RUNNING',steps=0,stored_gradient_tensors=0,checked_roles=[],max_loss_error=0.,
    max_gradient_stat_error=0.,optimizer_updates=0,model_forwards=0,
    scope='all 48 distance/loss rows; saved first-history three gradient groups per endpoint; runtime checks otherwise')
for cond in summary['conditions']:
    directory=root/cond['directory']
    for name,entry in cond['files'].items():
        b=(directory/name).read_bytes();assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
    rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
    assert len(rows)==8 and [r['step'] for r in rows]==list(range(1,9))
    data=torch.from_file(str(directory/'distances.f32'),shared=False,size=(directory/'distances.f32').stat().st_size//4,dtype=torch.float32)
    used=0
    for row in rows:
        assert row['distance_offset_bytes']==used*4
        count=len(row['memory']);part=data[used:used+row['distance_float_count']]
        assert len(part)==64*(64+count) and bool(torch.isfinite(part).all())
        dc=part[:4096].reshape(64,64);dh=part[4096:].reshape(64,count)
        hard,smooth,_=paired_objectives(dc,dh,row['identities'],[r['identity'] for r in row['memory']])
        target=hard if row['active_fused_metric']=='control' else smooth
        errors=[abs(float(target)-row['fused_loss']),abs(float(weighted_training_loss(row['components'],config))-row['total_loss'])]
        other=dict(row['components']);other['triplet_fused']=0.
        errors.append(abs(float(weighted_training_loss(other,config))-row['other_loss']))
        assert max(errors)<=2e-6;result['max_loss_error']=max(result['max_loss_error'],*errors)
        for k in ('current_decomposition','full_decomposition'):
            assert row[k]['relative_to_sum_of_component_norms']<=.005
        used+=row['distance_float_count'];result['steps']+=1
    assert used==len(data)
    payload=torch.load(directory/'first_history_gradients.pt',map_location='cpu',weights_only=True)
    assert payload['names']==cond['parameters'] and len(payload['names'])==189
    assert payload['scale']==256. and payload['step']==cond['direct_history_proof']['step']
    record=rows[payload['step']-1]
    for group in ('fused','other','full'):
        assert len(payload[group])==189 and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in payload[group])
        result['stored_gradient_tensors']+=len(payload[group])
    for expert in ('cnn','transformer','mamba'):
        ix=[i for i,n in enumerate(payload['names']) if n.startswith('encoder.'+expert+'_')];assert ix
        for name,a,b in [('fused_vs_other','fused','other'),('fused_vs_full','fused','full')]:
            actual=compare([payload[a][i] for i in ix],[payload[b][i] for i in ix])
            expected=record['roles'][expert][name]
            for key,value in actual.items():
                if value is None:assert expected[key] is None
                else:
                    error=abs(value-expected[key]);assert error<=1e-9*(1+abs(expected[key]))
                    result['max_gradient_stat_error']=max(result['max_gradient_stat_error'],error)
        result['checked_roles'].append(dict(condition=cond['directory'],expert=expert,step=payload['step']))
    del payload,data
assert result['steps']==48 and result['stored_gradient_tensors']==3402 and len(result['checked_roles'])==18
result.update(status='PASS_COMPLETE_OBJECTIVE_GRADIENT_PREFLIGHT',summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest())
target=root/'preflight_verification.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

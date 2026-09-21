"""State witnesses and exact disk serialization checks for the new run only."""
import math
import torch
from tools.train_msvr310_signal_oof import sha256


def task_state_summary(optimizer,parameters,groups):
    result={}
    for role,indexes in groups.items():
        keys=set(optimizer.state[parameters[indexes[0]]])
        assert all(set(optimizer.state[parameters[i]])==keys for i in indexes)
        tasks={}
        for key in sorted(keys):
            states=[optimizer.state[parameters[i]][key] for i in indexes]
            counts={s['step'] for s in states};assert len(counts)==1
            tasks[key]=dict(step=counts.pop(),
                first_moment_norm=math.sqrt(sum(float(s['exp_avg'].double().square().sum()) for s in states)),
                second_moment_norm=math.sqrt(sum(float(s['exp_avg_sq'].double().square().sum()) for s in states)))
        result[role]=tasks
    return result


def save_optimizer_state(optimizer,scaler,path,names,endpoint):
    assert not path.exists()
    original=optimizer.state_dict()
    payload=dict(optimizer=original,scaler=scaler.state_dict(),parameter_names=names,endpoint=endpoint)
    torch.save(payload,path)
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    assert loaded['parameter_names']==names and loaded['endpoint']==endpoint
    assert loaded['scaler']==scaler.state_dict()
    assert loaded['optimizer']['param_groups']==original['param_groups']
    assert set(loaded['optimizer']['state'])==set(original['state'])
    count=0
    for index,state in original['state'].items():
        restored=loaded['optimizer']['state'][index]
        assert set(restored)==set(state)
        for key,task in state.items():
            assert restored[key]['step']==task['step']
            for moment in ('exp_avg','exp_avg_sq'):
                assert torch.equal(restored[key][moment],task[moment].cpu())
                count+=1
    optimizer.load_state_dict(loaded['optimizer'])
    scaler.load_state_dict(loaded['scaler'])
    current=optimizer.state_dict()
    assert current['param_groups']==loaded['optimizer']['param_groups']
    for index,state in current['state'].items():
        for key,task in state.items():
            assert task['step']==loaded['optimizer']['state'][index][key]['step']
            for moment in ('exp_avg','exp_avg_sq'):
                assert torch.equal(task[moment].cpu(),loaded['optimizer']['state'][index][key][moment])
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha256(path),
                endpoint=endpoint,parameter_count=len(names),moment_tensors_checked=count,
                disk_payload_exact=True,reloaded_moments_exact=True,scaler_exact=True,
                scope='Final state serialization/reload only; no extra model update or resumed training.')

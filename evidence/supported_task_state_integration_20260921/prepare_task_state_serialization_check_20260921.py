from pathlib import Path
root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');tmp=Path('D:/Program Files/UserCache/gb/codex/tmp')
modules={name:(root/'tools'/f'{name}.py').read_text() for name in ['msvr_task_state_optimizer','msvr_task_state_records','verify_msvr_task_state_records']}
payload="import os,sys,types,json,copy\nfrom pathlib import Path\nos.environ['CUDA_VISIBLE_DEVICES']=''\nsys.path.insert(0,'/root/autodl-tmp/trifusion-v2/TriFusion-ReID')\nimport tools,torch\n"
payload+='modules='+repr(modules)+'\n'
payload+='''for name,source in modules.items():
    module=types.ModuleType('tools.'+name);sys.modules[module.__name__]=module
    exec(compile(source,name,'exec'),module.__dict__)
from tools.msvr_task_state_optimizer import SupportedTaskAdamW
from tools.msvr_task_state_records import task_state_summary,save_optimizer_state
from tools.verify_msvr_task_state_records import verify_states
root=Path('/root/autodl-tmp/trifusion-v2/artifacts/task_state_serialization_synthetic_r2_20260921')
root.mkdir()
torch.manual_seed(42)
names=[f'encoder.{role}_fixture.p{i}' for role,n in [('cnn',42),('transformer',54),('mamba',93)] for i in range(n)]+[f'head.p{i}' for i in range(14)]
groups={role:[i for i,name in enumerate(names[:189]) if name.startswith('encoder.'+role+'_')] for role in ('cnn','transformer','mamba')}
results=[]
for split in (False,True):
    endpoint='split' if split else 'control'
    parameters=[torch.nn.Parameter(torch.randn(2)) for _ in names]
    roles=parameters[:189]
    opt=SupportedTaskAdamW(parameters,roles,split=split,lr=.0003,weight_decay=.01)
    scaler=torch.amp.GradScaler('cpu',init_scale=256.)
    scaler.scale(parameters[0].sum())
    audits=[]
    for step in range(4):
        observed=step!=1
        r=[torch.randn_like(p) if observed and step!=2 else torch.zeros_like(p) for p in roles]
        a=[torch.randn_like(p) for p in roles]
        for p in parameters:p.grad=torch.randn_like(p)
        for p,x,y in zip(roles,r,a):p.grad=x+y
        before=[p.detach().clone() for p in roles]
        opt.step(rank_gradients=r,auxiliary_gradients=a,rank_observed=observed)
        from tools.probe_msvr_history_candidate_gradients import compare
        updates={role:compare([before[i] for i in indexes],[roles[i].detach() for i in indexes]) for role,indexes in groups.items()}
        ids=['a','a','b'];scenes=['x','y' if observed else 'x','x']
        support=dict(eligible_anchors=2 if observed else 0,eligible_identities=int(observed),identity_directed_scene_relations=2 if observed else 0)
        audits.append(dict(support=support,rank_observed=observed,identities=ids,scenes=scenes,memory=[],
            relation_objective=dict(cross_scene_positive_counts=[1,1,0] if observed else [0,0,0]),
            direct_single_group_check={},direct_component_backward_calls=0,actual_parameter_updates=updates,
            direct_ra_assembly=True,classification_head_gradients_unchanged=True,
            current_rank_backward_calls=1,current_auxiliary_backward_calls=1,
            task_states=task_state_summary(opt,roles,groups),rank_auxiliary_reference_checks={}))
    proof=save_optimizer_state(opt,scaler,root/(endpoint+'.pt'),names,endpoint)
    tr=dict(optimizer_state=proof,task_state_split=split,steps=[dict(amp_scale_after=256.)]*4,
        mode='comparison',current_rank_backward_calls=4,current_auxiliary_backward_calls=4,direct_component_backward_calls=0)
    check=verify_states(audits,tr,endpoint,0)
    corrupted=copy.deepcopy(audits);corrupted[-1]['rank_observed']=False
    rejected=False
    try:verify_states(corrupted,tr,endpoint,0)
    except AssertionError:rejected=True
    assert rejected
    results.append(dict(endpoint=endpoint,proof=proof,check=check,corrupted_support_rejected=True))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_SYNTHETIC_DISK_STATE_CHECK',results=results,
    cuda_initialized=False,model_forwards=0,training_updates=0,scope='203 small synthetic tensors, no TriFusion model or images'),indent=2))
'''
(tmp/'task_state_serialization_payload_20260921.py').write_text(payload,encoding='utf-8')

"""CPU synthetic checks against native AdamW, not training evidence."""
import copy
import json
import torch
from tools.msvr_task_state_optimizer import SupportedTaskAdamW


def checks():
    torch.manual_seed(42)
    rows=[]
    for split in (False, True):
        role=torch.nn.Parameter(torch.randn(17))
        head=torch.nn.Parameter(torch.randn(7))
        reference_role=role.detach().clone()
        reference_head=torch.nn.Parameter(head.detach().clone())
        opt=SupportedTaskAdamW([role,head],[role],split=split,lr=.0003,weight_decay=.01)
        # Native optimizers supply task directions; their parameters reset to
        # zero before every step, leaving their moments intact.
        native_parameters=[torch.nn.Parameter(torch.zeros_like(role)) for _ in range(2 if split else 1)]
        natives=[torch.optim.AdamW([p],lr=.0003,weight_decay=0.,foreach=False) for p in native_parameters]
        head_opt=torch.optim.AdamW([reference_head],lr=.0003,weight_decay=.01,foreach=False)
        observed_count=0
        maximum_error=0.
        for step in range(12):
            observed=step not in (0,4,5,10)
            r=torch.randn_like(role) if observed and step!=7 else torch.zeros_like(role)
            a=torch.randn_like(role)
            h=torch.randn_like(head)
            lr=.0003 if step<6 else .0001
            opt.param_groups[0]['lr']=lr
            head_opt.param_groups[0]['lr']=lr
            for native in natives:native.param_groups[0]['lr']=lr
            previous_rank=copy.deepcopy(opt.state[role].get('rank'))
            role.grad=r+a;head.grad=h.clone();reference_head.grad=h.clone()
            opt.step(rank_gradients=[r],auxiliary_gradients=[a],rank_observed=observed)
            if observed:observed_count+=1
            contributions=[]
            for index,(p,native) in enumerate(zip(native_parameters,natives,strict=True)):
                if split and index==0 and not observed:continue
                with torch.no_grad():p.zero_()
                p.grad=(r if index==0 else a) if split else r+a
                native.step()
                contributions.append(p.detach().clone())
            reference_role.mul_(1-lr*.01)
            for contribution in contributions:reference_role.add_(contribution)
            head_opt.step()
            error=float((role-reference_role).abs().max())
            maximum_error=max(maximum_error,error)
            torch.testing.assert_close(role,reference_role,rtol=2e-6,atol=2e-7)
            torch.testing.assert_close(head,reference_head,rtol=2e-6,atol=2e-7)
            if split:
                assert opt.state[role]['auxiliary']['step']==step+1
                assert opt.state[role].get('rank',{'step':0})['step']==observed_count
                if not observed and previous_rank is not None:
                    for key in ('exp_avg','exp_avg_sq'):
                        assert torch.equal(previous_rank[key],opt.state[role]['rank'][key])
            # Resume through the public state_dict interface with identical next
            # gradients. Check exact state/parameter continuation every step.
            clone_role=torch.nn.Parameter(role.detach().clone())
            clone_head=torch.nn.Parameter(head.detach().clone())
            resumed=SupportedTaskAdamW([clone_role,clone_head],[clone_role],split=split,lr=lr,weight_decay=.01)
            resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
            snapshot=copy.deepcopy(opt.state_dict())
            role_before=role.detach().clone();head_before=head.detach().clone()
            clone_role.grad=role.grad.clone();clone_head.grad=head.grad.clone()
            opt.step(rank_gradients=[r],auxiliary_gradients=[a],rank_observed=observed)
            resumed.step(rank_gradients=[r],auxiliary_gradients=[a],rank_observed=observed)
            assert torch.equal(role,clone_role) and torch.equal(head,clone_head)
            with torch.no_grad():role.copy_(role_before);head.copy_(head_before)
            opt.load_state_dict(snapshot)
        # A nonfinite task buffer must be rejected before any update, even if
        # combined p.grad supplied to the scaler is finite.
        before_role=role.detach().clone();before_head=head.detach().clone()
        state_before=copy.deepcopy(opt.state_dict())
        bad=torch.full_like(role,float('inf'))
        rejected=False
        try:
            opt.step(rank_gradients=[bad],auxiliary_gradients=[a],rank_observed=True)
        except AssertionError:
            rejected=True
        assert rejected and torch.equal(role,before_role) and torch.equal(head,before_head)
        for parameter_id,state in state_before['state'].items():
            for key,task in state.items():
                now=opt.state_dict()['state'][parameter_id][key]
                assert task['step']==now['step']
                assert torch.equal(task['exp_avg'],now['exp_avg']) and torch.equal(task['exp_avg_sq'],now['exp_avg_sq'])
        rows.append(dict(split=split,steps=12,observed_rank_steps=observed_count,
                         max_native_parameter_error=maximum_error,resume_checks=12,
                         absent_steps=[1,5,6,11],supported_zero_step=8,nonfinite_atomic_stop=True))
    amp_rows=[]
    for split in (False,True):
        parameter=torch.nn.Parameter(torch.tensor([.3,-.8]))
        reference=torch.nn.Parameter(parameter.detach().clone())
        opt=SupportedTaskAdamW([parameter],[parameter],split=split,lr=.0003,weight_decay=.01)
        ref_opt=SupportedTaskAdamW([reference],[reference],split=split,lr=.0003,weight_decay=.01)
        scaler=torch.amp.GradScaler('cpu',init_scale=256.)
        for step in range(6):
            observed=step!=3
            opt.zero_grad(set_to_none=True)
            rank=parameter.square().sum() if observed else parameter.sum()*0
            auxiliary=(parameter-.7).square().sum()
            scale=scaler.get_scale()
            r=torch.autograd.grad(rank*scale,parameter,retain_graph=True)[0]
            a=torch.autograd.grad(auxiliary*scale,parameter,retain_graph=True)[0]
            scaler.scale(rank+auxiliary).backward()
            parameter.grad.copy_(r+a)
            scaler.unscale_(opt)
            reference.grad=parameter.grad.clone()
            ref_opt.step(rank_gradients=[r/scale],auxiliary_gradients=[a/scale],rank_observed=observed)
            scaler.step(opt,rank_gradients=[r/scale],auxiliary_gradients=[a/scale],rank_observed=observed)
            scaler.update()
            assert torch.equal(parameter,reference)
        amp_rows.append(dict(split=split,steps=6,scaler_device='cpu',scale=scaler.get_scale(),exact_parameter_match=True))
    return dict(status='PASS_SYNTHETIC_CPU_ONLY',torch=torch.__version__,cuda_initialized=torch.cuda.is_initialized(),
                seed=42,rows=rows,grad_scaler_kwargs_checks=amp_rows,model_forwards=0,training_updates=0)


if __name__=='__main__':
    print(json.dumps(checks(),indent=2))

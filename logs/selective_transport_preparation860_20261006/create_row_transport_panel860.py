from pathlib import Path
import ast

r=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
for kind in ('queue','report'):
 text=(r/f'tools/{kind}_selective_role_transport.py').read_text(encoding='utf-8')
 text=text.replace('selective_role_transport','row_mass_role_transport')
 text=text.replace('trifusion-selective-role-transport-v1','trifusion-row-mass-role-transport-v2')
 text=text.replace('selective_transport_prepare_unused','row_transport_prepare_unused')
 text=text.replace('same-P total matrix mass, not message energy or OT column constraints',
                   'same-score total row-null real mass, not message energy; there are no OT column constraints')
 target=r/f'tools/{kind}_row_mass_role_transport.py'
 assert not target.exists();target.write_text(text,encoding='utf-8')
old=(r/'tools/check_selective_role_transport.py').read_text(encoding='utf-8')
tail=old[old.index('def real_pair('):]
tail=tail.replace('queue_selective_role_transport','queue_row_mass_role_transport')
tail=tail.replace('selective_transport_pair_unused','row_transport_pair_unused')
tail=tail.replace('entry.previous.configure()','entry.previous.previous.configure()')
tail=tail.replace('entry.original_build_core(args, protocol)','entry.previous.original_build_core(args, protocol)')
header='''"""Exact row-mass controls, activity and full author-batch initialization."""
import argparse,json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_row_mass_role_transport as entry
from trifusion.row_mass_role_transport import RowMassTransport,row_log_assignment
from trifusion.selective_role_transport import message_weights


def component_witness():
    torch.set_num_threads(1);torch.manual_seed(42)
    strong=torch.full((2,16,16),-8.0)
    strong.diagonal(dim1=-2,dim2=-1).fill_(8.0)
    weak=torch.full_like(strong,-8.0);mixed=torch.randn_like(strong)
    rows=[];real_masses={}
    for name,scores in [('diagonal',strong),('unmatched',weak),('mixed',mixed)]:
        assignment=row_log_assignment(scores)
        assert torch.isfinite(assignment).all()
        residual=float((assignment.exp().sum(-1)-1).abs().max())
        assert residual<=1e-5
        for oriented in (scores,scores.transpose(-1,-2)):
            real=row_log_assignment(oriented)[...,:16]
            candidate,mass,q=message_weights(real,'slot_mass')
            control,other_mass,other_q=message_weights(real,'uniform_mass')
            assert mass.amin()>=0 and mass.amax()<=1+1e-5
            assert torch.equal(mass,other_mass) and torch.equal(q,other_q)
            assert torch.allclose(candidate.sum((-2,-1)),control.sum((-2,-1)),atol=1e-5,rtol=1e-5)
            assert torch.allclose(control.sum(-1),mass.mean(-1,keepdim=True).expand_as(mass),atol=1e-5,rtol=1e-5)
            if name=='mixed':assert not torch.allclose(candidate,control,atol=1e-5,rtol=1e-5)
        real_masses[name]=float(assignment[...,:16].exp().sum(-1).mean())
        rows.append(dict(condition=name,maximum_augmented_row_residual=residual,real_mass_mean=real_masses[name]))
    assert real_masses['unmatched']<real_masses['diagonal']
    activity=[]
    for mode in ('slot_mass','uniform_mass'):
        torch.manual_seed(42);module=RowMassTransport(mode)
        semantic,private,target=[torch.randn(2,3,16,128) for _ in range(3)]
        assert len(list(module.parameters()))==2 and sum(p.numel() for p in module.parameters())==32768
        assert torch.equal(module(semantic,private),private)
        initial={n:p.detach().clone() for n,p in module.named_parameters()}
        active=set();optimizer=torch.optim.Adam(module.parameters(),lr=0.001)
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(module(semantic,private),target).backward()
            for name,parameter in module.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                if parameter.grad.abs().max()>0:active.add(name)
            optimizer.step()
        assert active==set(initial)
        assert all(not torch.equal(p,initial[n]) for n,p in module.named_parameters())
        restored=RowMassTransport(mode);restored.load_state_dict(module.state_dict(),strict=True)
        assert torch.equal(restored(semantic,private),module(semantic,private))
        activity.append(dict(mass_mode=mode,all_two_tensors_active_and_changed=True,
                             zero_initial_exit=True,strict_component_reload_exact=True))
    return dict(status='CPU_SYNTHETIC_COMPONENT_PASS',assignment_rows=rows,activity_rows=activity,
                boundary='Directional row softmax with null; same-score total real matrix mass matches, not vector energy. No Sinkhorn or OT column constraints. Old V1 mathematical FAIL unchanged. Real full-batch M0 still required.')


'''
text=header+tail
target=r/'tools/check_row_mass_role_transport.py';assert not target.exists();target.write_text(text,encoding='utf-8')
for name in ('modeling/trifusion/row_mass_role_transport.py','tools/run_row_mass_role_transport.py',
             'tools/queue_row_mass_role_transport.py','tools/report_row_mass_role_transport.py',
             'tools/check_row_mass_role_transport.py'):
 ast.parse((r/name).read_text(encoding='utf-8'))
print('FIVE_ROW_VARIANT_SOURCES_AST_PASS')

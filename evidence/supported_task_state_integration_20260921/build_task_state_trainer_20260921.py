from pathlib import Path
import ast
root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
s=(root/'tools/train_msvr_supported_gradient_balance.py').read_text()
s=s.replace('Only the role-block gradient combination changes; M0 precedes heldout access.',
'''Shared versus separate task AdamW states; both use direct R+A.
M0 precedes heldout access. No R2 EMA controller is active.''')
s=s.replace("ENDPOINTS = ('control', 'balanced')","ENDPOINTS = ('control', 'split')")
s=s.replace("'msvr310-supported-gradient-balance-paired-v1'","'msvr310-supported-task-state-paired-v1'")
s=s.replace("    assert spec['implementation_revision']=='r2_direct_auxiliary'\n",'')
s=s.replace("    from tools.msvr_supported_gradient_balance import RULE\n    assert spec['candidate_reduction']=='mean_eligible_anchors' and spec['gradient_balance']==RULE", "    assert spec['candidate_reduction']=='mean_eligible_anchors'\n    assert spec['task_state_rule']=='equal_direct_ra_supported_split_adamw_v1'")
s=s.replace('(SupportedBalance, role_indices, scalar_gradients, combine, check_reference, support_counts)', '(role_indices, scalar_gradients, check_reference, support_counts)')
s=s.replace('    controller=SupportedBalance()\n','')
s=s.replace("    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],", "    from tools.msvr_task_state_optimizer import SupportedTaskAdamW\n    from tools.msvr_task_state_records import task_state_summary, save_optimizer_state\n    optimizer = SupportedTaskAdamW([p for p in model.parameters() if p.requires_grad], parameters, split=endpoint=='split',")
start=s.index('                balance,full_rank,auxiliary=combine(')
end=s.index('                before_parameters=',start)
s=s[:start]+'''                full_rank=[r+h for r,h in zip(current_rank,history_gradient,strict=True)]
                auxiliary=current_auxiliary
                rank_observed=(not active) or support['eligible_anchors']>0
                if not rank_observed:
                    assert all(not bool(g.abs().sum()) for g in full_rank)
                for p,r,a in zip(parameters,full_rank,auxiliary,strict=True):
                    p.grad.copy_(r+a)
                head_preserved=all(torch.equal(p.grad,g) for (_,p),g in zip(heads,head_gradients,strict=True))
                assert head_preserved
                assembled={};reference_checks={}
                for expert,indexes in role_groups.items():
                    actual=[parameters[i].grad/scale for i in indexes]
                    assembled[expert]=compare([current[i]/scale for i in indexes],actual)
                    if direct_parts is not None:
                        reference_checks[expert]=dict(
                            current_rank=check_reference([current_rank[i]/scale for i in indexes],[direct_parts['current'][i]/scale for i in indexes]),
                            historical_rank=check_reference([history_gradient[i]/scale for i in indexes],[direct_parts['history'][i]/scale for i in indexes]),
                            full_rank=check_reference([full_rank[i]/scale for i in indexes],[direct_parts['rank'][i]/scale for i in indexes]),
                            auxiliary=check_reference([auxiliary[i]/scale for i in indexes],[direct_parts['auxiliary'][i]/scale for i in indexes]),
                            assembled=check_reference(actual,[(direct_parts['rank'][i]+direct_parts['auxiliary'][i])/scale for i in indexes]))
                rank_values=[g/scale for g in full_rank]
                auxiliary_values=[g/scale for g in auxiliary]
                del direct,direct_parts,full_rank,auxiliary,current_rank,current_auxiliary,head_gradients
'''+s[end:]
s=s.replace('                scaler.step(optimizer);scaler.update()', '''                # Separate buffers must be finite even when their sum is finite.
                assert all(torch.isfinite(g).all() for g in rank_values+auxiliary_values)
                scaler.step(optimizer,rank_gradients=rank_values,auxiliary_gradients=auxiliary_values,
                            rank_observed=rank_observed)
                scaler.update()
                task_states=task_state_summary(optimizer,parameters,role_groups)
                del rank_values,auxiliary_values''')
s=s.replace('roles=role_stats,applied_gradients=applied,','roles=role_stats,assembled_gradients=assembled,')
s=s.replace('support=support,gradient_balance=balance,actual_parameter_updates=updates,','support=support,rank_observed=rank_observed,task_states=task_states,actual_parameter_updates=updates,')
s=s.replace('final_gradient_addition_bitwise=True','direct_ra_assembly=True')
s=s.replace("event='supported_balance_epoch'","event='supported_task_state_epoch'")
s=s.replace('    report=dict(mode=mode,','''    optimizer_proof=save_optimizer_state(optimizer,scaler,directory/'optimizer_state.pt',
        [n for n,p in model.named_parameters() if p.requires_grad],endpoint)
    report=dict(mode=mode,''')
s=s.replace('gradient_balance_state=controller.states,current_rank_backward_calls=len(steps),','optimizer_state=optimizer_proof,current_rank_backward_calls=len(steps),')
s=s.replace("gradient_balancing_applied=endpoint=='balanced'", "task_state_split=endpoint=='split'")
s=s.replace("f['endpoints']['balanced']","f['endpoints']['split']")
s=s.replace("result['endpoints']['balanced']","result['endpoints']['split']")
s=s.replace('PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_M0','PASS_COMPLETE_SUPPORTED_TASK_STATE_M0')
s=s.replace("update_rules=dict(control='cross_scene_ap_original_gradient_sum', balanced='cross_scene_ap_supported_role_gradient_balance'),\n                   gradient_balance=spec['gradient_balance'],", "update_rules=dict(control='direct_ra_shared_adamw', split='direct_ra_supported_task_adamw'),\n                   task_state_rule=spec['task_state_rule'],")
s=s.replace("result['endpoints']['balanced']['initialization']", "result['endpoints']['split']['initialization']")
assert 'balanced' not in s, [line for line in s.splitlines() if 'balanced' in line]
ast.parse(s)
(root/'tools/train_msvr_supported_task_state.py').write_text(s,encoding='utf-8')
print('TRAINER_WRITTEN_NOT_RUN')

s=(root/'tools/verify_msvr_supported_gradient_balance.py').read_text()
s=s.replace('train_msvr_supported_gradient_balance','train_msvr_supported_task_state')
s=s.replace('SUPPORTED_GRADIENT_BALANCE','SUPPORTED_TASK_STATE').replace("'balanced'","'split'")
s=s.replace('final_gradient_addition_bitwise','direct_ra_assembly').replace('applied_gradients','assembled_gradients')
start=s.index("                if endpoint=='control' or index<warmup or not counts.any():")
end=s.index("            direct=audit['direct_single_group_check']",start)
s=s[:start]+s[end:]
s=s.replace('    from tools.verify_msvr_supported_gradient_balance_stats import verify_balance\n    balance_check=verify_balance(audits,tr,endpoint,warmup)',
'''    from tools.verify_msvr_task_state_records import verify_states
    state_check=verify_states(audits,tr,endpoint,warmup)''')
s=s.replace('gradient_balance=balance_check','task_state_check=state_check')
s=s.replace("local/'memory_distances.f32'): record(p)","local/'memory_distances.f32',Path(tr['optimizer_state']['path'])): record(p)")
ast.parse(s);(root/'tools/verify_msvr_supported_task_state.py').write_text(s,encoding='utf-8')
s=(root/'tools/check_msvr_supported_gradient_balance.py').read_text()
s=s.replace('train_msvr_supported_gradient_balance','train_msvr_supported_task_state')
s=s.replace('check_msvr_supported_gradient_balance_math','check_msvr_task_state_math')
s=s.replace('balance_checks','task_checks').replace('gradient_balance_math','task_state_math')
s=s.replace('SUPPORTED_GRADIENT_BALANCE','SUPPORTED_TASK_STATE')
ast.parse(s);(root/'tools/check_msvr_supported_task_state.py').write_text(s,encoding='utf-8')
s=(root/'tools/run_msvr_supported_gradient_balance.py').read_text()
s=s.replace('msvr_supported_gradient_balance','msvr_supported_task_state')
s=s.replace('SUPPORTED_GRADIENT_BALANCE','SUPPORTED_TASK_STATE')
ast.parse(s);(root/'tools/run_msvr_supported_task_state.py').write_text(s,encoding='utf-8')
print('T0_RUNNER_VERIFIER_WRITTEN_NOT_RUN')

from pathlib import Path
import hashlib,json,ast
root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
spec=json.loads((root/'configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json').read_bytes())
spec['schema']='msvr310-supported-task-state-paired-v1'
spec['previous_config']='configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json'
spec['task_state_rule']='equal_direct_ra_supported_split_adamw_v1'
spec['maximum_expected_additional_disk_bytes']=3221225472
spec['estimated_m0_minutes']=[15,30]
spec['estimated_q1_gpu_hours']=[4,6]
files=['tools/'+name+'.py' for name in ['msvr_task_state_optimizer','msvr_task_state_records','train_msvr_supported_task_state','check_msvr_task_state_math','check_msvr_supported_task_state','verify_msvr_task_state_records','verify_msvr_supported_task_state','run_msvr_supported_task_state']]
files+=['configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json','refine-logs/msvr310_supported_task_state_v1/EXPERIMENT_PLAN.md']
files+=['tools/msvr_supported_gradient_balance.py','tools/verify_msvr_supported_gradient_balance_stats.py']
for name in files:
    if name.endswith('.py'):ast.parse((root/name).read_text())
spec['project_file_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
spec['both_endpoint_objective']='cross_scene_smooth_ap'
spec['both_endpoint_role_input']='direct_rank_plus_direct_auxiliary'
spec['rank_state_support']='warmup_or_eligible_cross_scene_anchor'
spec['rank_state_transition']='preserve_triplet_moments_at_ap_activation'
spec['task_preconditioned_update_reduction']='sum'
spec['role_task_weights']=[1,1]
spec['classification_head_rule']='original_current_total_gradient'
(root/'configs/MSVR310/TriFusion-supported-task-state-paired-v1.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
print('CONFIG_WRITTEN_AST_PASS_NOT_RUN')

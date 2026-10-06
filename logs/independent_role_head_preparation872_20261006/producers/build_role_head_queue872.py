from pathlib import Path
import ast

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
source = (root/'tools/queue_semantic_capacity.py').read_text(encoding='utf-8')
source = source.replace('semantic_capacity', 'independent_role_heads')
source = source.replace('semantic-capacity-control-v1', 'independent-role-heads-v1')
source = source.replace('independent_role_heads_control_v1', 'independent_role_heads_v1')
source = source.replace("variant=='native'", "variant=='semantic'")
source = source.replace("('native',)", "('semantic',)")
source = source.replace("variant='native'", "variant='semantic'")
source = source.replace("'native'", "'semantic'")
source = source.replace("expected_active_reader_parameters=159096,\n        native_reader_parameter_gap=-200,storage_required_bytes=STORAGE_BYTES,", "role_head_policy='independent_raw_fused_heads',storage_required_bytes=STORAGE_BYTES,")
start = source.index("        assert binding['active_reader_parameters']")
end = source.index("        manifest['initialization_sha256']", start)
source = source[:start] + '''        assert binding['objective_gradient_policy']=='global_author_heads_to_global_task_separate_fused_heads_to_role_task'
        assert binding['role_head_initialization']=='deepcopy_original_global_heads_no_rng_consumption'
        assert binding['trainable_parameters']==old['trainable_parameters']+binding['role_head_parameters']
        assert binding['trainable_parameter_tensors']==old['trainable_parameter_tensors']+binding['role_head_tensors']
        for key in ('public_clip_sha256','protocol_sha256','author_source_commit','visual_initial_sha256',
                    'camera_initial_sha256','shared_initializer_sha256','batch_size','num_instances','cfg_yaml',
                    'head_names','role_input_gradient_policy','visual_placement'):
            assert binding[key]==old[key],key
''' + source[end:]
source = source.replace('Three new near-capacity semantic-source controls; reuse sealed raw results.',
                        'Three independent fused-head interventions; reuse sealed raw controls.')
source = source.replace('Three new semantic-capacity controls only. Internal native factory slot binds an explicit semantic source. Raw author objectives and sealed baselines reused. No power/temperature action or parity repair.',
                        'Three fresh semantic evidence runs with independent fused heads. Original raw objectives, role evidence, sampler, recipe and L2_1536 output preserved. Extra trainable head capacity disclosed. No power/temperature action or parity repair.')
target = root/'tools/queue_independent_role_heads.py'
assert not target.exists()
ast.parse(source)
target.write_text(source,encoding='utf-8')

source = (root/'tools/report_semantic_capacity.py').read_text(encoding='utf-8')
source = source.replace('semantic_capacity', 'independent_role_heads')
source = source.replace("dataset,'native'", "dataset,'semantic'")
source = source.replace("for control in ('raw_native','raw_semantic','raw_global_only'):",
                        "for control in ('raw_semantic','raw_global_only'):")
source = source.replace('reader_parameters=159096,native_reader_parameters=159296,parameter_gap=-200,',
                        "role_head_parameters={r['dataset']:r['initializer']['role_head_parameters'] for r in rows},")
source = source.replace('Seed42 exploratory comparison on consumed benchmarks, near capacity not exact capacity. Old raw controls reused with full batch-order equality. New semantic MLP plus resized candidates versus image CNN source; no universal causal, novelty, seed-stability or SOTA claim. No normalized metric-stage rescue.',
                        'Seed42 exploratory training-head ownership comparison on consumed official benchmarks. Old raw semantic/global controls reused with full batch-order equality. Evidence, raw objectives, deployment and recipe unchanged; extra trainable head capacity and independent BN buffers disclosed. Not an original module, proof of exclusive cause, training-seed stability or SOTA. No normalized metric-stage rescue.')
source = source.replace('近似同容量语义来源控制', '独立角色训练头对照')
target = root/'tools/report_independent_role_heads.py'
assert not target.exists()
ast.parse(source)
target.write_text(source,encoding='utf-8')
print('TWO_NEW_EXECUTION_FILES_AST_PASS')

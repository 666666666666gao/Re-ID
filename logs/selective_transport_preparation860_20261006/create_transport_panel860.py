from pathlib import Path
import ast,json,hashlib
from datetime import datetime

root=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
old=(root/'tools/queue_region_reconstruction.py').read_text(encoding='utf-8')
replacements={
 'RAW-semantic endpoints':'RAW-semantic transport endpoints',
 'trifusion-region-evidence-reconstruction-v1':'trifusion-selective-role-transport-v1',
 'QUERY_MODES':'MASS_MODES', "'patch', 'native': 'mean'":"'slot_mass', 'native': 'uniform_mass'",
 'region_evidence_reconstruction_v1':'selective_role_transport_v1',
 'run_region_reconstruction':'run_selective_role_transport',
 'check_region_reconstruction':'check_selective_role_transport',
 'report_region_reconstruction':'report_selective_role_transport',
 'region_reconstruction_prepare_unused':'selective_transport_prepare_unused',
 'active_reconstruction_parameters':'active_transport_parameters',
 'active_reconstruction_tensors':'active_transport_tensors',
 'reconstruction_parameters':'transport_parameters',
 'reconstruction_updates':'transport_updates',
 'reconstruction_query_mode':'transport_mass_mode',
 'query_mode':'mass_mode', 'query_modes':'mass_modes',
 '105232':'32768', "== 15":"== 2", " + 15":" + 2", 'tensors=15':'tensors=2',
 'Six fresh patch/mean-query visual-only attention interventions. Internal variant slots do not denote native images. Original RAW responsibilities, author batch, recipe and bridge unchanged. No power/temperature actions, parity repair or N2/N3.':
 'Six fresh slot_mass/uniform_mass endpoints; both use original semantic input and private 3x16 Mamba. Pair uniform mass retains differentiable same-P total matrix mass, not message energy or OT column constraints. Original RAW duties and author recipe; no native input, reconstruction, new loss, power/temperature or parity repair.'
}
for a,b in replacements.items():old=old.replace(a,b)
old=old.replace('active_transport_tensors=15','active_transport_tensors=2')
assert 'region_reconstruction' not in old and '105232' not in old
q=root/'tools/queue_selective_role_transport.py';assert not q.exists();q.write_text(old,encoding='utf-8')
report=(root/'tools/report_region_reconstruction.py').read_text(encoding='utf-8')
for a,b in {
 'patch/mean':'slot/pair-uniform mass', 'queue_region_reconstruction':'queue_selective_role_transport',
 "control == 'mean'":"control == 'uniform_mass'", 'QUERY_MODES':'MASS_MODES',
 'query_mode':'mass_mode', 'query_mode=mode':'mass_mode=mode',
 "('mean', 'raw_semantic', 'raw_global_only') if mode == 'patch'":"('uniform_mass', 'raw_semantic', 'raw_global_only') if mode == 'slot_mass'",
 "r['control'] == 'mean'":"r['control'] == 'uniform_mass'",
 'primary_patch_vs_mean_progress_count':'primary_slot_vs_uniform_progress_count',
 'active_reconstruction_parameters=105232, active_reconstruction_tensors=15':'active_transport_parameters=32768, active_transport_tensors=2',
 'Same active parameters and initial state; query routing differs.':'Same active parameters and initial state; receiving-slot mass allocation differs. Same-P matrix mass matches, not vector norm or separately trained budgets.',
 'Single seed42 on consumed official benchmarks. Same active parameters and original RAW-semantic pipeline; patch-specific routing versus broadcast mean context. No text resources or full SAGA reproduction; no seed stability, true-part correspondence or SOTA claim.':
 'Single seed42 on consumed official benchmarks. Original RAW duties; slot-specific versus pair-uniform real mass. Both change old48 Mamba to private3x16. Same-P total matrix mass matches, not message energy or trained budget. No correspondence truth, text, new loss, seed stability or SOTA claim.',
 '# 区域候选重建：patch查询与均值广播对照':'# 跨光谱槽位传输：局部质量与模态对统一质量',
 '| 查询 |':'| 质量分配 |',
}.items():report=report.replace(a,b)
assert 'region_reconstruction' not in report and '105232' not in report
p=root/'tools/report_selective_role_transport.py';assert not p.exists();p.write_text(report,encoding='utf-8')
for name in ('tools/run_selective_role_transport.py','tools/check_selective_role_transport.py',
             'tools/queue_selective_role_transport.py','tools/report_selective_role_transport.py',
             'modeling/trifusion/selective_role_transport.py'):
 ast.parse((root/name).read_text(encoding='utf-8'))

folder=root/'refine-logs/selective_role_transport_v1'
review=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selective_transport_method_review860')
trace=Path('C:/Users/gb/.aris/traces/research-refine/2026-10-05_run860')
for name in ('REVIEW_ROUND1.json','round-1-review.md','REVIEW_ROUND2.json','round-2-review.md'):
 (folder/name).write_bytes((review/name).read_bytes())
 (trace/name).write_bytes((review/name).read_bytes())
(trace/'round-2-revised-proposal.md').write_bytes((folder/'round-2-revised-proposal.md').read_bytes())
request='完成最后第2轮（本次max_rounds=2，不新增轮次）。只读完整修订稿 C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/selective_role_transport_v1/round-2-revised-proposal.md，SHA 11d0f805430a56e27c9607a7c0a0a421e5ef0e8c744e122b6a669106abd2cad7；同文件夹PROBLEM_ANCHOR正文保持。请按同七维权重独立复核，不因修改完成而自动给9。明确区分paper READY与窄机制科学对照是否可解释/可实施，不把源码不存在写成已通过。核心修订：slot_mass=P，uniform_mass=mean(rowmass)Q，same P总矩阵质量相同不代表消息能量相同/独立训练后质量相同；两矩阵32768参数；fixednull只是简化；同模态3×16与旧RAW48不等价。数学组件/M0/旧源/存储门写在完整稿。请指出任何仍属阻塞的控制/数学/协议错误，若无则明确标注无此类阻塞、但论文品质仍可REVISE。CALIBRATION:none，same-family/provisional，actual_backend未独立核实。输出只新增 C:/Users/gb/.codex_tmp/independent_evidence_draft/selective_transport_method_review860/REVIEW_ROUND2.json 和 round-2-review.md，包含完整verbatim review、维度分、总分、gap/方法所解决问题/漂移、实施前必须解决项、输入SHA。不要SSH/NN/实现/删除/修改原稿或R1。完成后结束并报告路径。'
(trace/'ROUND2_REQUEST_VERBATIM.txt').write_text(request,encoding='utf-8')
r=json.loads((review/'REVIEW_ROUND2.json').read_text(encoding='utf-8'))
(folder/'REFINE_STATE.json').write_text(json.dumps(dict(phase='review_complete',round=2,
 agent_id='/root/review_selective_transport860',last_score=r['weighted_composite'],last_verdict=r['verdict'],
 status='bounded_review_complete_paper_not_ready_design_implementable',timestamp=datetime.now().astimezone().isoformat(),
 max_rounds=2,review_independence='same-family',acceptance_status='provisional'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('FIVE_NEW_SOURCES_AST_PASS; TWO_ROUND_REVIEW_ARCHIVED')

from pathlib import Path
import json,hashlib,subprocess,statistics
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp');paths=[]
for src,dest in [('trifusion_supported_balance_q1_complete_20260921','supported_gradient_balance_q1_complete_20260921'),('trifusion_supported_balance_q1_gradient_analysis_20260921','supported_gradient_balance_q1_gradient_analysis_20260921'),('trifusion_supported_balance_q1_training_tables_20260921','supported_gradient_balance_q1_training_tables_20260921'),('trifusion_supported_balance_q1_ranking_analysis_20260921','supported_gradient_balance_q1_ranking_analysis_20260921'),('trifusion_supported_balance_cpu_equations_audit_20260921','supported_gradient_balance_cpu_equations_audit_20260921')]:
    for f in sorted((t/src).rglob('*')):
        if not f.is_file():continue
        assert f.suffix in ('.json','.jsonl','.csv','.log','.md','.py','.stderr','.txt')
        data=f.read_bytes();data.decode('utf-8');rel='evidence/'+dest+'/'+f.relative_to(t/src).as_posix();target=p/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()==data:continue
        assert not target.exists(),rel
        target.write_bytes(data);paths.append(rel)
folder='evidence/supported_gradient_balance_q1_analysis_20260921';(p/folder).mkdir()
names=['trifusion_supported_balance_q1_source_log_analysis_20260921.json','prepare_supported_balance_repaired_intake_20260921.py','intake_trifusion_supported_balance_repaired_q1_20260921.py','prepare_supported_balance_repaired_analysis_20260921.py','analyze_supported_gradient_balance_q1_20260921_repaired.py','analyze_trifusion_supported_balance_q1_gradients_20260921_repaired.py','export_supported_balance_training_tables_20260921_repaired.py','audit_msvr_paired_ranking_text_repaired_20260921.py','launch_supported_balance_cpu_arithmetic_recheck_20260921.py','trifusion_supported_balance_cpu_arithmetic_recheck_launch_20260921.json','trifusion_supported_balance_cpu_arithmetic_recheck_observation_1_20260921.json']
for name in names:
    rel=folder+'/'+name;(p/rel).write_bytes((t/name).read_bytes());paths.append(rel)
r=json.loads((t/'trifusion_supported_balance_q1_ranking_analysis_20260921/ranking_replay.json').read_bytes());s=json.loads((t/'trifusion_supported_balance_q1_source_log_analysis_20260921.json').read_bytes());g=json.loads((t/'trifusion_supported_balance_q1_gradient_analysis_20260921/summary.json').read_bytes())
lines=['# MSVR310 supported gradient balance R2 — complete Q1','', 'Status: Q1_FAIL; full repaired CPU and executor text replay complete; fresh Q1 integrity audit IN_PROGRESS. Seed42, three identity folds, six fixed 20-epoch endpoints. Internal Q1 only; official table unchanged. Execution 1381639f778f77f124a2726ee55c07610092a438.','', '| Output | Control mAP | Balanced mAP | Delta | Control R1 | Balanced R1 |','|---|---:|---:|---:|---:|---:|']
for key in ['baseline_only','fused','cnn','transformer','mamba']:
    a=r['endpoints']['control']['metrics'][key];b=r['endpoints']['balanced']['metrics'][key]
    lines.append(f"| {key} | {a['mAP']:.6f} | {b['mAP']:.6f} | {r['paired_gains'][key]:+.6f} | {a['Rank-1']:.6f} | {b['Rank-1']:.6f} |")
lines+=['',f"Paired fold fused gains: {r['paired_fold_gains']}. Identity-bootstrap lower bound: {r['paired_lower_bound']:.9f}. Paired gates {sum(r['paired_gates'].values())}/5; candidate vs Signal gates {sum(r['endpoints']['balanced']['gates'].values())}/5. Gates unchanged.",'', '600 legal queries, 60 identities, 2,069,520 rank positions, all five outputs. Fused query changes: '+json.dumps(r['paired_changes']['fused'])+'.','', '## Source optimization and actual intervention','', '| Endpoint | Last65 batch hard | Last65 extended hard | Last65 standard AP | Last65 cross-scene AP |','|---|---:|---:|---:|---:|']
for end in ['control','balanced']:
    vals=[x['phases']['last_five_epochs_steps_196_260']['mean_losses'] for x in s['endpoints'] if x['endpoint']==end]
    means=[statistics.mean(x[k] for x in vals) for k in ['batch_hard','expanded_hard','smooth_ap','cross_scene_ap']]
    lines.append('| '+end+' | '+' | '.join(f'{x:.8f}' for x in means)+' |')
lines+=['', 'Registered cross-scene objective improved in each fold late in training, while both hard-margin diagnostics worsened in each fold. This is descriptive evidence, not proof of an AdamW cause. Balanced supported-role median ranking coefficients range '+f"{min(x['applied_rank_weight']['median'] for x in g['role_summaries'] if 'balanced' in x['endpoint']):.6f} to {max(x['applied_rank_weight']['median'] for x in g['role_summaries'] if 'balanced' in x['endpoint']):.6f}." ,'', 'Paired source records/pixels match, but warmup numeric trajectories already differ from step2: maximum total-loss differences per fold '+str([x['warmup']['max_abs_total_loss_difference'] for x in s['paired_warmup']])+'. Do not interpret the tiny endpoint delta as a bitwise-isolated coefficient effect. Supported balanced updates also retain the disclosed separate-backward versus combined-backward numerical difference.','', '## Evidence and limitations','', '- All 1560 step records, 120 epoch records and 4680 role-step records retained. Epoch tables contain real training objectives, not invented heldout epoch scores.','- Q1 current-rank and direct-aux backward each run1560 times across both arms; Q1 direct-reference vector checks0. M0 reference checks are separate engineering evidence. Saved norm/cosine witnesses cannot reconstruct full parameter gradients or task-specific AdamW contributions.','- Original CPU stopped on sqrt exactness; first repair stopped on execution binding; bound repair stopped on weighted norm using double coefficients. All failures remain. Fresh same-family/provisional arithmetic audit supports only sqrt plus FP32 coefficient formula fixes, with all thresholds unchanged. Original execution files/config/checkpoints unchanged.','- Full final CPU receipt q1_cpu_arithmetic_recheck/verification.json: 1560 steps,116501504 memory distances,581120 historical-VJP record forwards,2069520 retrieval/rank elements,0 model forwards,0 updates. Original pipeline remains STOPPED_AT_Q1_CPU, not rewritten.','- Final verification rechecks stored arrays/checkpoint hashes; it does not independently regenerate every training gradient. AdamW moment/scaler states were not saved, so exact original task-state attribution is unavailable.','- Full Q1 independent audit pending; do not confuse arithmetic review with full scientific closure. No next experiment launched, no gate changes, no official-test access.','', '## Location','', 'Raw evidence: evidence/supported_gradient_balance_q1_complete_20260921. Training tables, gradient summary and all-query/all-identity ranking tables have adjacent dedicated evidence directories. Arithmetic audit: evidence/supported_gradient_balance_cpu_equations_audit_20260921.','']
report='results/MSVR310_SUPPORTED_GRADIENT_BALANCE_R2_Q1_2026-09-21.md';(p/report).write_text('\n'.join(lines),encoding='utf-8');paths.append(report)
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md';b=(p/tracker).read_bytes().replace(b'R2_Q1_RUNNING',b'R2_Q1_FAIL / FULL_AUDIT_IN_PROGRESS')
old='| R2 Q1 | RUNNING | 六端训练完整；独立sqrt复核通过执行绑定，现加权范数恒等式1/95343标量比较超限；全量诊断归档、独立审查中，未宣告CPU通过 |'
new='| R2 Q1 | Q1_FAIL | 六端/1560步齐全；fused53.399384→53.452649，+0.053266；配对1/5，Signal1/5 |'
assert old.encode() in b;b=b.replace(old.encode(),new.encode()).replace('| R2 Q1_CPU/Audit | NOT_STARTED | 六端齐全后核验；无中间选模或调参 |'.encode(),'| R2 Q1_CPU/Audit | CPU_PASS / AUDIT_IN_PROGRESS | 10:51事后算术修正全量核验通过，原失败保留；新鲜Q1独立审查进行中 |'.encode());(p/tracker).write_bytes(b);paths.append(tracker)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
b+='''

### §41.238 R2完整Q1未晋级，CPU核验已闭环，独立审查进行中

原训练六端10:29完成。10:51:19事后算术复核退出0，完整1560步/116501504记忆距离/2069520检索排名通过。原失败pipeline和各次失败日志未改写。43份文本94652953B逐份SHA接收，模型/数组/原图均未下载。全部120epoch、1560步及4680角色记录已归档。

MSVR310内部Q1：fused53.39938365→53.45264937，+0.05326572；R1 62.166667→62.333333。CNN+0.01821327、Transformer−0.14394053、Mamba+0.12282911。三折fused+0.12996671/−0.10721385/+0.14677428，身份bootstrap下界−0.12614131。配对1/5、相对Signal1/5，Q1_FAIL不变。候选比Signal高0.32326881但三角色均低于Signal；fused修复3条Rank1、新增2条。正式成绩不变。

末65步跨场景AP目标各折下降，但批内与扩展hard目标各折上升。系数确实改变，候选有支持角色的排名系数中位数1.074397—1.156140；未形成所需稳定未知身份收益。配对记录/像素一致，但warmup第2步已有数值差异，最大总loss差约0.00112/0.00140/0.00171；不将+0.0533解释为精确隔离的系数因果贡献。AdamW历史状态未保存，不能反推任务更新份额。

结果报告results/MSVR310_SUPPORTED_GRADIENT_BALANCE_R2_Q1_2026-09-21.md。原始、梯度、epoch、排名证据分别归档evidence/supported_gradient_balance_q1_*_20260921。新鲜gpt-6-astra/max全Q1独立审查audit_supported_gradient_balance_q1_20260921进行中；算术审查已WARN/same-family/provisional，不能替代终态完整审查。暂不启动后继，Goal ACTIVE/UNMET。
'''.encode();(p/master).write_bytes(b);paths.append(master)
agents='AGENTS.md';b=(p/agents).read_bytes();old='10:29六端训练完整结束，原CPU因sqrt与幂运算1ULP精确比较差异停止；仅修核验器，待完整重核和独立审查，不重训';new='R2六端完整Q1_FAIL，fused+0.053266、配对1/5；10:51事后CPU全量通过、原失败保留；完整独立审查中，见§41.238，不启动后继'
assert old.encode() in b;(p/agents).write_bytes(b.replace(old.encode(),new.encode()));paths.append(agents)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_q1_terminal_publication_20260921.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(paths),master_sha256=m['new_master_sha256'])))

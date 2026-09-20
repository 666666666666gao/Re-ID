from pathlib import Path
import json,datetime,hashlib,subprocess
out=Path(__file__).parent
repo=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
trace=repo/'.aris/traces/experiment-audit/2026-09-20_smooth_ap_source_coverage'
final='''独立审计已完成。结论为 **WARN / CLOSED_WITH_LIMITS**；全量确定性核验 **PASS**，工程结论 **PASS_WITH_LIMITS**，审计归属 **same-family / provisional**。没有发现需要更正实验代码、掩码、分母或报告数值的问题。

- [完整审计报告](C:/Users/gb/.codex_tmp/smooth_ap_source_coverage_independent_audit_20260920/EXPERIMENT_AUDIT.md)
- [机器可读审计 JSON](C:/Users/gb/.codex_tmp/smooth_ap_source_coverage_independent_audit_20260920/EXPERIMENT_AUDIT.json)
- [独立核验摘要与终态](C:/Users/gb/.codex_tmp/smooth_ap_source_coverage_independent_audit_20260920/independent_verification_summary.json)
- [完整私有审计 trace](C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-20_smooth_ap_source_coverage/001-review.response.md)

R2 原 wrapper **1924**、CPU 子进程 **1925** 于北京时间 **18:13:44** 退出 0，用时 **706.165 秒**，stderr 为空。实际核验范围包括全部 **12 条件、60 个特征数组、82,560 条完整 query、1,996,800 条候选/完整配对行、120 条条件汇总、40 条聚合和 20 条配对结果**。AP 最大重算误差 **0**，直接差平方和与登记距离的最大误差 **2.6645e-15**。实际数组未出现非 self 精确距离并列；显式并列语义夹具也通过。

1032 个训练记录的 GT、来源分区、六端各 260 步候选池、self/去重/cross-scene 掩码、全部干扰记录、共同合法集合及两种加权口径均核对通过。159 份文本、99 项递归配置绑定、17 份 Signal 源码和六个终态 checkpoint 的哈希一致。报告中的增强 cross-scene 来源 fused **95.444495 → 96.672922（+1.228427 pp）**及 **251 改善 / 70 下降**得到支持。

WARN 保留具体证据边界：本次从保存数组完整重算，没有重放源图像到模型特征的生成过程；历史像素配对、状态不变和无梯度检查依赖一致的源码与原运行回执。固定来源视图和唯一记录候选池不等于动态训练过程，也不能替代身份外泛化、官方结果或目标参数梯度测量。原 Q1_FAIL 保持。

R1 审计脚本曾因 NumPy int64 摘要序列化退出 1，尚未进入数组核验；失败源码、完整 traceback 和终态均已保留。R2 仅把三个摘要计数转为 Python int，随后完整核验一次。报告及 trace 使用实际 canonical agent id，未虚构 UUID 或后端身份。

后续只需将调用方的“待独立审计”状态链接到本次 **CLOSED_WITH_LIMITS** 报告。本次未编辑 master、commit/push、重做旧 Q1/M0，也未运行模型前向、训练、官方图像读取或权重删除。

<oai-mem-citation>
<citation_entries>
MEMORY.md:369-382|note=[Preserve fixed source diagnostic scope and inspect existing receipts before continuation]
</citation_entries>
<rollout_ids>
01a06f22-cc0d-7da2-8cae-68a700778767
</rollout_ids>
</oai-mem-citation>'''
(out/'final_response.md').write_text(final,encoding='utf-8')
(trace/'001-review.response.md').write_text(final,encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc)
run=json.loads((trace/'run.meta.json').read_bytes())
started=datetime.datetime.fromisoformat(run['started_at'])
meta=dict(call_number=1,purpose='review',timestamp=now.isoformat(),agent_id='/root/audit_smooth_coverage_resume_20260920',agent_uuid=None,model='gpt-6-astra',model_requested='gpt-6-astra',reasoning_effort='max',fork_turns='none',reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',backend_identity_independently_verified=False,duration_ms=round((now-started).total_seconds()*1000),status='ok',verdict='WARN',closure_status='CLOSED_WITH_LIMITS',deterministic_checks_status='pass',full_report=str(out/'EXPERIMENT_AUDIT.md'),json_report=str(out/'EXPERIMENT_AUDIT.json'))
(trace/'001-review.meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
event=dict(event='review_trace',skill='experiment-audit',purpose='review',agent_id=meta['agent_id'],trace_path=str(trace),status='ok',verdict='WARN',note='Event kept in private audit trace; unrelated pre-existing .aris/meta/events.jsonl mutation preserved.')
(trace/'events.jsonl').write_text(json.dumps(event)+'\n',encoding='utf-8')
for name in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json'):
    (trace/name).write_bytes((out/name).read_bytes())
status=subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True)
(out/'final_git_status.txt').write_text(status,encoding='utf-8')
git_status_unchanged=status==(out/'initial_git_status.txt').read_text(encoding='utf-8')
assert (out/'final_response.md').read_bytes()==(trace/'001-review.response.md').read_bytes()
receipt=dict(status='COMPLETE_REVIEW_ARTIFACTS',generated_at=now.isoformat(),agent_id=meta['agent_id'],verdict='WARN',closure_status='CLOSED_WITH_LIMITS',deterministic_checks_status='pass',git_status_unchanged=git_status_unchanged,git_status_note='Shared workspace observations are preserved; concurrent changes outside this audit were not touched.',
    files={name:dict(bytes=(out/name).stat().st_size,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in ('EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','final_response.md','independent_verification_summary.json')},
    trace_files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(trace.iterdir()) if p.is_file()})
(out/'AUDIT_COMPLETION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))

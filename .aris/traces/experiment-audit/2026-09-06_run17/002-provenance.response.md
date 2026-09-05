已完成限定跟进，只改写了这两个原报告路径：

- `C:/Users/gb/.trifusion_github_publish_22c3bee/EXPERIMENT_AUDIT_MSVR310_SIGNAL_B0.md`
- `C:/Users/gb/.trifusion_github_publish_22c3bee/EXPERIMENT_AUDIT_MSVR310_SIGNAL_B0.json`

未修改实验输入、执行器报告、评分规则、replay 脚本或 replay 输出；未导入模型/张量/图像库，未使用远端命令、凭据或委派。

最终报告 SHA-256：

- Markdown: `aa04e91403b24d2c4bf4beb24c01411103353762de58e0f009e6fdca7c2e3c49`
- JSON: `34c00b4f8b4add6e5abff55cf9ee8fd4e89b2e491f8aa65224623c2eb9d23a23`

最终 verdict 保持不变：

- overall verdict: `WARN`
- integrity_status: `WARN`
- engineering conclusion: `PASS：B0 工程与结果封存完成，三折 source-only 模型各 50 epoch / 650 step，总 1950 step，固定 epoch50 后做一次完整 held-out gallery 检索。`
- evaluation type: `real_gt / train_internal_identity_oof_baseline`
- A-F 状态保持不变：
  - A Ground Truth Provenance: `PASS`
  - B Metric Denominator And Normalization: `PASS`
  - C Result Existence / Number / Status: `PASS`
  - D Metric Call Path / Output Retention: `PASS`
  - E Scope / Seed / Protocol Limitations: `WARN`
  - F Evaluation Type Classification: `PASS`

本次重新核验了原始 input manifest：

- `input_count=71`
- `all_ok=true`
- `mismatch_count=0`
- manifest `created_at=2026-09-06T05:19:29.510965+08:00`

本次没有重复 AP/Rank 算术。理由是本次 follow-up 是 metadata-only：只修正 reviewer dispatch/provenance 与 binary-evidence Action Items，未改变输入、评分规则或执行侧报告。原独立算术仍由先前报告中的 stdlib replay 支持，原命令、结果和 timing 保持不变。

已解决的两处表述问题：

1. reviewer dispatch/provenance

   原报告把“root 是否派发当前 reviewer”和“当前 reviewer 是否继续委派”混在一起。已改为：

   - root 确实通过 `spawn_agent` 请求并派发当前 reviewer：
     - `001-complete-baseline.request.json:5-10` 记录 `tool=spawn_agent`、`task_name=audit_msvr_signal_b0`、`model=gpt-5.5`、`reasoning_effort=xhigh`、`fork_turns=none`
     - `dispatch_observation.json:3-9` 记录 `tool=collaboration.spawn_agent`、`requested_model=gpt-5.5`、`requested_reasoning_effort=xhigh`、`fork_turns=none`、`returned_task_name=/root/audit_msvr_signal_b0`、`status=accepted_and_running`
     - `001-complete-baseline.meta.json:5-8,20-21` 记录当前 `agent_id=/root/audit_msvr_signal_b0`、`status=ok`、`input_files_rechecked=71`
   - 当前 reviewer 没有再委派下级 reviewer。
   - `requested_model/requested_reasoning_effort/fork_turns/status` 只支持“请求并被接受的配置记录”，不构成独立后端身份认证。
   - `resolved_backend_independently_attested=false` 明确保留在 dispatch observation 与 meta 中。
   - review-family scope 已限定为 Codex/GPT-family `Type-A` 审计；不声称 `Type-B cross-family acquittal`。
   - no-delegation 是执行边界，不作为模型可用性或不可用性的证据。

2. binary-evidence Action Items

   原 action item 中“把 `.pth`、`.pt`、CLIP 权重纳入本地 manifest”的写法没有充分限定原计划边界。已改为：

   - 在本 run17 合同/计划边界内，二进制证据只能表述为远端 artifact 收据与本地 JSON/SHA 可追溯。
   - `EXPERIMENT_PLAN.md:67` 明确“特征、距离、checkpoint 留在远端，JSON 逐 query 数值及文件 SHA 可归档本地”。
   - `EXPERIMENT_PLAN.md:87` 明确本地只做 AST/文本检查，不 import 模型、张量、图片或执行检索。
   - 不得追改 run17 input manifest，也不得把当前审计改称二进制直接复核。
   - 如果未来需要 `.pth`、`.pt`、CLIP 权重或图像像素的直接二进制复核，应作为新的授权 provenance contract 和新 manifest，而不是补丁式改变本次审计边界。

最终结论：`WARN` 不变。B0 内部基线的工程完成、标签来源、排序算术和结果落盘仍可信；官方复现、多 seed 稳健性、方法资格和二进制/图像直接复核仍不在本证据支持范围内。

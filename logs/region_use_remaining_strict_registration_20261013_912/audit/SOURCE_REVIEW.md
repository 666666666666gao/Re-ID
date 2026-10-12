**Bounded source review — remaining strict evaluation 912**

**Verdict: source checks PASS; overall WARN because runtime is unverified.** No concrete implementation blocker and no required source correction were found.

Reviewed controller SHA-256: `2709b390c42fc37a74678e185b98cf4ff84c15f490964c61c7fc0225246d83aa`.
Reviewed plan SHA-256: `be82ba56fd6e0191a8230d1e9ea2102e5120f8a22a5ce203cf84a59fd31fb0a1`.
Seal addendum SHA-256: `7210728e8c32e575396f97af72cacf43bab70cd35277155ee5e87eb4c2345df7`.

Native Codex reviewer `/root/audit_strict_completion912`; requested `gpt-6-astra`, effort `max`. Actual service model and effort: **UNATTESTED**. Review independence: **same-family**; acceptance: **provisional**. Date: 2026-10-13. This is a fresh source review, not execution or acceptance of a strict result.

**Findings**

1. **Recorded original stop is internally consistent.** `TERMINAL_STORAGE_912.json:27` records only prepare/M0/train, each COMPLETE with exit 0. Lines 613–618 record the before-evaluate storage stop and full50 completion; lines 714–716 record both strict artifacts absent. `FINAL_NODE_912.json:735`, `:745`, `:851` and `:874` agree on M0/full50, absent controller/supervisor and exit 1. The two full state objects and terminal exit objects are identical. Original `continue_region_use_remaining.py:47–54` checks storage before constructing/appending the next stage. These are checked local observation-record bytes; the full remote history, steps, order, checkpoint and original file bytes were not independently accessed.

2. **Exactly one original evaluate call.** `finish_region_use_evaluation.py:54–56` calls the existing producer once with `panel.command(original, 'RGBNT100', 'ambiguity', 'evaluate')`. The initializer, protocol, Signal source, CLIP and full output paths stay original (`queue_region_use_probe.py:33–40`). Producer seed42/full50 remain fixed (`run_region_use_probe.py:101–122`). Its inherited `run_foundation_recipe.py:287–312` strictly loads the existing `best_map.pth`, checks it against the original selected history, and uses the existing strict metric tolerance. Computing the historical maximum there verifies the saved selection; no replacement checkpoint is selected or saved. No prepare/M0/train replay, optimizer step, precision/tolerance/gain/seed change or new diagnostic is introduced.

3. **Administrative reuse preserves the original failed record.** The old campaign and EXIT are read, copied, and recorded as original path/exit1; only the new campaign is written (`finish_region_use_evaluation.py:24–55`). The same basename produces the same original trained folder (`queue_region_use_probe.py:29–30`). The existing CPU reporter requires four successful stages, reads that original folder, compares the parsed receipt with `job.result`, checks checkpoint/distance hashes and full training/order, and writes its own report outputs (`report_region_use_endpoint.py:27–58,94–106`). It does not rewrite the strict files. No output-directory/reporter override or fallback is needed. Line 51 retains the original controllers' existing source-map wiring. The stub preserves the original three stage objects and protected fixture bytes. Real strict bytes do not yet exist in the supplied record, so actual reporter reuse is not established.

4. **Source/seal interface and inherited admission are connected.** Entry checks the controller SHA, panel source map and all supplied artifact SHA values; it checks original stop, completed stages/full50, absent strict files and 2,684,354,560 bytes free (`finish_region_use_evaluation.py:19–40`). It records the new seal SHA and rechecks immutable artifacts after the child. `queue_native_research.run_logged:62–78` actually calls `queue_foundation_recipe.require_sources:110–115`, checks the 2 GiB reserve, then queries only GPU0/1 memory via the existing admission routine (both below 500 MiB; existing wait 240 seconds) and sets `CUDA_VISIBLE_DEVICES=0,1`. The new seal's full 463-entry source map matches the old seal and both captures; all old50 artifact pins are retained, with exactly nine remaining911 stage/full inputs plus two accepted M0 text inputs. Controller and original-seal digests agree. The new seal's `original_source_seal_sha256` is documentary in this controller; entry checks that seal's source map, while the unchanged reporter later checks the old seal digest. There is no present mismatch warranting an extra branch. These are local interface/source checks, not current host/device/disk/remote-byte admission.

5. **Child and validation failures retain distinct, honest states.** The inherited runner stores the exact `process.wait()` code and clears `active_command` (`queue_native_research.py:72–78`). The controller returns the same nonzero code with FAILED state. After child exit0 it writes `STRICT_EXIT0_VALIDATION_PENDING` before receipt/hash assertions (`finish_region_use_evaluation.py:57–69`). A post-child assertion therefore retains child exit0 and a nonaccepted pending state; it does not label the job COMPLETE or leave its campaign RUNNING. The in-memory cases passed for success, child exit17 and a wrong-condition assertion. A real new supervisor EXIT receipt has not been observed.

6. **Protected dependencies and stopping scope remain intact in source.** The continuation contains no deletion, checkpoint save, restoration, training or extra diagnostic launch. It rechecks the original target-best pin. `CLEANUP_912.json:18–43` records one historical H2a100 remote duplicate retired after archive, zero current-best deletions, old50 dependencies and fourteen extra hashes for six current bests/eight Signal bests. The target-best digest matches the terminal capture and seal. Recorded free space after cleanup is 3,019,280,384 bytes, not proof of current free space. The historical local archive and remote protected binaries were not rehashed here. The retired historical path requires explicit restoration before future use; this continuation does not use it.

7. **No source correction is required by the observed code/state.** Publish/deploy, current runtime admission, the single producer's real exit, strict artifact validation, and the later CPU report are still separate unperformed steps. No fallback, extra try/except, tolerance relaxation, new candidate, new training or NN diagnosis is requested by this review.

**Bounded checks performed**

Thirteen Python files were AST-parsed without importing production modules. Ten relevant inherited-source hashes match the captured 463-entry map; the entire source463 implementation was not re-audited. Eight in-memory cases passed: successful completion, child exit17, post-child validation assertion, controller/source/artifact hash mismatch, insufficient disk, and existing busy-device wait. The simulated successful call exactly matches the original train command with only its mode changed to evaluate. Their fixtures contain synthetic initializer/checkpoint/training data and cannot prove numerical or runtime behavior.

The new seal was compared with the original full JSON maps and recorded M0/initializer/best/order pins: **463 source pins; 50 + 9 + 2 = 61 artifact pins**. No remote bytes were checked.

Check details: [BOUNDED_CHECKS.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/remaining_strict912/BOUNDED_CHECKS.json), [SEAL_INTERFACE_CHECKS.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/remaining_strict912/SEAL_INTERFACE_CHECKS.json), [CHECK_EXECUTION_TRACE.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/remaining_strict912/CHECK_EXECUTION_TRACE.json). The initial PATH-first Python launch failed before executing checks with `No pyvenv.cfg file`; it is retained in the execution trace. An explicitly discovered existing `E:/python.exe` executed the stdlib checks successfully. No installation occurred.

**Runtime and claim limits**

No SSH, real GPU query, NN run, production import, source mutation, shared-memory mutation or additional reviewer was used. The audit has not established actual deployment, fresh runtime admission, strict success, endpoint report acceptance or all-six acceptance. Existing five results were not re-audited. It makes no performance forecast or stronger-teacher/compression causal claim.

**Exact input SHA-256 values**

Paths below are relative to `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903`. Additional inherited files were read only to trace `require_sources` and the evaluator/configuration path. The old seal and new seal are the read-only addendum inputs.

| File | SHA-256 |
|---|---|
| `remaining_strict912/tools/finish_region_use_evaluation.py` | `2709b390c42fc37a74678e185b98cf4ff84c15f490964c61c7fc0225246d83aa` |
| `remaining_strict912/EXPERIMENT_PLAN.md` | `be82ba56fd6e0191a8230d1e9ea2102e5120f8a22a5ce203cf84a59fd31fb0a1` |
| `TERMINAL_STORAGE_912.json` | `db67c81069d91ab1952a51b822dd3076f754cf283b1b23057f0ed97384d8cef7` |
| `FINAL_NODE_912.json` | `f1427071ca49903517c04dfd6dbca0ee2a437dc7b9e21b99877e2f92415695b7` |
| `CLEANUP_912.json` | `278684423998ff40bc2766277499e3d54de0f3ae31c0ab72459a380002288755` |
| `remaining911/tools/continue_region_use_remaining.py` | `190d6a6dfaa02b3b68c7fec4cd9668c650bfde2c9b00f04759cbd68809435381` |
| `sealed_source_snapshot911/tools/queue_region_use_probe.py` | `1cdbeb294ce6d1686ffe0dcfb3def1ac11d4fb4aac2e988638199544353dfdba` |
| `sealed_source_snapshot911/tools/queue_native_research.py` | `b138144780a31dc8a5a4922cc4e15dbc0537d0f75ce47d823434d3d3a3c5bc65` |
| `sealed_source_snapshot911/tools/run_region_use_probe.py` | `176e1e88a74ede5447f57cf0e3a89c1a0847e729647d37b11fdb6c76ea715591` |
| `review/remaining911/evidence_snapshot/production/tools/report_region_use_endpoint.py` | `807746c6ab295fc39de40bd3a91028c57dc7d346f89e1351d803d5ac401a314d` |
| `sealed_source_snapshot911/tools/queue_foundation_recipe.py` | `3afe051358aa25d99cb3900b4c6fdb13d5d79d9dcee76d01018fccd27e27d302` |
| `sealed_source_snapshot911/tools/run_global_task_role.py` | `151a3350a9da7d86ce6b4b7d5160537b182792ab08727ccb5e4ab0c64bb56d5f` |
| `sealed_source_snapshot911/tools/run_role_input_detach.py` | `a177fd5aa4d5ba25422a49cb619062001722d6c7640891cb3da9645a8eb32fe5` |
| `sealed_source_snapshot911/tools/run_native_research.py` | `b48725622371fc6d9671c464ce97252e2e06eba465e213ce221d84c798e52cc2` |
| `sealed_source_snapshot911/tools/run_native_partitioned.py` | `2c3abac37b122760895b97ccf01d4fefe2cfb8223b35b6e2e335a1a4144f72fa` |
| `sealed_source_snapshot911/tools/run_independent_native_evidence.py` | `2429af8d2f803d70b9d15113a430609ead28621eb4341a746c3025cc5685fe23` |
| `sealed_source_snapshot911/tools/run_foundation_recipe.py` | `c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758` |
| `remaining_strict912/INPUT_SEAL.json` | `7210728e8c32e575396f97af72cacf43bab70cd35277155ee5e87eb4c2345df7` |
| `review/remaining911/evidence_snapshot/production/refine-logs/region_use_supervision_v1/INPUT_SEAL.json` | `d9c56b40c377d658fcc879a8db538a0c585e54219e83c7fd6755411dbaeff7ed` |

Private trace: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/remaining_strict912/.aris/traces/experiment-audit/2026-10-13_run01`. Root stores the native reviewer return verbatim and completes that trace.

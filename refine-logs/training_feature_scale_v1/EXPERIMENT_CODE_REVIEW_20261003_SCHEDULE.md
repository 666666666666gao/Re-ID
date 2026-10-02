# F2 resource scheduling source review

**PASS.** No blocking or non-blocking code issues found in the minimal scheduling change or prepared launcher745.

Scope: **SOURCE_ONLY**. Review independence: **same-family**. Acceptance: **provisional**. Fresh native reviewer `/root/review_feature_scale_scheduling745`, configured as `gpt-6-astra`, max reasoning, `fork_turns=none`; spawn configuration was confirmed by the parent, not independently attested. Reviewed 2026-10-03 under experiment-bridge Phase 2.5.

This is not neural execution, result/integrity acceptance, or SOTA acceptance. No SSH, project-module import, model construction, forward, GPU call, optimizer update, scorer or launcher execution was performed. No implementation was edited.

## Reviewed change

The comparison base is `c329aa099b4acd717ecc0f15f8c95a69b0918c4b`. The change adds `released_gpus()` and an F2-owned copy of the existing scheduling loop, checks initial-GPU release, records release paths, and calls that loop. The plan append accurately explains the actual resource problem. The runner, pair helper, reporter, foundation queue, correspondence queue and foundation runner are byte-identical to that base; the inherited sources also match the supplied original cache.

| File | Exact SHA256 |
|---|---|
| `tools/queue_training_feature_scale.py` | `0560bca5fea416cba6621056520ddb4d9dc66f6e9e6c2bef1d841d7728ee78a9` |
| `refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md` | `41e241ee0c42d030598a2923f5a652268808ac72a70c2ac639e5a125e8bd2fbd` |
| `C:/Users/gb/.codex_tmp/deploy_training_feature_scale745.py` | `a2d21d504b3db9eb25e6f83a319fe2f891ec09e995a15f121243cdbeb71b8e75` |

The companion JSON records exact hashes for all checked implementation files, protocols, manifests and principal receipts.

## Why the release condition is correct

I read the actual scheduler sources embedded in `OTHER_SCHEDULERS.json`, including all three known controller commands and their recorded live state at 00:04. Primary jobs run sequentially per card; repeats wait for that slot's successful primary predecessor; full evaluation waits for its final successful repeat and evaluates that slot's jobs sequentially. Each stage stops on failure. The last successful full-evaluation marker therefore means that slot has finished its entire known training/evaluation sequence. A controller may remain alive while other cards finish, without using the released card again.

The final markers in `queue_training_feature_scale.py:24` match those exact schedules:

| Physical GPU | Final run under `runs/three_seed_extension` |
|---|---|
| 0 | `RGBNT100_demo_s44/evaluation_exit.json` |
| 1 | `RGBNT100_ordinary_s44/evaluation_exit.json` |
| 2 | `RGBNT100_dual_s44/evaluation_exit.json` |
| 3 | `MSVR310_dual_s44/evaluation_exit.json` |

`OTHER_PROGRESS.json`, observed at 00:05:58, records all five GPU3 training exits and all five evaluation exits as zero. The final GPU0-2 markers are absent. This supports GPU3 eligibility at that observation; actual launch still rechecks the markers and capacity.

The new loop (`queue_training_feature_scale.py:42`) additionally requires memory below 500 MiB and no active F2 worker on that GPU. The release map only contains physical GPUs0-3, so there is at most one worker per card and four in total. It cannot take an idle interval between the known other project's phases because their final evaluation marker is not yet present. Newly released cards join on later 240-second polls.

## Preserved behavior

Completion validation, FIFO pending work, child commands, disk checks, exclusive logs and failure handling retain the existing loop's behavior. A nonzero worker exit stops new dispatch and drains already active workers before returning failure. The log-name simplification is equivalent for F2's two nonempty variant names.

All six initialization witnesses and three actual paired initial-forward checks precede M0. All six eight-batch M0 workers complete and their receipts are checked before full training begins. Each full run reconstructs the public initialization and a fresh optimizer in its own directory; M0 weights are never carried over. Strict reload/evaluation and six accepted endpoints still precede one CPU report.

The learning contrast remains normalized versus raw training features for both BN/CE and margin0.3 Triplet. Public CLIP, fresh camera/current heads, seed42, B64/K8, AdamW, original learning rates/warmup, six fresh full50 runs and L2 deployment are unchanged. No loss, seed, protocol, gallery, threshold or training-budget change was found.

The `source_map()` AST is unchanged. Its 246 original hashes plus the three explicit sealed243 canonical hashes match all 249 inherited entries in the supplied runtime742 receipt. The three F2 protocols and seven explicit additions still give 259 unique entries. Their non-root metadata/order matches the SHA-verified original cache. The supplied data receipt retains query/gallery counts 836/836, 1715/8575 and 591/1055, including all 103 MSVR gallery-only identities. This review did not revalidate remote files or image bytes.

## Launcher and checks

Launcher745 requires this fresh PASS and an empty blocker list, preserves the port2026 policy, source-map, warm-environment, disk and fresh campaign/report/log checks, and retains process-start confirmation. It selects the first released card with memory below 500 MiB and rejects any compute app on that exact GPU UUID. It passes the selected physical index into the coordinator and records the initial/released indices. Other cards' existing jobs continue. No retry, preemption or environment rebuild was added.

Supplied receipts identify wait744's process tree and its stop at 00:05:16, before the registered 00:17 first capacity check; its state still records zero deployment invocations. The local deploy745 directory was absent during review.

Static AST checks passed for the seven inspected repository Python files and both launchers. The supplemental protocol check first encountered an inherited protocol omitted from the local sparse checkout; that reviewer-helper failure is retained in `SOURCE_CHECK_SPARSE_OMISSION.json` and `source_checks.py`. The separate `source_map_checks.py` completed successfully using the task-specified original cache and verifying its existing manifest hashes. No production correction was needed.

I also read the parent's `SCHEDULE_SIMULATION_V2/RESULT.json`. It matches the reviewed queue digest and records GPU3 first, GPU0 joining only after release, no use of reserved GPUs1/2, six fake completions and failure drain. It was not rerun here. It is supplemental CPU simulation evidence, not M0 or GPU acceptance; the original mock-harness failure remains preserved by the parent.

**Blocking issues: none. Non-blocking issues: none.** Initial-forward parity, M0 success, full training, strict evaluation, report completion and scientific benefit still require their actual executions and subsequent evidence review.

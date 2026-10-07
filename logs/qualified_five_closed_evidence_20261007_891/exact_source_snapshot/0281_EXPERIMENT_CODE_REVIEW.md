# Experiment code review: checkpointed native evidence v2

**Verdict: PASS — SOURCE_ONLY. Blocking findings: none.**

Fresh native Codex context; requested `gpt-6-astra`, reasoning effort `max`, `fork_turns=none`. Actual backend identity is not independently verified. Review independence: **same-family**. Acceptance: **provisional**.

The new per-modality checkpoint implementation is a justified computation-only response to the real OOM. It preserves the registered author batches and training method. This source verdict supports running the declared engineering gates; v2 numerical parity and full-B128 fit remain unmeasured.

## Actual primary evidence

- Terminal campaign: FAILED during M0; 3 complete, 1 failed, 5 pending. All 9 formal jobs remain PENDING and report invocations are 0.
- RGBNT100/global-only fails at `correspondence_roles.py:86` while stacking role snapshots, before its first update; its training-steps file is empty. The error requested 218 MiB, with 78.56 MiB free on a 23.56 GiB card; this process occupied 23.42 GiB, including 22.52 GiB PyTorch allocated. Campaign physical GPU1 appears as CUDA GPU0 inside that worker.
- Original successful M0 allocated peaks: RGBNT201/global-only 13,792,172,544 bytes; MSVR310/global-only 13,790,093,824 bytes; RGBNT201/semantic 14,905,913,344 bytes. All have 8 effective updates and exact reload output in their receipts. These B64 measurements do not establish B128 fit.
- Independently verified all 274 collected source hashes and all 51 primary text sizes/hashes, equality to the original manifest, all 265 predecessor source hashes, and the 9 unchanged v1-added repo files.

Evidence root: `C:\Users\gb\.codex_tmp\independent_evidence_draft\terminal_intake771`. The JSON contains every actual source/input digest.

## Source findings

- **Actual failure and intervention — PASS.** The real full-B128 first forward ran out of memory in role-stage stacking before any recorded training step. Per-modality activation recomputation directly targets retained visual activations without reducing the scientific batch.
  Evidence: `primary/logs/independent_native_evidence_20261003_v1/independent_native_evidence_20261003_v1_m0_global_only_RGBNT100/m0.log:18-43`; `source/modeling/trifusion/correspondence_roles.py:58-90`; `primary/logs/independent_native_evidence_20261003_v1/campaign.json`.
- **Forward axes and adapters — PASS.** Per-modality results stack to the original depth, role, batch, modality, token, channel layout. Adapter output, shared output plus mean delta, and modality concatenation preserve the same operations.
  Evidence: `modeling/trifusion/checkpointed_evidence_clip.py:20-51`; `source/modeling/trifusion/correspondence_roles.py:63-90`.
- **Backward hook lifetime — PASS.** Every checkpoint invocation installs its own local hooks; finally removes them during both ordinary return and checkpoint early-stop unwinding. Stage tensors and global features are returned as differentiable outputs. A checkpoint around the old outside-hook visual call would not be equivalent, but the candidate does not do that.
  Evidence: `modeling/trifusion/checkpointed_evidence_clip.py:20-46`.
- **RNG, AMP, camera, BN — PASS.** The non-reentrant checkpoint has preserve_rng_state=True and CUDA image tensor inputs. Camera embedding stays within recomputation; active ViT and adapter paths use LayerNorm. Author BN heads remain outside recomputation. Standard checkpoint autocast replay is the appropriate mechanism; actual CUDA equivalence is still a required runtime gate.
  Evidence: `modeling/trifusion/checkpointed_evidence_clip.py:32-46`; `source/comparators/Signal-cd1b0a6/modeling/meta_arch.py:96-112`; `source/comparators/Signal-cd1b0a6/modeling/clip/model.py:168-179,227-231,447-488`; `source/modeling/trifusion/evidence_author_heads.py:42-59`.
- **State, optimizer and initializer preservation — PASS.** Existing Signal and adapter module objects are reused under identical registration names without new random initialization. Full state digest is compared before and after replacement. Original author optimizer, scheduler, sampler, losses, parameter ownership, seed42, 50 epochs and full batch configuration are reused.
  Evidence: `modeling/trifusion/checkpointed_evidence_clip.py:12-18`; `tools/run_independent_native_checkpointed.py:17-36`; `source/tools/run_independent_native_evidence.py:36-115`; `source/modeling/trifusion/evidence_author_heads.py:63-73`.
- **Production equivalence witness — PASS.** For every variant and dataset, unchanged v1 and candidate production models start with matching full state. First32 real samples are sliced from the full author loader; CPU/CUDA RNG is reset; AMP/GradScaler256 backward and unscaling precede complete trainable-gradient comparison including None masks. Outputs/logits/loss use fixed atol/rtol1e-5 and gradients1e-4. All buffers, state, RNG, BN count1 and removed hooks are checked. There is no optimizer step or scoring. Runtime receipt is not yet available.
  Evidence: `tools/check_native_checkpoint_backward.py:26-109`.
- **Full batch and formal-training gates — PASS.** Full-author-batch initialization parity and backward witnesses precede all nine eight-update M0s. M0 uses original B64/K8, B128/K16 and B64/K4, cumulative finite nonzero gradients, visual/camera updates, BN count8, native14 tensor updates and strict full-state reload. All nine M0s must complete before any full50 arm.
  Evidence: `tools/check_checkpointed_native_pair.py:33-74`; `tools/queue_independent_native_checkpointed.py:163-199`; `source/tools/run_foundation_recipe.py:189-284`; `source/tools/run_independent_native_evidence.py:126-174`.
- **Schema, source closure, strict reload and report — PASS.** New entry and queue configure the inherited trainer/checkpoint/verification schema consistently. Existing274 closure must remain unchanged. Full-state strict load, selected best epoch and less-than1e-5 metric checks remain inherited. Report checks all18 jobs, all9 complete50-epoch results and actual complete batch-order equality; one report invocation is preserved.
  Evidence: `tools/run_independent_native_checkpointed.py:10-36`; `tools/queue_independent_native_checkpointed.py:39-43,152-210,213-216`; `source/tools/run_foundation_recipe.py:174-185,287-312`; `source/tools/queue_foundation_recipe.py:57-115`; `tools/report_independent_native_checkpointed.py:22-58`.
- **Resources and failure handling — PASS.** Canonical2026 root, physical GPUs0-3 only, at most4 single-GPU jobs, currently idle memory gate and240-second queue cadence remain. V1 must be failed with no formal launch and its controller absent; source guards and new-output assertions prevent unmodified retry or reuse. Failure stops new launches, lets already active jobs finish, and performs no deletion, automatic retry, batch reduction or fallback.
  Evidence: `tools/queue_independent_native_checkpointed.py:76-106,109-159`; `source/tools/queue_foundation_recipe.py:212-231`.

## Fixed verification boundaries

The backward witness builds the unchanged v1 class and new class from equal complete initialized states, resets CPU/CUDA RNG, and compares actual AMP outputs, author-head logits/loss, all unscaled trainable gradients including None masks, every buffer, final state and RNG. It requires BN count1 and cleaned hooks. Forward tolerance is atol/rtol1e-5; gradient tolerance is1e-4. No optimizer step or official scoring occurs. Its first32 real samples are an equivalence witness, while all M0 and formal batches remain B64/K8, B128/K16 and B64/K4.

All six new Python files parse via standard-library AST. The initial-pair and terminal-report copies differ from v1 only in the entry/queue imports; the queue diff is restricted to new-version source/entry paths, preserved-v1 terminal conditions and the added backward witness. No training/model module was imported and no GPU, optimizer, scorer, training or terminal-report code was executed by this reviewer. Only these reviewer files were written.

## Remaining gates

- Run the declared full-author-batch initial pairs and real production backward witnesses for all three datasets.
- Pass all nine actual eight-update full-author-batch M0s, including RGBNT100 B128/K16; preserve the native14-tensor gradient/update and strict-reload checks.
- Formal training remains gated on every M0 succeeding. No measured-memory-fit, retrieval-gain or formal-completion claim is supported by this review.
- Existing one-seed, consumed-official-selection and native-capacity limitations remain. Engineering acceptance is not performance acceptance.

Inherited evidence is read from the exact terminal source closure. This does not claim every unrelated file in the local publication checkout equals the remote runtime tree. The reviewer did not query remote process state; the new queue checks prior-controller absence at execution.

## Candidate input hashes

| File | SHA-256 |
|---|---|
| `modeling/trifusion/checkpointed_evidence_clip.py` | `a41841c1a8b7be58c263b33b8575395d56e95327ab0c252a308567ffd881d884` |
| `tools/run_independent_native_checkpointed.py` | `36d680fcd0ed2339c897b06a7f695025b7a2c057af4575228951ff32c2da6bfa` |
| `tools/queue_independent_native_checkpointed.py` | `a9ea96aaf5f577bf31a73349f4ef1691c399b221be13709623f1b886e141dae6` |
| `tools/check_checkpointed_native_pair.py` | `56c11fb822423c30a359161d1e2affeb23552b058318058ef5aedcfeec64cc60` |
| `tools/check_native_checkpoint_backward.py` | `b5e866f8600e40cb36d07c97b8d6c76303be17797fe750d67824ca19e6285c4b` |
| `tools/report_independent_native_checkpointed.py` | `5fa464ed2d15d4e368c546cf68772a11785bdbc1f29560d0a393643ae39d933a` |
| `refine-logs/independent_native_checkpointed_v2/EXPERIMENT_PLAN.md` | `7adb75c7d7f7cb4efee25cb108b301b361997715e2c4dac9eb5eaefe2ce42972` |

Created: 2026-10-03T13:33:12.692057+08:00.

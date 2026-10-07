# Experiment code review: original-block checkpointing v3

**Verdict: PASS — SOURCE_ONLY. Blocking findings: none. V3 runtime: not established.**

Fresh native Codex context; requested `gpt-6-astra`, reasoning effort `max`, `fork_turns=none`. Actual backend identity is not independently verified. Review independence: **same-family**. Acceptance: **provisional**.

The 18-line block helper is the simplest justified computation-only revision after the observed v2 gradient-gate failure. It supports proceeding to the unchanged engineering gates. It does not establish numerical parity, full-B128 memory fit or scientific improvement.

## Actual failure and diagnosis

The original RGBNT201 v2 witness fails at `check_native_checkpoint_backward.py:84`: `semantic`, `evidence_model.backbone.signal.clip_vision_encoder.cv_embed`, maximum absolute difference **0.0009765625**, against the pre-fixed `atol=rtol=1e-4` comparison. Its preceding raw/deployed/global, author-head and loss checks passed. Later semantic gradient/buffer/RNG checks and native-arm checks cannot be credited from this traceback; no complete backward witness JSON was written.

Three RGBNT201 initializers and one full-author-batch eval pair exist. The original campaign file still says `INITIALIZING`, with an empty job list. The preserved launcher `CalledProcessError`, witness traceback and observer's absent controller establish the terminal initialization failure; they do not justify rewriting the raw campaign status. No v2 M0, formal training or report is established.

The pinned camera path gathers `self.cv_embed[cam_label]`, scales it and adds it to CLS before the visual transformer. V2 reordered the stage-stack construction and enclosed each modality's stage/global outputs together in checkpointing. Different branch-gradient accumulation ordering is a plausible source-level explanation for the numerical discrepancy. No saved complete v2 gradient differences or original-versus-original control establish that as the unique cause. The final v3 plan states that limit correctly.

Independently verified every one of the **282** collected v2 source digests and **11** primary artifact sizes/digests against `terminal_intake772/INTAKE.json`. All **274** predecessor v1 source entries remain unchanged. The published v2-failure intake used by the new queue is byte-identical to the collected intake.

## Source findings

- **Minimal repair — PASS.** `block_checkpointed_evidence_clip.py:8-18` stores each actual block's original bound forward as a plain attribute and binds one replacement with `MethodType`. It leaves the original backbone instance and `CrossLayerAdaptedCLIP.forward` intact. There is no new constructor, copied module, registered child, parameter path or `.block` prefix. `run_native_block_checkpointed.py:17-24` checks complete state before and after installation.
- **Hook and branch structure — PASS.** Normal residual-block `__call__` owns the outer adapter snapshot hooks. The replacement checkpoint-calls only the saved bound forward, so backward recomputation does not re-enter those block hooks. Original modality order, adapter outputs/writeback, depth/modal stacking and camera path are retained. The original forward's count-three assertion and hook cleanup remain. Source: sealed `correspondence_roles.py:58-90`, pinned Signal `clip/model.py:388-404,447-488` and the v3 helper.
- **AMP, RNG and BN boundaries — PASS.** Checkpointing is non-reentrant, explicitly preserves RNG, and is enabled only in training with gradients. Visual block internals use attention, LayerNorm and MLP on this no-prompt/no-author-adapter path. Camera gathering, visual stem/projection, outer adaptation and author BN heads remain outside replay. Runtime AMP gradients, RNG, buffers and BN counts still require the numerical witness; source inspection does not prove their values.
- **Fixed numerical gates — PASS.** The backward witness changes only imports and pre-gate diagnostic logging. It retains equal complete starting state/configuration; first32 real author-batch samples; CPU/CUDA RNG reset; production FP16 autocast, GradScaler256 backward and unscaling; all trainable gradients including None masks; forward/head/loss `1e-5` and gradient `1e-4` atol/rtol; exact buffers, final state/RNG, BN count1 and removed capture hooks. It performs no optimizer step or official scoring. New per-parameter delta files explicitly are measurements, not PASS receipts. Source: `check_block_checkpoint_backward.py:33-137`.
- **Training contract — PASS.** Original raw author heads/losses, complete optimizer ownership/groups, semantic/native feature computation, sampler/augmentation, seed42,50 epochs,1536D deployment and strict full-state reload remain inherited. Full batches remain RGBNT201 B64/K8, RGBNT100 B128/K16 and MSVR310 B64/K4. All nine eight-effective-update M0s, cumulative live-gradient checks, visual/camera changes, author BN counts, native14-tensor updates and reload gates remain before any formal job.
- **Versioning and queue — PASS.** V3 checks the actual v2 raw `INITIALIZING`/empty-jobs/no-manifest condition, absent parent and published terminal-failure intake. The original282-source closure must match exactly at runtime. The initial-pair and terminal-report differences are imports only. Only canonical server2026 physical GPU0-3/max4, existing storage guards and240-second queue polling remain. No automatic retry, deletion, fallback, batch reduction or tolerance change was added. Source: `queue_native_block_checkpointed.py:39-43,152-200`; inherited trainer and queue verifiers.

## Remaining gates and limits

Run all declared full-batch initial pairs and unchanged training-mode backward witnesses. Then all nine real full-author-batch M0s must pass before any full50 arm. Block checkpointing targets retained block intermediates, while outer snapshots, adaptation/role graphs, stem/projection and optimizer storage still consume memory. **B128 fit is unmeasured.**

If v3 fails the same fixed numerical witness, retain the new measured deltas and perform one same-state/input/RNG original-versus-original control before another repair. Such a control is necessary to distinguish baseline numerical variability when diagnosing a repeated failure; it is not required to invent an explanation after a candidate already passes. Do not relax the gate or rerun unchanged computation hoping for a different outcome.

All six candidate Python files pass standard-library AST parsing. Source comparisons confirm that the witness's original assertions are preserved. No SSH, model/torch import, GPU, optimizer, scorer, training, terminal report or deletion was executed by this reviewer. Only these two review files were written.

The exact inherited source is the sealed282-file runtime tree. Of175 corresponding local publication files,148 are byte-identical and27 differ only in CRLF/LF;107 comparator/runtime paths are absent locally. No substantive difference was found among existing files. This review does not claim the whole publication checkout equals the runtime tree; the queue enforces the exact collected source bytes before execution.

The unchanged single-seed, consumed official epoch selection and added159296 native-parameter capacity limits remain. Engineering acceptance is not retrieval gain, robustness, unique causal attribution, three-module success or SOTA.

## Exact candidate inputs

| File | SHA-256 |
|---|---|
| `modeling/trifusion/block_checkpointed_evidence_clip.py` | `cdeeeb3e13988637ef4bf96856c4df7ba66d0f19712cbbe129426db871e3a522` |
| `tools/run_native_block_checkpointed.py` | `ca0417d46daf3bddba5a986fd7429219ef6b978322ef754af0d8e1a36be2beb7` |
| `tools/queue_native_block_checkpointed.py` | `6e6770c832920be5dda321c81d7e2028a54730105b390775cb0c0753a0350748` |
| `tools/check_block_checkpointed_native_pair.py` | `efc2d61d85567d8f8f44cd1a047b910519c18e43aecdcfa0c06d1e639d7cc13a` |
| `tools/check_block_checkpoint_backward.py` | `7fbc30d005e0034c8d58002d7586cd548644cd8f9859afa7af55da3f286fcbd2` |
| `tools/report_native_block_checkpointed.py` | `31ae49a2d8c6d5f04f170566bdf25015fd3f14b738742008b4fb375cf6eb0a45` |
| `refine-logs/native_block_checkpointed_v3/EXPERIMENT_PLAN.md` | `99643fd4796d7cc5faf29c5ef196fc1b85ffbeeaefd2d54ae47c4a7097d25229` |

The accompanying JSON records every verified predecessor source/artifact digest, the explicitly inspected inherited sources, exact candidate versions and review limits.

Created: 2026-10-03T14:02:01.127894+08:00.

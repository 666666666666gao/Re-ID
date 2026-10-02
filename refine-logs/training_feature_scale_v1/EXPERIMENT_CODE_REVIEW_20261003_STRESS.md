# F2 GPU3 stress reservation review746

**PASS - SOURCE_ONLY; same-family / provisional.** Blocking issues: none. Non-blocking issues: none.

Bounded continuation by the original native reviewer `/root/review_feature_scale_scheduling745`, `gpt-6-astra` / max. The original 745 review is preserved; this is not a newly spawned fresh-context review. No SSH, project imports, model/GPU execution, launcher/waiter execution or implementation edits were performed.

## Findings

- **Original failure is correctly classified.** The 745 launcher started controller 2042332 at 00:34:46. Startup 00:38:25 and the corrected failure receipt show the inherited initial-memory assertion failed before campaign creation. There were no initializers, M0 or training. PID 2041908 maps to the other project's `RGBNT100_dual_s42/stress_launch.json`. Original logs and the first failed ps inspector remain preserved.
- **All 14 markers are necessary and sufficient for the known reservation.** Actual `launch_stress.py` selects ready checkpoints from five primary runs plus nine repeats and always launches them on physical GPU3. Its ready order can skip still-training checkpoints. The new 14-path set matches that schedule exactly; markers are written after child completion, and all 14 successful exits exhaust the known pending set. GPU3 also retains its ordinary-evaluation gate. GPUs0-2 remain independently releasable.
- **The scientific run is unchanged.** Compared with the exact 745 queue blob, only the release predicate and manifest recording change. `run_phase`, `source_map`, commands and bindings have identical ASTs; all 13 other originally checked files retain their hashes. Memory checks, max 4,240 s polling, failure drain, initial pairs, all six M0 before six fresh full50, loss/seed/protocol/budget, strict evaluation and one CPU report are preserved.
- **The new launcher preserves 745 checks and evidence.** Exact comparison finds only six expected review/output/receipt path substitutions. The new campaign is `training_feature_scale_20261003_v2`, with a new dated report and log; no old attempt is rerun or overwritten.
- **The waiter invokes at most once.** First check 01:48+08:00, then240 s. It checks exact reviewed remote queue bytes, evaluates only the extracted release-function AST and constants, and checks current memory plus compute UUID occupancy. It imports no ML/project module. On availability it calls the fresh launcher once and exits on success or failure, without automatic retry.

AST parsing and exact-byte comparisons passed. The supplied CPU predicate harness and matching receipt cover independent 0-2 release,13/14 pending, final-marker failure and all 14 success; they were read, not rerun by this reviewer, and do not establish M0/GPU success.

## Required execution order

**Stage and verify the reviewed source/text on 2026 before starting wait746.** The waiter's exact-queue assertion will reject the old745 source. The parent explicitly confirmed: prepare/commit the failure-closeout publication, run `sync_scale_launch745.py`, verify the actual `five_copy745.json`/source bytes, then start one wait746. The deploy746 upload is later revalidation. This staging sequence and both746 helpers remain unexecuted in this review; their completion is not claimed.

## Exact reviewed files

| File | SHA256 |
|---|---|
| `queue_training_feature_scale.py` | `43de0eae7b9660de12d8729fdee06efa82f6adfcaf74599c6fda5faf00e8ae16` |
| `EXPERIMENT_PLAN.md` | `04e3f88fe9e1e11351a34e9e6aab375ae0e3946653f3abba1e9f2c33903900f0` |
| `deploy_training_feature_scale746.py` | `6e1826ff92e407307311bc4241a0fa0930b0eb0cae53e6f9232e273f7294327b` |
| `wait_and_deploy_scale746.py` | `8a8d70f2364eea22a26077af6f27c5d45e7f439d439a99e922b5742a94e052f6` |

The JSON contains unchanged-source and evidence hashes. This PASS covers the bounded source correction only; actual resource checks and initial-forward/M0/full-training/report gates remain required. It is not result, integrity or SOTA acceptance.

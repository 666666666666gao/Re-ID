# E925 launcher source audit

**PASS — `source_ready_for_prepare_m0: true`; `unresolved_blocking_source_defects: []`.** Limited to two fresh prepare/8-step M0 arms. No confirmed blocking controller error. Same-family/provisional; requested gpt-6-astra/max, actual model/effort **UNATTESTED**.

Controller: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_evidence_m0_925.py`.

- **Finite fresh scope.** Lines28–31 require unused campaign/output paths. Lines51–58 launch only gradient/random × prepare/m0, each once. `child.wait()` serializes them; nonzero exits stop at69–72. No formal50, evaluation, retry or old checkpoint replay. The final status explicitly says no formal training (102). The parent entry's mode-dependent eight-step path was already directly reviewed (`tools/run_foundation_recipe.py:211`, `:222`).
- **Initializer comparison is valid.** All14 keys at42–44 exist in the actual seal's **`rows[0].initializer`** (`INPUT_SEAL.json:569`), including full initial-state hash, trainable counts, B64/K8, heads, config and two-card placement. There is no `initializer_binding` key. The first row carries region-use-supervision metadata; comparing these selected initializer fields establishes state/config equality, not identical old/new objectives. No old weights are loaded.
- **M0 evidence fields match the parent.** Lines79–85 read actual M0 status/binding, gradient counts, reload difference, frozen/visual/camera flags, eight effective updates and BN counts. Lines86–96 require eight step rows, exposure-VJP `.grad` absence, fused-task isolation, positive checked-parameter count, `excluded=32*active`, availability192, and at least one supported active step. These names match the previously read entry/model receipts; no model re-audit was performed. Assertion failures exit nonzero and retain outer stdout/stderr/exit receipts (109–115); the campaign may retain its last running status if an acceptance assertion fails, so that status alone is not success.
- **Resources and pins are bounded.** SSH uses server port2026 (107). GPU query uses only0/1 memory-used/index (32), and children receive `CUDA_VISIBLE_DEVICES=0,1` (40). The fixed new disk admission is **2,684,354,560 bytes =2.5GiB** (35–37). Source hashes are checked initially, before each stage and finally (20,54,101); protocol/CLIP, prior seal and source-audit hashes are checked too. No install/delete path exists.

The required `EVIDENCE_SOURCE_SYNC_925.json` had not yet been generated when checked. Its actual future values and the separate sync generator were not audited. Supplying that declared exact staging receipt is an input prerequisite, not a controller-source defect or a requirement to run models before this approval. No remote sync or execution success is asserted.

**Actual SHA256**

- Controller: `1b74a7c5933aa7fd4cc8ac7686fe63df080c1eed3221c0c470fdf262b7db0514`
- INPUT_SEAL: `8e5c37e57c088a8ab53344266195bbc7183ccd27a94d0f0910ff9c20998b4b41`

Outer and embedded-remote Python AST checks passed without execution. The report only read the controller, first initializer/schema location and existing reviewer receipt references; no SSH, network, torch, GPU, model/weight call, project edit or new agent. This continuation inherits the prior report's disclosed session-note context; no new session notes were read. JSON records the exact scope and pins.

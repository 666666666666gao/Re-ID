# Source review R3: draft905

**Verdict: PASS (source only). No remaining evidenced source bug was found.** R2-M0-01 is fixed; the R1 metadata correction is retained. This is not initialization, real M0, NN launch or retrieval qualification.

`tools/run_region_use_probe.py:67–68` now disables CUDA autocast only around the isolated auxiliary VJP. The actual production backward runs outside the forward/loss autocast block (`tools/run_foundation_recipe.py:230–235`), so the observer context now matches it. The nested context restores the enclosing enabled state after the observer.

AST comparison proves that removing this one wrapper makes the whole entry identical to R2. Diagnostic scale256, targets, retained graph, finite/global checks, M0-only branch, original-plus-auxiliary loss, CURRENT_MODEL binding, CLI, optimizer and scaler are unchanged. Objective, head and text gate bytes match R2. The actual loss function was exercised only with pure Python scalar/context placeholders: the VJP sees disabled autocast, the outer enabled state is restored, and formal train makes no observer call. No torch import or numerical gradient was executed.

The plan retains the R1/R2 findings and pending R3 history, adds the observer context to its description, and correctly names four Python source files and old439 sources. The sealed903 `source_sha256` map contains439 entries; this is a count check, not current byte verification of all439 sources.

Requested reviewer route: `gpt-6-astra`, effort `max`; actual serving model/effort: **unattested**. Native Codex, same-family provisional follow-up by the same R1/R2 reviewer, not another independent review.

## Evidence and retained scope

| Check | Source verdict | Evidence |
|---|---|---|
| R2-M0-01_fix | PASS_SOURCE | tools/run_region_use_probe.py:67-68; tools/run_foundation_recipe.py:230-235 |
| minimal_change | PASS_AST_AND_HASH | R2_TO_R3.diff; SOURCE_ONLY_CHECKS_R3.json; tools/run_region_use_probe.py:56-77 |
| unchanged_objective_and_training_ownership | PASS_SOURCE_RETAINED | modeling/trifusion/region_use_objective.py:6-40; modeling/trifusion/region_use_heads.py:7-29; tools/run_region_use_probe.py:34-53; tools/run_region_use_probe.py:56-90; tools/run_foundation_recipe.py:194-241 |
| gate_reload_initialization | PASS_SOURCE_RETAINED_RUNTIME_PENDING | tools/accept_region_use_m0.py:8-36; tools/run_foundation_recipe.py:263-312; tools/run_independent_native_evidence.py:126-174 |
| plan_scope_and_count | PASS_TEXT | refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md:3; refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md:24-30; INPUT_SEAL_903.json:source_sha256 |

**R2-M0-01_fix.** Only the isolated auxiliary backward now explicitly disables CUDA autocast. This matches the actual production backward outside the enabled forward/loss block. The nested scope restores the enclosing context on exit. R2 finding is resolved for R3; no numerical draft905 activity is claimed.

**minimal_change.** Removing exactly the new one-statement disabled-autocast wrapper makes the entire entry AST identical to R2. Thus targets, scale256, finite/global checks, retained graph, CURRENT_MODEL binding, original+auxiliary addition, M0-only branch, CLI, optimizer and scaler calls are unchanged. Objective, head and text gate are byte-identical to R2.

**unchanged_objective_and_training_ownership.** R1/R2 checks retain their scope: actual final readout values are captured without detach, legal masks and empty support remain unchanged, detached global weighting and mass normalization are unchanged, no inference labels are used, original task/BN/optimizer ownership is retained, and the auxiliary remains added once. Observer changes no parameter, buffer, optimizer or GradScaler ownership.

**gate_reload_initialization.** The native M0 producer/reload path and the text acceptance gate are unchanged. R2 nine text fixtures remain preserved and applicable; they were not repeated. Current-model replacement at fresh reload occurs after training loss calls. Actual initialization equality, sampled support/activity, effective updates and strict reload remain unverified.

**plan_scope_and_count.** Plan retains R1 WARN and R2 FAIL history, states the observer-only correction and pending R3, and keeps NN unregistered. Four current Python source files are present; the old sealed source map has exactly439 entries. This checks the count, not all439 current source bytes. Two conditions, objectives, real M0 prerequisites and performance limits are unchanged.

R1/R2 artifacts and snapshots were verified unchanged against both saved artifact manifests. The five R3 inputs are preserved under `r3_inputs/`, and their current hashes were rechecked at finalization. No producer source was edited by the reviewer.

Real initialization state/parameter/RNG/global-output equality, sampled legal support, scale256 full-model gradients, eight effective updates, strict full-state reload, memory/time and controller/storage/queue qualification remain pending. Source PASS does not establish these. The fixed observer is not a numerical repeatability repair or a performance result. Known/repeated MaxSim, sigmoid and softplus mechanisms remain intermediate supervision of final role values; the deployed additive1536 score is unchanged and no novelty guarantee is made.

Review actions: zero torch imports, neural constructions/forwards, GPU queries, weight/tensor-cache loads, optimizer updates, queue actions and production modifications.

## Five current source/plan hashes

| File | SHA256 | R2 relation |
|---|---|---|
| `modeling/trifusion/region_use_objective.py` | `556876002cc7abef5a3525940781b8facdf6a1e8bfc792e5b900f16a5f72fe6d` | unchanged |
| `modeling/trifusion/region_use_heads.py` | `1680d04f7b26f513380509af88bfce555cbc04497162e4af10c47dcbbcc9b63e` | unchanged |
| `tools/run_region_use_probe.py` | `176e1e88a74ede5447f57cf0e3a89c1a0847e729647d37b11fdb6c76ea715591` | minimal reviewed diff |
| `tools/accept_region_use_m0.py` | `f378797bcb9c641156de215d38b9db8dc34668c5ac7dad15bf214decd442f3ac` | unchanged |
| `refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md` | `576138fc39d97d251fcb2522a11ad184c8eb66c0f388b009f2b8f1e634214074` | minimal reviewed diff |

Final five-file/current source hash recheck: 2026-10-12T10:56:41.925973+00:00.

The JSON report includes the full inherited/direct input hash inventory and precise line coverage. The existing mamba.py CRLF versus sealed LF qualification remains unchanged.

Same-family provisional source PASS only. Actual serving model/effort is unattested. No initialization/M0/NN/queue/performance qualification; no novelty guarantee.

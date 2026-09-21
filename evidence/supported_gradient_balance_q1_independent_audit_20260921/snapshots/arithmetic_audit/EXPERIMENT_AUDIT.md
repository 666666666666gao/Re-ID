# Supported gradient balance CPU equations audit

Date: 2026-09-21. Fresh delegated Codex reviewer, same-family, provisional.

Overall verdict: **WARN**. The two observed CPU stops are verifier arithmetic mismatches. The narrowly specified posthoc replay repair is **recommended**. No tolerance change is justified or needed. This review does not establish a full Q1 verification or scientific pass.

Review target: `/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/q1`. Frozen stats SHA256 `5f8f157b39a9b11dd6ac5447b2a8619915cf20ddb6815fdc4ac107d7a571b327`. Frozen config SHA256 `9ce36299299efbc09ccfb72f5b1c4fed20fc1026bd9fee1bd55a591d6950ec7b`. The saved summary currently records `Q1_FAIL`; this audit records that field, without independently adjudicating its retrieval metrics.

Packaging disclosed one concurrent executor edit: the posthoc recheck entrypoint changed after the initial source read. The original snapshot was recovered from Git object `f881972c4a66d5153946f9df5f11e7bb1090e996` and its SHA verified as the originally read `4974606b9be040845f904e4c3826135cfe6c9b7bda3daa364407097f54a43299`. The updated entrypoint was then read in full and separately snapshotted (SHA `59e76b3ded4d097b8cc647d62681c91151d75b671c7099a9860df2f54ebbaddd`); it contains exactly the recommended source transformation and asserts the tested candidate-module hash. The approval below applies to that inspected posthoc entrypoint. No frozen training/statistics source was changed by this audit.

## Findings

1. The first stop is an exact-replay mismatch. The controller uses `math.sqrt` at `tools/msvr_supported_gradient_balance.py:37-40`; the scalar verifier uses exponentiation by `.5` at `tools/verify_msvr_supported_gradient_balance_stats.py:60`, then compares the resulting ratio exactly at line 63. On the actual remote Python 3.10.14 runtime, 6 of 3,486 supported role records differ by one binary64 ULP between these expressions. All recorded ratios exactly match `math.sqrt`, as do all controller EMA states and proposed/applied Python coefficients. This is not a reason to replace exact comparison with a tolerance.

2. The later failure is **line 77, the supported balanced applied-gradient weighted-norm identity**, not line 74's `direct_sum_vs_original` identity. The precise location is fold 0, balanced endpoint, one-based step 71, Mamba. The executor's diagnostic collector is correctly labeled diagnostic-only; it continues after failed `close` checks and cannot serve as a verification PASS. Its one recorded failure was independently reproduced from the six original remote scalar logs.

3. The line 77 formula uses the logged Python binary64 weights. Actual `combine()` multiplies float32 tensors by Python scalars (`tools/msvr_supported_gradient_balance.py:70-86`). PyTorch's multiplication uses float32-converted scalars in this case. Its results are only later converted to double for norm/cosine statistics (`tools/probe_msvr_history_candidate_gradients.py:59-66`). Double-precision measurement does not undo earlier coefficient conversion or float32 arithmetic.

4. Converting only the formula's coefficients to float32 removes the demonstrated mismatch while keeping the original threshold. This is representation alignment, not fitting a tolerance to the observed maximum.

| Step 71 Mamba quantity | Value |
|---|---:|
| Recorded rank norm | 2.0612712148802683 |
| Recorded auxiliary norm | 0.3682791489070195 |
| Recorded rank/auxiliary cosine | 0.20741744932986447 |
| Logged Python rank weight | 1.0421102637890658 |
| Actual float32 rank coefficient | 1.0421102046966553 |
| Logged Python auxiliary weight | 0.9578897362109341 |
| Actual float32 auxiliary coefficient | 0.9578897356987 |
| Recorded applied norm squared | 5.0530119215775136 |
| Original binary64-coefficient prediction | 5.05301246766921 |
| Original absolute error | 5.460916963073714e-7 |
| Original `close` threshold | 5.053012467669209e-7 |
| Float32-coefficient prediction | 5.053011926248935 |
| Corrected absolute error | 4.671421471869053e-9 |

5. All 1,743 supported balanced role records satisfy the **unchanged** `1e-7 * max(1, abs(lhs), abs(rhs))` test with the correctly represented coefficients. The maximum error/threshold ratio is 0.025020372852300826 (fold 2 balanced, step 138, Mamba). The original coefficient version has exactly one failure with ratio 1.0807250126561945. No other formula needed adjustment.

6. All 4,680 direct-sum norm identities already pass, with maximum error/threshold ratio 0.008122998940152115. All 4,680 current-plus-history rank identities also pass, maximum ratio 0.008598460606910828. The actual nonzero `direct_sum_vs_original.difference_norm` at the failing row is 0.0019766294160528234. That records the already-disclosed difference between direct component backpropagation and combined backpropagation; the R2 plan explicitly does not require these two vectors to be equal (`EXPERIMENT_PLAN.md:58-61,100-102`). It must remain visible and must not be relabeled as zero.

## Mathematical validity and thresholds

Let r = ||R||, a = ||A||, c = cosine(R,A), and d = r*a*c (or zero when either vector is zero). In real arithmetic,

`||wR*R + wA*A||^2 = wR^2*r^2 + wA^2*a^2 + 2*wR*wA*d`.

The algebra is valid. The coefficients must describe the actual tensor arithmetic: `wR32 = float32(wR)` and `wA32 = float32(wA)`. The actual float32 multiply/add sequence still rounds, so the formula with converted coefficients remains an approximate consistency check, not a bitwise reconstruction of the vector. All recorded AMP scales were 256.0, and the source takes direct/current/history components to float32 before combining them (`train_msvr_supported_gradient_balance.py:174-180,222-244,267-297`).

For normal finite arithmetic, with unit roundoff u = 2^-24 and gamma2 = 2*u/(1-2*u), a standard forward-error estimate for two products and one sum after coefficient conversion is

`||y - (wR32*R+wA32*A)|| <= gamma2 * (abs(wR32)*r + abs(wA32)*a)`.

This follows by bounding each product and the subsequent sum; no fitted experiment constant is used. It gives an associated squared-norm bound `2*||z||*E + E^2`. The audit collector records this only as a descriptive diagnostic. It is **not** a replacement acceptance gate: per-step vectors and underflow statistics are unavailable, and the displayed estimate omits the tiny binary64 measurement/reconstruction error. The fixed `1e-7` threshold is not a universal proof for arbitrary float32 vectors, cancellation patterns, or dtypes. In these saved records it remains unchanged and is satisfied by the correct formula. Future failures should be investigated rather than automatically absorbed by a larger threshold.

The ordinary pair identity `difference_norm^2 = first_norm^2 + second_norm^2 - 2*first_norm*second_norm*cosine` is appropriate for the double-precision statistics of a pair of stored-runtime vectors. It verifies scalar self-consistency, not equality between those vectors. The zero-norm branches are consistent with `compare()`'s `cosine=None` convention. M0's independent direct-reference bounds remain exactly 0.005 relative L2, or 1e-8 absolute L2 for a zero reference (`msvr_supported_gradient_balance.py:53-61`; stats lines 85-96). They must not be altered by this repair.

## Independent checks performed

- Read the five named primary files, the actual imported `compare()` implementation, controller and trainer call sites, the existing synthetic math tests, fixed configuration, R2 plan, and model-construction import path. No training or model evaluation was run.
- Used a new read-only script on the existing remote environment to independently replay controller arithmetic and norm formulas. Each of the six `memory_steps.jsonl` files was SHA/size checked against its `training.json` audit receipt. Remote hashes of the seven directly bound source/config files match local snapshots.
- Checked 1,560 steps, 4,680 role records, and 1,743 supported balanced role combinations. Supported steps per endpoint are 194/194/193 across folds; post-warmup unsupported steps are 1/1/2, repeated in both endpoints. Their locations are fold 0 step 180, fold 1 step 221, fold 2 steps 133 and 232.
- In installed PyTorch `2.5.1+cu121`, using CPU only and leaving CUDA uninitialized, checked all 9,360 applied Python coefficients via `torch.ones(1, dtype=torch.float32) * weight`; every result exactly equals `struct`'s IEEE float32 conversion.
- A clearly labeled synthetic one/zero vector using the step-71 coefficients reproduces the original threshold failure: 1.231616115759948e-7 error against 1.0859938018945163e-7. With float32 coefficients the discrepancy is zero. This is a software arithmetic fixture, not a training-gradient replay or performance measurement.
- Applied exactly the two proposed arithmetic substitutions in memory and executed the original `verify_balance()` with every original assertion and threshold intact. All six complete scalar logs pass. The candidate in-memory source SHA256 is `4f21a5ec2ace07ed446d56c39881b3e3ce7c3783ff6edfa0bad26bc0b0e85231`. The original stats file remained byte-identical.

PyTorch's version-specific primary sources corroborate the conversion mechanism: [2.5.1 dtype promotion documentation](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/tensor_attributes.rst), [CUDA multiplication kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cuda/BinaryMulKernel.cu), [CUDA scalar extraction into the operation type](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cuda/Loops.cuh), and [operation arithmetic types](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/OpMathType.h). The CPU fixture establishes the installed runtime behavior; the CUDA implementation was inspected as source, not re-executed. The [2.5.1 numerical accuracy note](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/numerical_accuracy.rst) explains why mathematically equivalent floating-point expressions need not match bitwise.

## Minimum recommended repair

**Recommend the parent's proposed posthoc entrypoint change.** Keep frozen training/statistics/configuration bytes intact, including their hashes. In `recheck_msvr_supported_balance_sqrt.py`'s existing in-memory, source-hash-guarded transformation:

1. Retain the `math.sqrt` substitution already required for exact controller replay.
2. Add `import struct` only to the temporary transformed module and replace the original line 77 formula with the following representation-aware expression:

```python
wr32,wa32=(struct.unpack('<f',struct.pack('<f',w))[0] for w in (wr,wa))
close(actual**2,wr32*wr32*r*r+wa32*wa32*a*a+2*wr32*wa32*ra)
```

Do not round `wr`/`wa` earlier: logged states, ratios, proposed/applied coefficients, controller range and sum assertions retain their original Python-double meaning. Do not change line 74, line 82, `close()`, any reference bound, any scientific gate, training source, checkpoint, or saved row. Disclose both repairs in the new receipt, preserve both failed receipts/logs, and use a new output destination. Then run the full original CPU verifier through this entrypoint; its success must be observed separately before claiming a complete replay. No retraining is needed to diagnose or repair these two verifier defects.

## Experiment-audit checklist

| Check | Status | Evidence and boundary |
|---|---|---|
| A. Ground-truth provenance | WARN / scoped | This is scalar runtime-witness consistency, not accuracy evaluation. Support counts replay saved identity/scene metadata; independent dataset/image reinspection is outside this audit. No synthetic quantity is presented as dataset GT. Full verifier label binding is at `verify_msvr_supported_gradient_balance.py:75-83,117-122`. |
| B. Score normalization | PASS in scope | No score is normalized to create a performance result. EMA/weight ratios are the registered optimizer controller, and relative L2 checks remain engineering checks. Raw norms/errors are retained. |
| C. Result existence | PASS in scope | Both original failure receipts and the new diagnostic files exist; original remote scalar-log hashes/sizes match their receipts; the independent replay artifacts are actual outputs. This does not validate retrieval arrays/checkpoints. |
| D. Executed code | PASS in scope | Trainer imports/calls the reviewed `combine` and `compare` functions at training lines 61,68,267,276-287; CPU calls `verify_balance` at verifier lines 178-179. All original scalar assertions were exercised by the independent in-memory replay. |
| E. Scope | WARN | Exactly one seed, three folds, two endpoints, 1,560 saved steps; full parameter-gradient vectors were not saved/reconstructed. The six Q1 logs contain zero direct-reference steps; those checks belong to earlier M0 and are not re-established here. |
| F. Evaluation classification | declared | Main: `runtime_witness_scalar_consistency`. Supplementary: `simulation_only` synthetic arithmetic fixture. Neither supplies a retrieval-performance or generalization claim. |

## Claim impact and remaining work

- **Supported:** the two observed scalar-verifier failures have concrete arithmetic causes; coefficient representation is the cause of the remaining weighted-norm failure; all original scalar assertions pass under the two minimal representation/operation corrections without tolerance changes.
- **Supported with limitation:** no training discrepancy is evidenced by this particular failed assertion. Scalar norms/cosines can neither reconstruct vector coordinates nor rule out every training defect. Runtime `torch.equal` and preserved-head flags remain historical witnesses, not fresh gradient recomputation.
- **Not supported by this audit:** full Q1 CPU completion, scientific qualification, official evaluation, causal retrieval improvement, checkpoint/array integrity outside the files expressly checked, or whole-goal success. The saved summary's `Q1_FAIL` field is not converted into a pass by a verifier repair.
- **Still required:** execute the complete CPU replay with the bounded posthoc repair, retain its actual terminal receipt, and perform the full terminal evidence/claim audit. The root executor owns that work; this auditor launched no full verifier, training, model load, or GPU task.

No credentials were inspected. No checkpoint, saved feature/distance array, or image was downloaded. All new audit scripts, scalar outputs, findings, and source snapshots are confined to this new D: temporary audit directory. No repository, memory, or remote experiment file was modified by this auditor.

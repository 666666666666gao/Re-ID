"""Package this bounded audit under its new temporary directory only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
TMP = ROOT.parent
primary = json.loads((ROOT/'primary_scalar_audit.json').read_bytes())
replay = json.loads((ROOT/'minimum_repair_scalar_replay.json').read_bytes())
names = [
    'tools/verify_msvr_supported_gradient_balance_stats.py',
    'tools/msvr_supported_gradient_balance.py',
    'tools/recheck_msvr_supported_balance_sqrt.py',
    'tools/probe_msvr_history_candidate_gradients.py',
    'tools/verify_msvr_supported_gradient_balance.py',
    'tools/train_msvr_supported_gradient_balance.py',
    'tools/check_msvr_supported_gradient_balance_math.py',
    'tools/train_msvr310_trifusion_oof.py',
    'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json',
    'refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_PLAN.md',
]
provided = [
    'trifusion_supported_balance_ratio_diagnosis_20260921.json',
    'trifusion_supported_balance_cpu_bound_recheck_observation_1_20260921.json',
    'diagnose_supported_balance_scalar_equations_20260921.py',
    'trifusion_supported_balance_scalar_equations_diagnosis_20260921.json',
]
hashes = {}
for relative in names:
    # The executor changed this posthoc entrypoint during packaging. Preserve the
    # original reviewed bytes from the already-observed Git object explicitly.
    raw = (subprocess.check_output(['git', 'show',
        'f881972c4a66d5153946f9df5f11e7bb1090e996:'+relative], cwd=REPO)
        if relative == 'tools/recheck_msvr_supported_balance_sqrt.py' else (REPO/relative).read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if relative in primary['source_sha256']:
        assert digest == primary['source_sha256'][relative], relative
    target = ROOT/'snapshots'/'repo'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == raw
    else:
        target.write_bytes(raw)
    hashes['repo/'+relative] = 'sha256:'+digest
for name in provided:
    raw = (TMP/name).read_bytes()
    target = ROOT/'snapshots'/'provided'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    target.write_bytes(raw)
    hashes['provided/'+name] = 'sha256:'+hashlib.sha256(raw).hexdigest()
posthoc_name = 'tools/recheck_msvr_supported_balance_sqrt.py'
posthoc_raw = (REPO/posthoc_name).read_bytes()
assert hashlib.sha256(posthoc_raw).hexdigest() == '59e76b3ded4d097b8cc647d62681c91151d75b671c7099a9860df2f54ebbaddd'
posthoc_target = ROOT/'snapshots'/'posthoc'/posthoc_name
posthoc_target.parent.mkdir(parents=True, exist_ok=True)
assert not posthoc_target.exists()
posthoc_target.write_bytes(posthoc_raw)
hashes['posthoc/'+posthoc_name] = 'sha256:'+hashlib.sha256(posthoc_raw).hexdigest()
for name in ['collect_primary_scalars.py', 'primary_scalar_audit.json',
             'verify_minimum_repair_readonly.py', 'minimum_repair_scalar_replay.json',
             'write_findings.py']:
    hashes[name] = 'sha256:'+hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

report = r'''# Supported gradient balance CPU equations audit

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
'''

stamp = datetime.now(timezone.utc).isoformat()
output = dict(
    audit_skill='experiment-audit', verdict='WARN', overall_verdict='warn', integrity_status='warn',
    reason_code='verifier_arithmetic_mismatch_with_saved_scalar_scope',
    summary='Observed failures are sqrt-operation and coefficient-dtype replay mismatches; minimum posthoc repair passes all scalar assertions with unchanged thresholds.',
    date='2026-09-21', generated_at=stamp,
    agent_id='/root/audit_supported_balance_cpu_replay_failure_20260921',
    verdict_id='supported_balance_cpu_equations_20260921',
    executor_model='codex-gpt-6-astra', executor_family='openai',
    reviewer_model='gpt-6-astra', reviewer_family='openai', reviewer_reasoning='max',
    reviewer_model_attribution='requested reviewer routing; backend model identity not independently verifiable',
    review_independence='same-family', acceptance_status='provisional',
    trace_path=str(ROOT), audited_input_hashes=hashes,
    concurrent_input_change=dict(file='tools/recheck_msvr_supported_balance_sqrt.py',
        actor='parent executor', original_snapshot_source='Git f881972c4a66d5153946f9df5f11e7bb1090e996',
        original_sha256='4974606b9be040845f904e4c3826135cfe6c9b7bda3daa364407097f54a43299',
        updated_sha256='59e76b3ded4d097b8cc647d62681c91151d75b671c7099a9860df2f54ebbaddd',
        updated_entrypoint_read_and_reviewed=True),
    local_repository_head_at_start='f881972c4a66d5153946f9df5f11e7bb1090e996',
    remote_run=primary['run'], summary_sha256=primary['summary_sha256'],
    saved_summary_status=primary['original_scientific_status'], saved_summary_status_independently_scientifically_audited=False,
    repair_recommendation=dict(status='APPROVE_BOUNDED_POSTHOC_REPLAY',
        target='tools/recheck_msvr_supported_balance_sqrt.py in-memory transformation only',
        transformations=['replace exponentiation by .5 with math.sqrt',
                         'convert only line77 formula coefficients to IEEE float32 via struct'],
        original_stats_sha256=replay['source_sha256'],
        candidate_stats_sha256=replay['candidate_source_sha256'],
        tolerance_changes=False, scientific_gate_changes=False, training_changes=False,
        source_config_checkpoint_changes=False, whole_q1_pass=False),
    deterministic_scalar_replay=dict(status=replay['status'], steps=1560, role_records=4680,
        supported_role_rows=3486, supported_balanced_role_rows=1743,
        exact_sqrt_ratio_mismatches=0, exponentiation_one_ulp_mismatches=len(primary['ratio_power_mismatches']),
        original_weighted_norm_failures=1, corrected_weighted_norm_failures=0,
        direct_sum_identity_failures=0, torch_scalar_conversion_checks=replay['torch_python_scalar_conversion_checks'],
        unchanged_threshold='1e-7 * max(1, abs(a), abs(b))',
        maximum_corrected_error_to_threshold=primary['maxima']['weighted_norm_float32_weights']['ratio_to_original_threshold']),
    failure_witness=primary['failing_witnesses'][0],
    checks=dict(
        gt_provenance=dict(status='warn', details='No performance GT evaluated; saved support labels and runtime scalar consistency only.'),
        score_normalization=dict(status='pass', details='No performance normalization; registered controller and raw gradient statistics.'),
        result_existence=dict(status='pass', details='Primary scalar file hashes/sizes match receipts; full arrays/checkpoints outside scope.'),
        dead_code=dict(status='pass', details='Actual imported compare/controller/trainer and scalar verifier call paths inspected and exercised.'),
        scope=dict(status='warn', details='Single-seed saved scalar witnesses; no gradient regeneration or complete Q1 verifier execution.'),
        eval_type='runtime_witness_scalar_consistency', supplementary_eval_type='simulation_only'),
    claims=[dict(id='C1', claim='Two observed scalar failures are replay arithmetic mismatches', impact='supported'),
            dict(id='C2', claim='Minimal posthoc repair passes all original scalar assertions with original thresholds', impact='supported'),
            dict(id='C3', claim='Full vector gradients and actual training updates independently reconstructed', impact='unsupported'),
            dict(id='C4', claim='Complete Q1 or scientific qualification established', impact='unsupported')],
    limitations=[
        'same-family semantic review remains provisional',
        'model backend identity not independently verified',
        'saved scalar gradients are runtime witnesses and cannot uniquely reconstruct vectors',
        'Q1 contains zero direct-reference checks; earlier M0 not independently rerun here',
        'full CPU verifier and checkpoint/retrieval-array checks not run by this audit',
        'CPU synthetic arithmetic fixture is not GPU or model gradient replay',
        'fixed 1e-7 predicate is not a universal float32 forward-error theorem',
        'full retrieval/scientific interpretation remains outside scope'],
    external_primary_sources=[
        'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/tensor_attributes.rst',
        'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cuda/BinaryMulKernel.cu',
        'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/cuda/Loops.cuh',
        'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/OpMathType.h',
        'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/numerical_accuracy.rst'],
    model_forwards=0, optimizer_updates=0, training_launched=False, full_q1_verifier_launched=False,
    remote_experiment_files_modified=False, repository_modified=False,
    downloaded_models_arrays_images=0)
for name, raw in [('EXPERIMENT_AUDIT.md', report.encode('utf-8')),
                  ('EXPERIMENT_AUDIT.json', (json.dumps(output, indent=2, ensure_ascii=False)+'\n').encode('utf-8')),
                  ('INPUT_MANIFEST.json', (json.dumps(hashes, indent=2)+'\n').encode('utf-8'))]:
    path = ROOT/name
    assert not path.exists()
    path.write_bytes(raw)
    print(json.dumps(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())))

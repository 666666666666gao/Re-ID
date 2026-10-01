**Overall verdict: WARN. Engineering and metric-integrity checks pass; both registered scientific gates fail.**

`review_independence: same-family`  
`acceptance_status: provisional`  
`auditor_agent: /root/audit_visual_update_complete_708`  
`date: 2026-10-01`

This was a fresh, read-only audit. I performed no neural forward, optimizer update, training, report rerun, installation, or publication. Model weights, photographs and arrays remained remote. The parent should preserve this response verbatim with the reviewer request and actual call metadata.

For compact references below:

- `C` = `logs/visual_update_control_20261001_v1`
- `P` = `logs/visual_update_preflight_20261001_v1`
- `O` = `results/visual_update_control_complete_20261001`
- `A` = `logs/visual_update_complete_708_20261001`

Remote references are relative to `/data/gaob/Re-ID/Trifusion`. Preserved campaign/report references also exist under local `C:/Users/gb/.trifusion_github_publish_22c3bee/A/raw/`. I verified those archived bytes against their remote originals.

**Actual verification completed**

- All **185 intake files** matched their recorded sizes and SHA-256 hashes locally and matched remote originals. `SNAPSHOT.json` matched its intake hash.
- All **222 remote manifest-bound source/config/protocol files** matched. The preflight and initialization-witness hashes matched their manifest bindings.
- All **12 selected checkpoints**, **12 M0 checkpoints**, and **three upstream baseline checkpoints** were CPU-loaded and inspected.
- CPU reconstruction using the actual model classes and production Mamba reproduced **all 12 initial model hashes, common-initializer hashes and trainable-parameter counts**. All **24 selected/M0 checkpoints strictly reloaded**, with exact tensor equality afterward.
- That constructor check redirected the author's explicit CUDA constructor transfers and CUDA-availability query to CPU **inside the audit process only**. No source files were edited and no forward was called. All 222 source hashes were checked again afterward.
- All **12 distance matrices / 48 official metrics** were independently recomputed. Maximum discrepancy using the official float32 CMC convention was **0 percentage points**.
- Verified **600 complete epoch records, 40,832 formal training steps and 96 M0 steps**. Epoch means, loss decomposition, learning-rate schedules, best-epoch selection and training-log rows matched.
- Independently checked all **12 paired diagnoses**, including query AP changes, identity aggregates, repairs/new errors and the 2,000-resample bootstrap.
- Checked filenames, labels, modality membership and complete split membership for **27,266 protocol records**, without opening photographs.
- Checked all **48 milestone snapshots**. Completion counts progressed monotonically from 0 to 12; archived logs contained no traceback/assertion-failure entries.
- Verified all **24 fifty-point SVG curves** against the histories; maximum coordinate error was **4.9940×10⁻⁷ SVG units**. I also visually inspected the PNG.
- Verified the four newly mirrored local report artifacts against the original archive. The actual earlier report-code-review response exists and matches its recorded response hash.

The main remote checkpoint/array/history verifier exited **0** after **15,539 assertions**. The CPU-construction/strict-reload check exited **0** after **342 assertions**. Metadata verification exited **0** after **54,659 assertions**. Local history/cost checks completed **684 assertions**, followed by the successful SVG checks.

**A. Ground-truth provenance — PASS**

The retrieval targets come from dataset identities and camera/time metadata, not predictions or teacher outputs.

- The protocol builder constructs paths and labels from the audited dataset inventory, verifies train/test identity separation, retains the complete gallery and checks every query has a legal positive: `tools/build_official_three_dataset_protocols.py:21`, `:32`, `:44`, `:56`, `:74`.
- The actual loader uses training labels only for the training split and dataset identities for evaluation: `tools/official_three_dataset_data.py:8`.
- The executed evaluator obtains identities/cameras/scenes directly from protocol records and calls the bound author scorers: `tools/run_correspondence_roles.py:79`, `:88`, `:93`.
- RGBNT201/RGBNT100 discard **same identity AND same camera**. MSVR310 discards **same identity AND same scene/time identifier**; it does not discard every gallery item from the same scene. Evidence: `comparators/Signal-cd1b0a6/utils/metrics.py:68`, `:137`.
- The filename parsing matches the author loader semantics: `comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61`, `RGBNT100.py:63`, `msvr310.py:67`.

Verified scope:

| Dataset | Train / query / gallery records | Train / query / gallery identities |
|---|---:|---:|
| RGBNT201 | 3,951 / 836 / 836 | 171 / 30 / 30 |
| RGBNT100 | 8,675 / 1,715 / 8,575 | 50 / 50 / 50 |
| MSVR310 | 1,032 / 591 / 1,055 | 155 / 52 / 155 |

MSVR310’s **103 gallery-only identities are retained**. Every saved array’s labels/order matched the complete protocol. All queries had valid positives after filtering.

This verifies the supplied dataset/protocol metadata and filesystem membership. I did not re-download datasets or repeat image-content/ZIP-CRC provenance checks.

**B. Score normalization — PASS**

I found no prediction-derived metric denominator or self-normalized performance score.

- AP is computed from ranked true matches and the number of legal positives. CMC averages valid-query indicators: `comparators/Signal-cd1b0a6/utils/metrics.py:93`, `:104`, `:153`, `:166`.
- The independent scorers use the same legal-match definitions: `tools/train_rgbnt100_signal_oof.py:253`; `tools/train_msvr310_signal_oof.py:223`.
- L2 normalization applies to embeddings before squared Euclidean distances; it does not rescale reported mAP by a model-dependent maximum: `tools/run_official_three_dataset_roles.py:230`.
- Multiplication by 100 expresses percentages: `tools/run_correspondence_roles.py:100`.

The audit’s independently reconstructed official-format scores matched every stored metric exactly.

**C. Results, checkpoints and completion evidence — WARN for current documentation; numerical/artifact checks PASS**

All twelve claimed endpoints exist and have complete evidence.

- The manifest registers seed 42, 50 epochs and twelve conditions: `C/manifest.json:2`, `:3`, `:4`. Its immutable plan binding is at `:132`.
- The accepted matrix reports twelve verified endpoints, with separate run directories and checkpoint bindings: `C/accepted_matrix.json:4`, `:7`, `:47`, `:87`, `:127`, `:167`, `:207`, `:247`, `:287`, `:327`, `:367`, `:407`, `:447`.
- Parent completion and actual zero worker exits are retained: `C/campaign.json:2`, `:65`, `:126`, `:187`, `:248`, `:309`, `:370`, `:431`, `:492`, `:553`, `:614`, `:675`, `:736`, `:741`.
- I verified **12 unique formal-training processes, 36 unique M0/train/evaluate processes, 12 distinct formal output directories and 12 distinct selected-checkpoint hashes**.
- The worker obtains exit codes from `process.wait()` and verifies the completed endpoint before marking it complete: `tools/queue_visual_update_control.py:62`, `:79`, `:87`.
- The two reused RGBNT201 M0s are explicitly reused qualification evidence; formal training constructs a fresh model: `tools/queue_visual_update_control.py:58`, `:65`; `tools/run_visual_update_control.py:112`, `:116`. No M0 checkpoint is loaded by `train()`.

**Checkpoint/initialization findings**

All four conditions use FP32 visual storage. Only low-LR conditions train the 152 visual tensors; the visual learning rate is 5×10⁻⁶. Camera and other Signal states remain outside that update group. Evidence: `tools/run_visual_update_control.py:25`, `:66`, `:75`, `:82`, `:120`.

Direct inspection of both selected and M0 checkpoints found:

- **152/152 visual tensors changed** in every low-LR endpoint.
- **0/152 changed** in every frozen endpoint.
- All stored nonvisual Signal states, including camera state, matched their upstream baseline exactly.
- The full Signal key set was present.
- Role states were present only in independently trained role models.
- Saved state counts were 305/219 for RGBNT201 roles/global-only and 317/231 for RGBNT100 and MSVR310 roles/global-only.

Complete-model saving and strict loading are implemented at `tools/run_visual_update_control.py:93` and `:102`. The audit independently exercised strict CPU loading of all 24 saved models.

The selected checkpoint supplies all four metrics; ties use the later epoch. Evidence: `tools/run_visual_update_control.py:188`, `:227`; `tools/report_visual_update_control_complete.py:73`. All 600 epoch records and all selected epochs matched this rule.

The checkpoints contain complete **model state**. They do not contain optimizer/scaler/RNG state for exact interrupted-training resumption. Final-epoch frozen-state assertions are retained in the training receipts; the independently reloadable files are the selected and M0 checkpoints.

**Actual report execution**

The original CPU report was executed once through the recorded observer:

- `report_invocations: 1`: `C/analysis_waiter_status.json:8`
- PID **2719272**, start **17:36:41.743237+08:00**: `:12`, `:13`
- Actual exit code **0**, completion **17:37:10.786035+08:00**: `:23`, `:24`
- Observer-wrapper exit **0**: `logs/visual_update_analysis_waiter_20261001.json:13`

The preserved observer source launches the report only after complete acceptance and obtains the return code with `wait()`: `A/raw/.codex_tmp/wait_visual_update_complete_analysis_20261001.py:81`, `:87`, `:94`. I did not rerun it.

The actual prior code-review response is present at `logs/visual_update_launch_701_20261001/REPORT_CODE_REVIEW_RESPONSE.md:1`. Its limits explicitly exclude a completed-result audit at that earlier time (`:21`); it was not substituted for the current audit.

**Documentation qualification**

The canonical tracker remains a **dated 17:10 snapshot**, showing ten completed endpoints and two running endpoints: `refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER.md:3`, `:11`, `:17`, `:44`.

The identical local/remote handoff ends at the dated §41.707 partial record: `docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:13969`, `:13971`, `:13998`. Its “report invocation 0” statement belongs to that snapshot, before the actual report execution.

These are historical statements, not evidence of phantom completion. A new terminal tracker/handoff entry is still required to make the current documentation agree with the completed artifacts. Preserve the dated historical records.

**D. Executed metric paths versus dormant code — PASS**

The current entry executes the audited official metric path during every epoch and after strict selected-checkpoint reload:

`tools/run_visual_update_control.py:189` → `tools/run_correspondence_roles.py:79` → author `eval_func`/`eval_func_msrv`.

The saved-distance evaluation is called at `tools/run_visual_update_control.py:240`. The collector independently recomputes full-gallery scores at `tools/collect_visual_update_control.py:85`, `:97`.

The report calls `compare()` for each registered comparison at `tools/report_visual_update_control_complete.py:99`; that function consumes the actual distance files at `tools/analyze_correspondence_distances.py:28`.

Legacy checkpoint-saving functions that omit Signal exist in `tools/run_correspondence_roles.py:111` and `tools/run_correspondence_context_identity.py:44`. The current entry supplies its own full-state save/load functions, so those legacy omissions do not invalidate this panel.

Inherited M3/self-supervised prediction code is inactive here: the entry fixes `m3=False` and prediction weight zero at `tools/run_visual_update_control.py:71`; the active loss is CE plus triplet at `:167`. No teacher-generated target is used as retrieval truth.

**E. Scope and claims — WARN**

The actual experiment is **three datasets × four configurations × one adaptation seed**, with 50 epochs each. It is not a multiseed stability study or an untouched-test evaluation.

The plan explicitly retains the already trained ReID starting weights and acknowledges official-set epoch selection and historical method selection: `refine-logs/visual_update_control_v1/EXPERIMENT_PLAN.md:13`, `:17`, `:35`, `:58`. The current plan is byte-identical to the supplied dated plan.

The registered thresholds remain unchanged: `EXPERIMENT_PLAN.md:56`; `tools/report_visual_update_control_complete.py:24`, `:117`. The report correctly uses six visual comparisons for C1 and only three low-LR role comparisons for C2. Frozen-role comparisons are diagnostic.

| Registered comparison | Dataset | Δ mAP, pp | Δ Rank-1, pp | Result |
|---|---|---:|---:|---|
| Visual update, global-only | RGBNT201 | +1.249786 | +0.717703 | PASS |
| Visual update, roles | RGBNT201 | +1.344762 | +1.196172 | PASS |
| Visual update, global-only | RGBNT100 | +0.191517 | +0.116618 | PASS |
| Visual update, roles | RGBNT100 | +0.257943 | −0.116618 | **FAIL** |
| Visual update, global-only | MSVR310 | +0.987427 | +0.507614 | PASS |
| Visual update, roles | MSVR310 | +0.629945 | +0.676819 | PASS |
| Roles over global-only, low-LR | RGBNT201 | +0.109273 | +0.358852 | **FAIL** |
| Roles over global-only, low-LR | RGBNT100 | −0.021912 | −0.233236 | **FAIL** |
| Roles over global-only, low-LR | MSVR310 | +0.067307 | +0.169205 | **FAIL** |

Evidence: `O/REPORT.md:20` through `:31`; terminal keys at `O/SUMMARY.json:4091`.

Thus **C1 passes 5/6 component comparisons and fails overall; C2 passes 0/3 and fails overall**. The joint gate correctly remains **FAIL**.

**Meaning of the diagnostics**

Repairs/new errors count queries whose first legal correct match changes into/out of rank 1. AP changes use complete legal rankings. Identity statistics first average AP differences within each identity and then weight identities equally. The bootstrap resamples these fixed identity means with seed 42, 2,000 times: `tools/analyze_correspondence_distances.py:51`, `:57`, `:62`.

Consequently, the interval describes **fixed-model identity-macro AP differences**. It is not a training-seed confidence interval or a confidence interval for the query-weighted official mAP. All twelve reported bootstrap intervals include zero. That does not alter the registered gates, which do not use bootstrap thresholds.

The trajectory figure faithfully shows all epochs. Every endpoint ends below its selected best mAP. The records support that descriptive finding; they do not establish a unique cause or population-level training instability.

**Cost definitions**

The recorded costs are internally consistent:

- Campaign wall time: **12,773.422633 s**
- Summed endpoint wall time: **44,372.372088 s**
- Summed training-plus-epoch-evaluation receipt time: **42,880.143614 s**
- Summed training-step time: **19,406.389737 s**
- Logical bytes across the 12 M0 and 12 full output directories: **8,754,640,902**
- Report-time free disk: **84,166,303,744 bytes**, observed at **17:37:07.835817+08:00**

Evidence: `tools/report_visual_update_control_complete.py:86`, `:119`, `:161`; `O/SUMMARY.json:4094`, `:4097`, `:4125`.

These are observed elapsed times under the actual concurrent execution, not calibrated FLOPs or GPU-active-time measurements. Memory is peak allocated usage after initialization, including epoch evaluation; it excludes initialization transients and reserved memory. Upstream ReID training costs are excluded. The two preflight M0 intervals have explicitly different parent-observed timing semantics.

**F. Evaluation classification — PASS**

- Official retrieval and post-selection AP/flip diagnostics: **`real_gt`**.
- M0 gradients, initialization matching, state immutability and strict reloads: **engineering checks**, not retrieval-performance evaluations.
- Identity bootstrap: a descriptive resampling analysis of fixed-model `real_gt` outcomes.
- No current reported result requires classification as synthetic proxy, self-supervised proxy, simulation or human evaluation.

Evidence: `tools/run_correspondence_roles.py:88`; `tools/run_visual_update_control.py:204`; `tools/check_visual_update_initialization.py:49`; `tools/analyze_correspondence_distances.py:75`.

**Claim implications and remaining actions**

- **Supported:** twelve independently trained, complete 50-epoch endpoints; correct single-checkpoint metrics; matched initialization and FP32 storage; correct legal-gallery scoring; unchanged registered gates; genuine recorded report completion.
- **Supported with scope qualifiers:** ordinary visual fine-tuning improves mAP in all six comparisons and meets the complete global-only comparison set.
- **Unsupported:** registered overall C1 success, C2 role-increment success, joint success, stable multiseed gains, untouched-test significance, algorithmic novelty from fine-tuning, or completion of the three-dataset baseline/SOTA objective.
- Append a terminal tracker/handoff entry with the 12/12 status, actual report exit and failed gates. Preserve prior dated snapshots.
- Retain remote checkpoint/array access and its provenance. The local checkout contains **88 exact bound files, 27 LF-equivalent files and 107 absent bound files**; all 222 remote originals passed. A local clone alone cannot reproduce this audit.
- Preserve the full reviewer trace and same-family/provisional attribution. No further neural execution is needed to support this verdict.

Key verified bindings:

| Artifact | SHA-256 |
|---|---|
| Campaign manifest | `947b028e835a6c722fac47f5e6333f7f02838ec9b1fbd2beda35ec1885543e5c` |
| Accepted matrix | `45ce06b471c074ed86ec5617e71892932a032f8c22d4e8230c6e5087b5813889` |
| Original report source | `4c21b1cc6212ef0202512ef46afc8fc714e3b2329fd68f701e4a7ac24acd4765` |
| Original SUMMARY.json | `43634cf0c6acd7b998d9c9c2e39174bd60141598c4f4b9f1ba14d81fea6b18ad` |
| Original REPORT.md | `41d6d3b8d29eec6e9db77bc74e5b503047a50f55fbab6c22a8cebc54998dc5ea` |
| Audited local/remote handoff | `71816df830851f61d13e19297402fcb0b359919eb546d272c3c0c1d1bbc1f2ba` |
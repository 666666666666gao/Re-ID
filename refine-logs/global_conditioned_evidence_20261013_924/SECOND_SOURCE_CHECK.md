# Second endpoint 929 — narrow cold source check

**Source PASS; overall WARN for provenance labels and bounded scope.**

`source_ready_for_second_cpu_report=true`; `unresolved_blocking_source_defects=[]`.

Requested gpt-6-astra/max; actual model/effort **UNATTESTED**. Same-family/provisional. This is a narrow continuation of the earlier reviewer, not a fresh whole-source or tensor audit.

Current byte pins:

- Generic report: `0884d747e4d4719a87894bf5e0091a214995f2ddc04a9c551e50914d53e2accb`.
- Random CPU wrapper: `3feb920e0b67353a3c36970f87d8be1688c1d7f7174d682aea36f00654ed56d2`.
- Generated random collector: `92e2b1f1ec3bb911c44ed997549c22386f3b72d480093a35ad2e0c25e3e41820`.
- Preparation script: `c2b4c7fd5276ea6defbc4e3a41ccc91de797fbc0289bc66a0613822eea7e9c7d`.

References are relative to `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903` unless otherwise stated. Full absolute hashes and exact JSON key line numbers are in SECOND_SOURCE_CHECK.json.

## reviewed_report_binding: PASS

Generic report bytes remain unchanged at the audited 0884... SHA256 and match both the preparation record and the original audit hash at the explicit supplied C-path key. The corrected first wrapper bytes remain unchanged at the audited ed18... SHA256; no source replay or edit was needed. The random wrapper retains the exact SHA assertion before SSH.

Evidence: `run_evidence_random_cpu_report929.py:5-12`; `EVIDENCE_RANDOM_CPU_PREPARATION_929.json:3`; `SOURCE_AUDIT.json:1`.

## random_scope_and_cpu_only: PASS

Random wrapper text is exactly the stated replacement of the corrected reviewed gradient wrapper. Only random/uppercase filenames and fresh 929 CPU output/admin naming changed. CLI uses --exposure random; new remote admin is logs/evidence_random_cpu_report929 and must not exist. Only the private report copy is written there. CUDA_VISIBLE_DEVICES is empty, OMP/MKL threads are 1; no model/training command, source hot edit or retry branch is introduced.

Evidence: `prepare_evidence_terminal_analysis929.py:12-15`; `run_evidence_random_cpu_report929.py:17-35,37-48`.

## terminal_input_routing: PASS

Supplied terminal snapshot reports COMPLETE_TWO_FULL50_FIRST_STRICT, controller/observer exit0 and both supervisor/worker not live. It has exactly gradient then random accepted endpoints; random train/evaluate stages both exit0. Collection accepted_endpoint is exactly the random entry and its official-metrics hash matches the provided file, whose exposure is random, with full50 and seed42. This validates the intended second endpoint routing; it does not independently re-evaluate the numerical result.

Evidence: `prepare_evidence_terminal_analysis929.py:5-10`; `EVIDENCE_TERMINAL_OBSERVATION_929.json (exact key lines in artifact_line_references)`; `EVIDENCE_TERMINAL_OBSERVER_EXIT_929.json:1-4`; `evidence_random_results929/COLLECTION_RECEIPT.json (exact key lines in artifact_line_references)`; `evidence_random_results929/output/official_metrics.json:3-35`.

## preparation_write_boundary: PASS

Preparation reads the terminal snapshot, clones the local collector/wrapper, AST-parses strings and writes only these private generated files plus a preparation record. It makes no SSH/model/report call and changes no frozen production source. The generated collector adds terminal campaign/exit receipts and routes accepted_endpoint by exposure random.

Evidence: `prepare_evidence_terminal_analysis929.py:5-21`; `collect_evidence_random929.py:5-9,17-29`.

## normalized_text_vs_file_hashes: WARN

Preparation wrapper_sha256/collector_sha256 hash generated LF strings at .encode(), but Windows write_text emitted CRLF. These values are normalized-text hashes, not current raw file-byte hashes. Both normalize exactly to their recorded values; all outer/embedded AST parses pass. This audit pins actual current bytes separately. No report hash mismatch or semantic source deviation exists; do not mislabel prep hashes as raw file hashes or rerun a collector/report to fix provenance wording.

Evidence: `prepare_evidence_terminal_analysis929.py:11-17`; `EVIDENCE_RANDOM_CPU_PREPARATION_929.json:3-5`.

## historical_template_label: WARN

Collector docstring/print still says first endpoint/FIRST_ORIGINAL_ENDPOINT_COLLECTED. The actual selection, paths, exposure, fresh 929 names and accepted receipt identify the random second arm. Treat the wording as a retained template label; no collection rerun is needed.

Evidence: `collect_evidence_random929.py:1,9,27-31,58`; `evidence_random_results929/COLLECTION_RECEIPT.json (accepted_endpoint)`.

## exit_and_failure_capture: PASS

Same reviewed pattern waits for the CPU child, preserves remote child log/EXIT, writes local raw stdout/stderr and transport EXIT before parsing, and retains received child receipt/log before asserting success. No prior gradient report or failure evidence is overwritten.

Evidence: `run_evidence_random_cpu_report929.py:29-47`.

## Receipt and claim boundary

The supplied snapshot has two accepted original endpoints, terminal controller exit 0 and no live supervisor/worker. The random receipt matches the random accepted endpoint and full50/seed42 exposure. These facts establish the source route for the second CPU call. No metric recomputation, tensor load, earlier report replay, model/SSH/report execution or frozen-source edit was performed.

The preparation receipt records LF-string hashes (`a8a4e422884711c97fb0180d05cc59d7291484852cf9a537948829f56a611491` wrapper; `918f0fff17ac2db425007a884023db637be9978e3b831aafcd7b6eccd1646dc1` collector). Actual CRLF-byte hashes are pinned above. Keep this distinction when publishing provenance. Retain the historical FIRST_ORIGINAL_ENDPOINT_COLLECTED print label with its qualification; do not rerun to correct wording.

No unresolved source blocker remains for this one CPU-only random report. This does not approve training, certify the future CPU result, or establish scientific/full-goal success.

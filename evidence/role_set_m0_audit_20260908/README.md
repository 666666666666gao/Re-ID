# Independent M0 audit reproduction and artifacts

The scientific repository and remote run are read-only inputs. All local output paths are restricted to this directory. Credentials are obtained privately from the pre-existing helper; its contents are not included.

## Main outputs

- EXPERIMENT_AUDIT.md: complete independent reviewer analysis with exact source/receipt line evidence.
- EXPERIMENT_AUDIT.json: verdict, per-check statuses, claim impacts, boundaries and audited-input hashes.
- reviewer_full_response.md and trace/001-independent-audit.response.md: identical verbatim reviewer response.
- independent_arithmetic.json: all 248 saved-array/label/queue/loss step recomputations and 780 independently generated source batches.
- independent_checkpoints.json: three source checkpoint bindings, six terminal state reconstructions and 37 CPU receipt file hashes.
- supplementary_checks.json: label origin, exact tie derivatives, 12 synthetic mean-hinge/gradient cases and synthetic VJP.
- remote_inventory.json / local_inventory.json / snapshot_manifest.json: inventory and hashes.
- binding_and_launch_checks.json / local_report_checks.json: launch bytes, closed exits, logs, reports and resource accounting.

## Exact execution sequence

Local Python:
`C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe -X utf8`

Remote transport launcher:
`C:/Users/gb/AppData/Local/Programs/ClawX/resources/bin/uv.exe run --offline --with paramiko python -X utf8 remote_transport.py <script> <label>`

Run local inventory.py; run remote_transport.py with remote_inventory.py / remote_inventory; run local unpack_remote.py. Then execute independent_arithmetic.py / independent_arithmetic, independent_checkpoints.py / independent_checkpoints, supplementary_checks.py / supplementary_checks and binding_and_launch_checks.py / binding_and_launch_checks via remote_transport.py. Remote commands and stdin script SHA are preserved separately in each *.command.json. These scripts perform only CPU metadata, array arithmetic and safe checkpoint tensor deserialization, with CUDA hidden and two threads.

Run local finalize_checks.py. If reproducing the original newline issue, run fetch_raw_text_bytes.py through the offline Paramiko launcher; it compares the existing snapshot to the raw hash manifest and retrieves only mismatching source text bytes. Then run finalize_checks.py again. Finally run write_reports.py. See the original failed attempts below; the final audit did not accept mismatching snapshots.

## Preserved attempted-check failures

1. attempt_01_transport_failure.txt: the private helper called stdout.reconfigure, while the initial audit transport redirected to StringIO. Replaced the private output sink with a real null TextIOWrapper; no credentials emitted.
2. attempt_02_remote_inventory.command.json / stderr.txt / stdout.txt: initial optional mapping loop accessed signal_source on a configuration without that binding. Corrected the inventory loop; no scientific source changed.
3. attempt_03_snapshot_failure.txt and attempt_03_normalized_snapshots/: pathlib read_text normalized 22 remote CRLF text files while serializing snapshots. Their original remote byte hashes were already correct. raw_text_snapshot_recovery.json records binary SFTP receipt, full SHA/length matches and exact byte snapshots. Final remote_snapshot_byte_mismatches is empty.

Some shell rg searches returned exit 1 for no matches; these were discovery no-hit results, not experiment failures. No blocked check was converted into PASS.

## Scope limits

The scripts do not reconstruct any real-model forward/backward; no module instance is created. GPU gradients, original RNG/cache/augmented images, strict reload output arrays and initial role state are not saved. Existing assertions remain runtime witnesses. Official image content, partial Q1 output and ongoing training are deliberately outside this audit. Current package versions are observations, not retroactive proof of the original backend.

Final packaging attempt: attempt_04_report_packaging_failure.txt records the LF-string versus written-CRLF verdict hash mismatch. attempt_04_reference_validation.json preserves draft line-range problems. The writer now hashes actual written report bytes, and the final artifact_validation.json checks all source reference ranges, script ASTs and verbatim response/trace identity. These are audit packaging corrections only.

# Reviewer operational record

All paths below are reviewer outputs or the supplied immutable input bundle. No input, experiment config, home memory, remote system, image, checkpoint, tensor or model was modified or executed.

The executable actually used for successful local reads and replay was:

`C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe`

The default `python` command resolved to `E:\Scripts\python.exe` and failed before the inspection script ran with `No pyvenv.cfg file`. Interpreter discovery used `uv python list --only-installed`. No package was installed. This environment setup error is unrelated to an experiment result.

`audit_read.py` initially logged its first six successful invocations with the abbreviated prefix `python` rather than its actual interpreter path. Later invocations used Python `repr` for the Windows paths, which doubles backslashes in the helper's parameter log. `audit_commands.txt` preserves that original log; it is not claimed to be a raw shell transcript. `audit_exact_commands.ps1` records the exact commands used for the numeric replays, including stderr capture. All numerical conclusions come from those saved scripts and their stdout/JSON outputs.

The initial reader stdout used the Windows default encoding, so early Chinese tool display was garbled; the saved text was Unicode. The reader was set to UTF-8 stdout and all relevant prose documents were reread. Its current source is `audit_read.py`, and numbered text/AST/diff outputs are preserved in `audit_read_*.stdout.txt`, `audit_compact_*.stdout.txt`, `audit_schema_*.stdout.txt` and `audit_diff_*.stdout.txt`. The full 153-file byte/text pass and subsequent all-JSON/all-AST pass independently cover the whole manifest. Large arrays were traversed completely in the replay, not treated as the displayed first schema element.

One exploratory reader invocation passed the JSONL file 44 to the single-JSON schema action:

`audit_read.py schema 136 44 64 65 30 35 42 90`

It stopped with `json.decoder.JSONDecodeError: Extra data: line 2 column 1 (char 724)`. No input was changed. The following successful schema read omitted 44, and `audit_replay.py` explicitly parses every one of its 34 lines as JSON. This was an auditor reader error, not an experiment failure.

Numeric replay versions and outputs:

1. `audit_replay.attempt1.py` is the exact first numeric replay source. `audit_replay.attempt1.stdout.txt` preserves its `KeyError: 'path'`: M0 intake entries use `remote`/`local` and the failure intake uses `copied_files`; this reader had assumed the other intakes' `path`/`files` fields. The final `audit_replay.py` explicitly handles those observed schemas. It does not introduce an experiment fallback. `audit_replay.attempt2.stdout.txt` and `audit_replay.stdout.txt` record 973/973 checks passing. All partial output JSON was regenerated from the unchanged inputs by the final successful script.
2. `audit_stage_replay.attempt1.py` and `audit_stage_replay.attempt1.stdout.txt` preserve a failed exact comparison of a reported *double regrouping error*. The independent expression `abs(sum(heads) + .1*gram + .1*patch - loss)` differs from the saved number by about 3.1e-16. The final script separately calculates both ordinary parenthesizations on every row; `abs(loss - sum(heads) - .1*gram - .1*patch)` reproduces the reported 1.1775642632990552e-6 exactly. It records both sets of errors and does not increase a tolerance or alter a numerical/scientific gate. `audit_stage_replay.attempt2.stdout.txt` and `audit_stage_replay.stdout.txt` record 21/21 checks passing. All original float32-order loss recompositions already passed exactly in the first complete numeric replay.
3. `audit_write_reports.py` emits this reviewer's Markdown and structured JSON, checks that cited source line ranges exist, and rehashes every immutable input before and after writing. `audit_final_hash_recheck.json` is the final full input hash list. `audit_report_write.stdout.txt` identifies final report hashes.

The 994 explicit checks are the 973 successful complete replay checks plus 21 successful additional stage/number checks. Their count is bookkeeping, not a scientific test score. Tensor arithmetic, physical dataset provenance, remote checkpoint contents and GPU execution cannot be independently reproduced from these text inputs, and the reports say so.

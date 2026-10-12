# Original failures and limits retained

## Original production stop (not executed by reviewer)

Source: `evidence_snapshot/intake/logs/region_use_resume906/EXIT.json`, SHA `7d7e8e4fc7c3d4f86d99fb922313701fe84236121a62b131ecc9a71d0e73d95d`.

Supervisor 1708681, controller 1708684, original command `python -B /data/gaob/Re-ID/Trifusion/tools/queue_region_use_probe.py --campaign /data/gaob/Re-ID/Trifusion/logs/region_use_v1_20261012_906`; saved exit_code 1 at 2026-10-12T23:57:40.989290+08:00.

Saved traceback from `FIRST_100_NODE_909.json:2730`:

```text
Traceback (most recent call last):
  File "/data/gaob/Re-ID/Trifusion/tools/queue_region_use_probe.py", line 136, in <module>
    raise SystemExit(coordinate(parser.parse_args().campaign.resolve()))
  File "/data/gaob/Re-ID/Trifusion/tools/queue_region_use_probe.py", line 106, in coordinate
    assert shutil.disk_usage(ROOT).free >= RESERVE_BYTES+NEXT_ARTIFACT_BYTES
AssertionError
```

The original campaign remains unchanged and stale RUNNING. Fifth four-stage completion and sixth PENDING/empty stages are preserved; the failure is not converted into a model-quality failure or successful six-of-six completion.

## Reviewer-local command failures

1. A PowerShell inventory command attempted to pipe a `foreach` statement directly into `ConvertTo-Json`, causing parser error at its closing `} | ConvertTo-Json` (tool chunk `558f09`, exit1). The corrected inventory first assigned the loop output to `$items`, then converted that value. This was a read-only inventory command, not controller execution.
2. The ambient command `python -B -` resolved to `E:/Scripts/python.exe` and failed with exact output `No pyvenv.cfg file` (tool chunk `9d950d`, exit1). `uv python find --no-python-downloads` identified the existing local interpreter `C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe`; subsequent stdlib checks explicitly used it. No installation or network request was made.
3. Initial path reads of `P/trifusion/region_use_objective.py` and `P/trifusion/region_use_heads.py` failed because those paths do not exist (tool chunk `2b9599`, exit1). `rg --files` identified the real paths under `P/modeling/trifusion/`; those exact files were then read. No production file was changed.
4. Broad initial read-only output requests for an observation/protocol and a master regex exceeded tool display budgets. The checks later read exact files as stdlib JSON and the master scope was retained as the bounded §41.910 extract only. Truncated display output was not treated as a complete read or verification result.

## Raw local source hash check remains FAIL

`DETERMINISTIC_STDOUT_001.txt`, `DETERMINISTIC_CHECKS.json`, and `CHECK_LEDGER.jsonl` retain the first completed check result: 27,787 assertions; 27,653 passed; 134 failed. Every failed assertion is a raw-source pin check against the publication checkout: 27 CRLF-only differences, 107 locally absent files. The diagnosis is in `SOURCE_HASH_LIMITS.json`, with raw and LF-derived SHA for each differing file and exact missing path names.

No check output was rerun or overwritten to hide these failures, and no producer/manifest/experiment result was changed. All other 27,324 text/hash/AST checks passed. Source-byte normalization was used only as an in-memory explanatory calculation; it does not satisfy the original raw-byte pin and was never applied to a file. The local publication checkout is not a complete remote source/artifact admission environment.

## Attribution and execution provenance

This reviewer was requested as Astra/max, but the serving model and effort are UNATTESTED. The exact initial delegated prompt is retained in the trace request. Parent-supplied operational updates are not independent verification by this reviewer. No additional reviewer or external model call occurred. The stdlib checker source existed before its invocation and the original output is saved. Supplemental hash diagnosis was an inline stdlib command whose transcript is preserved with the review; it does not represent a historical native source check.

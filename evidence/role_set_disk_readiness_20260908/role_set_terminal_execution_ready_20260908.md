# Role-set Q1 terminal execution readiness

Prepared 2026-09-08 21:05 Asia/Shanghai. This is preparation, not a terminal result.

The original remote Q1 and automatic CPU verifier continue. Next planned remote observation: approximately 21:25, near the expected completion of fold 1 role_set. Last recorded observation is 20:55:05: three of six endpoints complete, fold 1 role_set epoch 5, free artifact storage 10,418,761,728 bytes.

Execution commit: 26c97390704c629237687d263b4381f5584cbe97.
Published documentation commit: 631167495cb4749ee44cd32d0cb3077a83d00953.

## Local runtime verified

Use the existing offline uv cache by invoking:

```powershell
& 'C:/Users/gb/AppData/Local/Programs/ClawX/resources/bin/uv.exe' run --offline --with numpy --with paramiko python -X utf8 SCRIPT ARGUMENTS
```

Imports succeeded with NumPy 2.5.3 and Paramiko 5.0.0. The ranking replay script's `--help` also succeeded. Do not bind subsequent work to the temporary Python executable returned by uv: the earlier temporary executable from the history-gradient run no longer exists. No experiment code was changed.

## Terminal sequence

1. Observe the original pipeline. Wait for all six fixed endpoints and CPU verification. An engineering error requires investigation, not bypassing the intake assertions.
2. Run `receive_role_set_q1_terminal_20260908.py NEW_INTAKE_DIRECTORY`. It requires all original processes to have exited, all five stages to have exit code zero, a complete pipeline status, the correct code binding, 1560 verified updates, and matching terminal hashes. It receives only text and validates every received file's bytes and SHA-256.
3. Run `analyze_role_set_q1_terminal_v2_20260908.py INTAKE_DIRECTORY NEW_OUTPUT_JSON`. Use v2, retaining the distinction between the 65-step warmup and the period before the first historical candidate. Do not substitute the earlier analyzer.
4. Run the existing repository tool `tools/audit_msvr_paired_ranking_text.py` with `--input-dir INTAKE_DIRECTORY --candidate role_set --summary-sha256 ACTUAL_VERIFIED_TERMINAL_SUMMARY_SHA --cpu-status PASS_COMPLETE_ROLE_SET_Q1 --output-dir NEW_RANKING_DIRECTORY`.
5. After the complete files exist, carry out the required fresh Q1 integrity review and archive its actual response. The executor ranking replay is not an independent reviewer or a model/distance recomputation.
6. Publish the complete result and evidence boundaries, then synchronize remote/GitHub/Desktop hashes. Preserve the registered gates regardless of outcome.

No role-set Q1 terminal intake, source analysis, or ranking replay has yet been executed. M0 review is closed and must not be repeated. This file does not change the training contract, the next experiment, or any scientific gate.

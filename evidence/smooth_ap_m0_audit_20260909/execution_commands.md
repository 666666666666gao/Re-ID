# Exact deterministic execution commands

All scripts and outputs are local to this audit directory. The remote payloads write only stdout/stderr.

## remote_inventory

```powershell
& 'C:\Users\gb\AppData\Local\Programs\ClawX\resources\bin\uv.exe' run --offline --with paramiko python -X utf8 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_readonly.py' 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_inventory.py'
```

Remote stdin command (script source supplied over standard input):

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/tri_reid/bin/python -B -u -
```

Exit 0; wall time 6.657444953918457 s. Raw stdout/stderr retained as remote_inventory.stdout/.stderr.

## remote_arithmetic

```powershell
& 'C:\Users\gb\AppData\Local\Programs\ClawX\resources\bin\uv.exe' run --offline --with paramiko python -X utf8 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_readonly.py' 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_arithmetic.py'
```

Remote stdin command (script source supplied over standard input):

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/tri_reid/bin/python -B -u -
```

Exit 0; wall time 31.477286338806152 s. Raw stdout/stderr retained as remote_arithmetic.stdout/.stderr.

## remote_provenance

```powershell
& 'C:\Users\gb\AppData\Local\Programs\ClawX\resources\bin\uv.exe' run --offline --with paramiko python -X utf8 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_readonly.py' 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_provenance.py'
```

Remote stdin command (script source supplied over standard input):

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/tri_reid/bin/python -B -u -
```

Exit 0; wall time 16.770435333251953 s. Raw stdout/stderr retained as remote_provenance.stdout/.stderr.

## remote_source_checkpoint_bindings

```powershell
& 'C:\Users\gb\AppData\Local\Programs\ClawX\resources\bin\uv.exe' run --offline --with paramiko python -X utf8 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_readonly.py' 'C:\Users\gb\.codex_tmp\smooth_ap_m0_independent_audit_20260908\remote_source_checkpoint_bindings.py'
```

Remote stdin command (script source supplied over standard input):

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/tri_reid/bin/python -B -u -
```

Exit 0; wall time 3.6590468883514404 s. Raw stdout/stderr retained as remote_source_checkpoint_bindings.stdout/.stderr.

Local preparation / inspection scripts were invoked with `C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe -X utf8 <absolute script path>`: collect_local.py, unpack_inventory.py, summarize_arithmetic.py, finalize_evidence.py, validate_snapshots.py, restore_exact_text_snapshots.py, write_report_metadata.py. Each complete source is retained. The failed transport attempt command is preserved in attempt_01_transport_failure.txt.

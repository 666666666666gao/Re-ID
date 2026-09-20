# Preserved local runtime discovery errors

1. E:\python.exe importing paramiko,numpy: ModuleNotFoundError: No module named 'paramiko'.
2. C:\Users\gb\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe importing paramiko,numpy: same missing paramiko.
3. E:\anaconda\python.exe importing paramiko failed with ImportError: DLL load failed in cryptography.hazmat.bindings._openssl. No packages or environments were modified.
4. Prior trace identified the established ClawX uv offline runtime route. Subsequent transport uses that existing route.

5. Two initial reads expected trifusion/aligned_data.py at the repository root; the actual module is modeling/trifusion/aligned_data.py, resolved from the builder sys.path. No source change.
6. One rg command supplied wildcard strings as literal Windows paths and returned os error 123; repeated as rg with directory and -g scope, locating build_msvr310_train_oof_protocol.py.
7. R1 independent checker exited 1 on NumPy int64 summary serialization before arrays; full traceback, scripts and terminal receipt preserved in remote_artifacts/failed_attempt01. R2 only casts three summary counts to int, with exact change receipt.

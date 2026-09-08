# Audit execution notes

These notes describe tooling corrections only. They did not change experimental files or numerical definitions.

- The default `python` shim failed with `No pyvenv.cfg file`; the existing `E:/python.exe` standard-library runtime worked but had no Paramiko.
- A parent-provided temporary uv interpreter had already disappeared when invoked later. A persistent existing cached environment at `D:/Program Files/UserCache/gb/uv/archive-v0/CGsVGhfI7mHn4w3k51V05/Scripts/python.exe` was located and verified to contain Paramiko 5.0.0. It was used without installing packages. `E:/python.exe` handled local stdlib-only processing.
- Silently executing the supplied SSH recovery script initially encountered `AttributeError: '_io.StringIO' object has no attribute 'reconfigure'`. The output sink was adjusted to support the actually requested `reconfigure` call while keeping recovery stdout/stderr in memory. No recovery-script contents, output or credentials were copied into the audit directory.
- The remote input inventory completed with exit 0, zero stderr, and a saved full stdout/receipt. The independent full-array replay completed once with exit 0, zero stderr, in 89.697854 seconds of remote script time. No numerical replay was repeated.
- The local summary formatter initially failed to match Windows backslash snapshot paths against slash prefixes; the path representation was corrected. A guessed expected primary-document count of 16 then failed because AGENTS.md had only been snapshotted locally. The actual 15 local/remote document pairs were enumerated, compared by bytes and hashes, and asserted. This corrected the audit's transport-binding summary, not the remote numerical replay or scientific result.
- Five supplementary original feature/forward mapping source files were read and independently matched against both their historical commit and current local bytes. No model code was imported or run.
- All reported numerical conclusions come from direct source/label/array evidence. A historical memory note was used only to locate an existing local stdlib Python executable, and that executable was verified live.

The full successful remote scripts, their raw stdout/stderr and execution receipts are retained. The complete reviewer response is duplicated verbatim in the trace response. No agent UUID or actual backend attestation was available, and none is invented.

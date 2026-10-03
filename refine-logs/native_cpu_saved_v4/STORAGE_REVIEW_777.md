# Native CPU saved-tensor v4: storage-only source review

**PASS — SOURCE_ONLY.** No blocking issue, nonblocking code defect or required fix was found in the storage correction.

This is a fresh-context experiment-bridge review requested as `gpt-6-astra` / `max`, **same-family / provisional**. Backend model and effort were not independently attested. The baseline is `2d2e2147cc943be7d666dde0bb0d5933a3a67a0f`. The scope is the new storage paragraph and `tools/queue_native_cpu_saved.py` diff; unrelated dirty files were ignored.

The original full-source review remains at **`refine-logs/native_cpu_saved_v4/EXPERIMENT_CODE_REVIEW.md` and `.json`**, attributed to `/root/review_native_cpu_saved776`. Neither original file was changed or renamed. This file supplements that review with the storage change and confirmation that the numerical source is unchanged.

## Findings

- **All output consumers agree.** `output_dir` returns a distinct child of `/home/gaob/trifusion-native-evidence-v4` for each of the 18 phase/dataset/variant combinations. `configure()` assigns it to `base.output_dir`; worker commands and the foundation `verify_m0`/`verify` functions therefore use the same directory. Source references: queue lines60–61,86,228–232; foundation queue lines70–108.
- **prepare remains correct.** Its existing `native_prepare_unused` argument does not create an output directory: the prepare branch writes initialization JSON to the campaign. M0 and training create their passed output directory; evaluate and strict reload read it. Weights, receipts, batch order and distances use that same absolute directory. The existing distance-file `os.replace` stays within it. References: independent entry lines187–213; foundation training/evaluation lines189–312.
- **The report follows the new path.** It calls `panel.configure()` before source and strict result verification, then reads the verified `row.run_dir`. The paired distance analyzer also reads `row.run_dir`. References: report lines22–34,43–49; distance analyzer lines28–38.
- **The storage correction matches the actual evidence.** The saved 15:56:40 resource receipt shows different devices: `/data` device2048 with9,491,521,536B free, `/home/gaob` device66306 with591,852,425,216B free. The former is below the unchanged10,292,822,016B campaign threshold. The new root must be absent; after creating it, the coordinator checks that threshold on its actual filesystem and records `output_root` in the manifest. The unchanged2,147,483,648B reserve is independently checked on both filesystems before job dispatch and each worker mode. References: queue lines35–37,93–94,140–141,169–173,199–203.
- **The source and numerical contract stay intact.** `source_map`, `SOURCES`, command arguments, extended M0 verification and fixed numerical gates are unchanged. The inherited293 entries plus11 listed sources with3 overlaps still yield301 paths. Queue/plan hashes change as expected; source-map and initializer-hash comparisons retain their existing semantics. The original review MD remains byte-identical. All9 extended M0 verifications still precede formal training. No data/model/loss/head/optimizer/batch/precision/seed42/full50 change, fallback, symlink, move, deletion or new flag is introduced.

## Checks actually performed

Reviewed the scoped diff and caller source, parsed11 relevant Python files with `ast`, and compared10 unchanged reference files to the baseline with CRLF transport accounted for. A stdlib-only check executed extracted path/configuration function definitions:18 distinct output directories,27 M0/train/evaluate command constructions, correct foundation-verifier global rebinding and report call order. It wrote no files and imported no model or torch.

The local archived original-repeat sources matched all293 existing recorded digests; their union with the unchanged11-entry source tuple matched the retirement receipt's301 keys. This verifies the local archive and closure rule, not live server deployment.

Both baseline and current queue bytes use CRLF throughout (225/237 lines). The raw diff check flagged those added CR characters; the established `cr-at-eol` whitespace check passes. A first bare-Python invocation failed before running the script (`No pyvenv.cfg file`); checks then passed with the existing uv-managed CPython3.12.12 executable and `-B`. No environment, package or source was changed.

## Limits

No server connection, launch, torch/model import, checkpoint load, GPU operation, training or real report execution occurred. The resource receipt proves only its recorded boundary, including v4 campaign/launcher absence. Actual output-root creation/permissions, future free space, CPU/GPU numerical parity, full B128 fit and every runtime initializer/pair/backward/M0 gate remain to be measured. **This PASS does not establish a v4 runtime PASS, full50 result, scientific gain or SOTA.**

Required fixes: **none**.

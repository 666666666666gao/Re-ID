# Signal public release and RGBNT100 comparison boundary — 2026-09-20

Live GitHub metadata reports HEAD `cd1b0a672d1fe642e7608731cb4899a19dda7d51`, unchanged from the pinned project comparator. The fetched RGBNT100 YAML is byte-identical to the archived 2026-09-06 copy (SHA256 `02641ce2941dcd50df77d379aab54650cae4a2a6b15ba1bcd43deba970b6900f`). This is a public-source refresh, not a reproduction or a new checkpoint result. [Commit metadata](https://api.github.com/repos/010129/Signal/commits/main), [pinned YAML](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/configs/RGBNT100/Signal.yml).

The published release table remains RGBNT201 80.3/85.2, RGBNT100 86.3/97.6 and MSVR310 53.2/72.4 (mAP/Rank-1). This README row must remain distinct from the previously verified MSVR310 paper-table row 53.6/71.9. Do not combine the best numbers from different rows. [Pinned author README](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/README.md).

| Configured RGBNT100 condition | Author release | Local fixed main run |
|---|---:|---:|
| Batch size | 128 | 64 |
| Instances per identity | 16 | 8 |
| Identities per P-K batch, derived B/K | 8 | 8 |
| Epoch budget | 30 | 30 |

Author columns come from the pinned YAML above. Local columns come from `configs/RGBNT100/Signal-main-v1.json`, which binds `Signal-source-oof-v1-r2.json`; both are existing registered inputs, unchanged here. The equal B/K count is a configuration-level arithmetic inference: a larger author batch does not, in this comparison, imply more distinct negative identities per batch. The differing K can affect instance relationships; no causal retrieval effect is established by this arithmetic.

The completed local report `results/TRIFUSION_RGBNT100_OFFICIAL_COMPARISON_2026-09-06.md` already records the broader recipe differences: local shared geometry and fixed epoch30 versus separate modality augmentation and per-epoch best selection in author code, plus the disclosed Gram numerical definition. These are not new discoveries or evidence for assigning the full 5.587838-point public/local baseline gap to any one factor. Equal epoch counts also do not establish equal updates, examples or compute. No upstream complete training receipt was acquired in this refresh.

Retain the verified local matched benefit +2.572608 mAP separately from public-report differences. No consumed official results are used to tune batch, augmentation, checkpoint, fusion or loss. No runtime/configuration was changed, no model/array/image was downloaded, and no experiment was launched. `receipt.json` records current URLs, hashes and the exact local config binding; fetched raw texts stay in the temporary research folder.

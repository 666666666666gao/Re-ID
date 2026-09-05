Completed the bounded V22 fixed-initialization audit and wrote only the two requested files:

- `C:\Users\gb\.trifusion_github_publish_22c3bee\EXPERIMENT_AUDIT_V22_INITIALIZATION.md`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\EXPERIMENT_AUDIT_V22_INITIALIZATION.json`

Verdicts:

- Overall verdict: `WARN`
- Integrity status: `warn`
- Engineering integrity: `pass_with_provenance_limitations`
- Scientific qualification: `fail_to_promote_descriptive_only`
- Evaluation type: real dataset identity/camera GT on reused train-internal complete-path OOF full-gallery diagnostic, not independent validation

Report hashes:

- `EXPERIMENT_AUDIT_V22_INITIALIZATION.md`: `5a4c4e45e80bbd5fc1af7e481a1677149d3cbb6532859182874393dfe1b8dedf`
- `EXPERIMENT_AUDIT_V22_INITIALIZATION.json`: `8c9d54948b505216d36740d07bbf2ec08589f0238086d3486ea977d530cdaa35`

Main findings:

- No evidence found of fake GT, self-normalized metrics, phantom aggregate values, hidden optimizer steps, checkpoint selection, hidden dev/official access, or metric arithmetic drift.
- Local recomputation matched masks, fold metrics, aggregate metrics, 21 identity rows, paired AP/Rank-1 change counts, and terminal-minus-initial mAP deltas. Maximum full-precision numeric difference: `2.842170943040401e-14`.
- The markdown result table matches JSON within six-decimal rounding. Largest observed rounding difference: `4.958391315312838e-7`.
- V22 terminal Q1 remains failed. This audit does not change Q1 qualification, does not authorize D1/dev/official evaluation, and does not select an initialization checkpoint.

Limitations recorded in the reports:

- I did not load models, checkpoints, tensors, images, GPU state, remote files, or run retrieval/inference/training.
- Remote V12 checkpoint files, CLIP weights, endpoint receipts, and image bytes are ledger-verified only in this local bounded audit.
- Metric recomputation starts from saved AP/Rank arrays and manifests, not from raw images or embeddings.
- The diagnostic is reused train-internal OOF evidence, not a new independent validation or official-test result.
- The derived comparison JSON stores paired-change count summaries rather than explicit per-query delta rows, though the raw initialization and terminal JSON files retain enough arrays to recompute those summaries exactly.
- Local `protocols/rgbnt201_dev_v1.json` is CRLF with raw hash `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d`; LF-normalized local bytes match the remote/source ledger hash `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946`, so that is a line-ending possession distinction, not a semantic protocol mismatch.

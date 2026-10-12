# Endpoint907 source integrity review R2

Verdict: **SOURCE-ONLY PASS, zero blocking findings; actual result acceptance remains pending.** Same-family/provisional. Requested Astra/max is distinct from the **UNATTESTED** actual native serving model and effort.

Reviewed report SHA256: `807746c6ab295fc39de40bd3a91028c57dc7d346f89e1351d803d5ac401a314d`. Plan SHA256: `7f58c0bf4fbb784d019a22d6e318c4f343892c0a8be0cedb36bde778e94ec936` (unchanged). R1 FAIL and original snapshots remain preserved.

Only the two reviewed changes appear in the AST/diff:

- **F1 closed:** report line 45 uses the unchanged producer's absolute `<1e-5` per-metric tolerance. The inside-tolerance scalar passes and an outside-tolerance scalar fails.
- **F2 closed:** report lines 55–58 validate the old raw-semantic order against its already registered artifact SHA before reading and comparing. Equal changed fixtures are rejected; equal original fixtures pass.

`SOURCE_ONLY_CHECKS_R2.json` records three narrow follow-up checks and inherits the passing A–F/math/scope/call-chain checks from R1 without repeating them. `R1_TO_R2.diff` and `REVIEWED_INPUT_MANIFEST_R2.json` preserve the exact reviewed revision. The plan, training/evaluator/source seal and all other captured inherited inputs remain byte-identical.

GT remains sealed dataset metadata with full legal query/gallery; AP/CMC, four-group mAP and Rank-1 count identities, ceiling-5% signed tail, formal epoch/batch coverage, one-seed scope and timing/partition-memory boundaries are as documented in R1. A complete individual endpoint may be reported while another job runs. No extra numerical/kernel or whole-campaign gate was introduced.

The last supplied actual witness is still training/RUNNING, not complete50. This review does not endorse a new retrieval score, strict-reload result, binary content or producer receipt authenticity. The missing upstream author scorer/local source-map limitations remain explicitly documented in R1 and `LOCAL_SOURCE_SCOPE_LIMITS_R1.json`; they are provenance limits, not a newly imposed production gate. Requested model identity is not serving-model attestation.

Reviewer activity: stdlib/isolated AST/scalar/text fixtures only, zero torch/NumPy/model/tensor/GPU/SSH/production execution. The reviewer wrote audit artifacts only and did not edit source or the live queue.

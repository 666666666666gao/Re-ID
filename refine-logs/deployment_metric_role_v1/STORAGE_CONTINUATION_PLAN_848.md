# Observed disk interruption: unchanged experiment continuation

The original RGBNT100 semantic training finished all 50 epochs with exit 0 at
2026-10-05 06:29:41.562464 CST. Its parent exited 1 at 06:29:41.662154 because
the unchanged 2GiB reserve check failed before launching the first strict
evaluation. No evaluation log or official receipt was created. Preserve the
original parent exit and original campaign snapshot SHA
`4d5b9e00e2b3739a59c0ae411c89b28b4955701b5bb16780eaab84e54130223d`.

Authorized legacy storage retirement completed at 06:51:31.142249, removing
41 exact SHA-verified files totaling 2,725,126,373 bytes. Three completed EV1
combined bests had lower mAP than their retained semantic counterparts; 38
consumed diagnostic caches were retired. Current source339 and control187
verified unchanged. No normalized-control weight was retired.

Run STORAGE_CONTINUATION_848.py once in a distinct administrative campaign:

1. Evaluate the existing semantic best for the first time, with its original
   initializer and full50 artifacts. Accept using the unchanged verifier and
   retire only its verified M0 probe. Add acceptance chronology; never rewrite
   the original campaign.json or parent EXIT1.
2. Prepare originally never-started RGBNT100 native in the new campaign.
   Use the original own-eight-step M0, fresh50, first strict evaluation and
   accepted-probe retirement. Preserve all339 scientific source bytes,
   control187, seed42, batch, precision, recipe and thresholds.

Only 2026 physical GPU0/1, one process using both cards. Existing warm conda
environment; no power/temperature action. No semantic retraining, M0 replay,
failed MSVR native retry, or scientific repair. The unchanged per-command
2GiB reserve remains in force. Keep one formal mAP-best with all CMC from
that checkpoint and all result evidence.

REPORT_AVAILABLE_FIVE.py from section846 remains unexecuted and immutable;
its old COMPLETE-campaign prerequisite no longer applies to the failed parent.
A future report must explicitly bind mixed artifact campaigns instead of
rewriting history or claiming the missing MSVR native endpoint succeeded.

Native timing estimate is approximately 7,231 seconds from the completed
matched raw-control run; schedule the first near-end observation only after
the actual new full-training start. Later polling interval: 240 seconds.

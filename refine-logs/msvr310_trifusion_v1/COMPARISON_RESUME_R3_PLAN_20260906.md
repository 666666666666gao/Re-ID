# MSVR310 original comparison continuation R3

Registered 2026-09-06T08:26:05.110072+08:00. PREPARED_NOT_RUN. This continues the original fixed study after
an observed and repaired evaluation execution difference; it is not another method version/run.
Original config49d82696..., original training runnera1771c2e... and all20 project bindings unchanged.
No training hyperparameters, seed42, source data, augmentation/sampler, losses, model or gates change.

Cause and validation are complete: freezing SIM switches two noncontiguous input projections
from mm to bmm in PyTorch2.5.1. Exact helper9b7a3168... restores B0 dispatch using one detached
weight view under no_grad; no registered weights or optimizer flags change. Full360 paired original/
repaired forwards prove exact B0 baseline features AND210x360 distances, unchanged three-role/modal
residuals, original five64+40 batches and unchanged full/frozen/Signal states. Original exact gates remain.
Verification9eba027 elapsed30.378821736201644 seconds,720 full-role records,0 optimizer/ranking.

Reuse exactly original fold0 epoch20 checkpoint b8a85e16... and its original260 training updates.
Use the verified full360 five-output feature arrays for its first ranking evaluation; do not reread
fold0 images or retrain its model. The new receipt explicitly cites the verification's strict reload,
original training execution1c444cd, original checkpoint and cached verification arrays.
Original comparison directory, RUNNING partial JSON, failed exit/log, receipts and checkpoint stay
byte unchanged; record/hash all original files before and after continuation.

Train only previously unstarted folds1/2, each fresh from its own B0 checkpoint and seed42 roles.
Initial states must match original M0 initialization; original build_model/train_roles functions run
unchanged for20epochs/260updates per fold. Same B64/K8/128x256/AMP/5epoch warmup cosine/seven losses.
Fixed final checkpoints only, strict reload, use verified inference helper for complete349/323
galleries and original evaluate/comparison_summary functions. Require bitwise B0 features AND distances
on each fold, original scene filtering, upstream metric crosscheck and all fixed engineering gates.
No ranking-based early stop, parameter scan, fold/epoch/seed choice or cherry-picking.

Continuation new budget520 training updates/33280 source record exposures,672 heldout role forwards;
formal study totals780 updates/49920 exposures with original260 reused once. All600 queries,60 query
identities,1032 gallery records/155 heldout identities and95 single-scene distractor identities retained.
B0 prerequisite1950 updates and all M0/diagnostic costs remain separate; no compute-matched claim.

Original five support conditions are unchanged: fused >=Signal+1pp; each fold fused nonnegative;
all three full branches >=Signal; identity-cluster query-weighted seed42/10000 linear2.5% lower bound>0;
fused strictly best across five outputs. Original evaluate/comparison_summary code is reused verbatim.
No cross-fold feature distance, dev/official/ablation access, new architecture or scientific promotion
from engineering PASS. Final metrics and complete artifacts require separate terminal verification/audit.

Output comparison_resume_r3 is new and must not exist before launch. Wrapper records PID, child PID,
command, launch time and terminal exit; estimate9-13minutes from original fold0 training. First status
read after240seconds, then180-300seconds or estimated completion; no repeated short training polls.
No fallback, blind relaunch, overwrite or original fold0 retraining if a check fails.

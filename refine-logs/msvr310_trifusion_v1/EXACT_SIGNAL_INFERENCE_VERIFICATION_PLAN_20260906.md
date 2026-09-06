# MSVR310 exact Signal inference repair verification

Registered 2026-09-06T08:19:47.876038+08:00. PREPARED_NOT_RUN. No ranking or training.

Measured cause: only freezing SIM parameters reproduces the failure on identical cached inputs;
restoring requires_grad restores bitwise B0. Two aten::mm become aten::bmm. The earliest module
difference is cross_attn, with exact inputs/token selection. Builder and mamba imports do not cause it;
construction freezes parameters, and loading final roles adds no difference. Original instrumented
Signal exactly matches stored B0. PyTorch v2.5.1 LinearAlgebra.cpp should_fold in the
archived source (locate function by name) inspects requires_grad even under no_grad, consistent with
observed noncontiguous Q/KV input projection dispatch. This is distinct from MHA's native self-attention fastpath.

The small inference-only helper uses torch.func.functional_call with one detached in_proj_weight
view requiring grad, under torch.no_grad. The original registered parameter stays frozen before/after,
shares unchanged storage values, no optimizer/backward/autograd graph, and no backend/config changes.
Original training/model/runner files remain immutable; this is an evaluation execution repair.

Fixed verification: all360 already visited original fold0 gallery records, five64 and one40 batches.
Load original epoch20 checkpoint, run unrepaired and repaired complete models on each identical batch.
Require all three role residuals, all modality residuals and direct embeddings bitwise unchanged;
require restored original parameter identities/flags and unchanged full/frozen/Signal state hashes.
Require repaired baseline3072D features and entire210x360 distance matrix bitwise equal stored B0.
Reproduce original unrepaired feature inequality. Keep all original exact gates; no tolerance.
720 full-role record forwards,360 distinct images,0 updates/backward/checkpoint/ranking/AP/Rank.
Arrays remain remote. A failed check stops; no fallback or alternative repair scan is registered.

Only after PASS may an explicitly registered continuation reuse original fold0 checkpoint/training
and train untouched folds1/2 with original functions/settings and this verified inference helper.
No fold0 retraining, no new M0 optimization, no official/dev/ablation access or scientific-gate change.

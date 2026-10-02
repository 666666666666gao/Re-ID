# ProxyTTT pinned primary-code boundary (2026-10-02)

This is an executor's read-only source inspection, not an independent integrity
review, author reproduction, or new TriFusion performance result. Fixed author
commit: `92fb0fa33d74813566e06820e56e8d8f48ca1205`. All17 selected source files
match their original Git blob IDs; exact source stays in the private intake.
The catalog records paths, primary URLs, bytes, Git blobs and SHA256. Earlier
API403, filtered-checkout and raw-response timeout records remain unchanged.

## Target adaptation is in the training entry

`datasets/make_dataloader.py` builds `val_loader` from `query + gallery`.
`processor/processor.py:111` onward iterates that loader with `TTT=True`, calls
backward and steps the ordinary optimizer before the epoch's metric evaluation.
RGBNT100 does this at epochs divisible by `TEST.TTT_epoch`; other datasets do
it from that epoch onward. The supplied three YAMLs set that value to2 and
train70 epochs, B64/K8, Adam, baseLR3.5e-5. This is target-data adaptation,
not an extra fixed-parameter source-only training module. The ground-truth
`vid` is unpacked but not passed to the TTT model call.

The model's training/TTT branch uses the source proxies and PESA pseudo-label
selection. In contrast, `do_test` sets eval/no_grad and invokes the default
`TTT=False`; its callee returns the concatenated three768-dimensional CLS
features (2304D for complete modalities), without calling the proxy-training
branch. Thus a fixed-parameter *test entry* does not establish that the loaded
checkpoint was trained without target adaptation. The saved training-best
state can already contain the preceding validation-loader updates.

The TTT loops have no explicit optimizer.zero_grad call in those loops, while
the ordinary source-training loop clears it at each batch. This is a static
implementation observation, not a measured training failure or a proposed
change to author code. No model execution was performed.

## Evaluation predicate and inventory remain distinct

`utils/metrics.py` filters MSVR by samePID AND sameScene, and201/100 by samePID
AND sameCamera, normalizes features and ranks squared Euclidean distance.
MSVR reads `query3`;201 uses `test` for both query and gallery;100 uses its
query/bounding_box_test directories. The parsers build gallery records without
restricting them to query identities. The exact author's historical images,
file order and checkpoint remain unavailable here. Predicate agreement alone
does not certify identical inventories, scores or an execution reproduction.
The MSVR scorer's short rank-list text output does not truncate AP computation.

## Comparison use

Keep the official paper's w/oTTT and fullPESA rows separate. Its82.3/88.4/62.1
mAP w/oTTT rows are reported fixed-inference references, not locally reproduced
results;85.0/89.3/63.6 include the target-adaptation resource. No claim about
historical w/oTTT checkpoint provenance follows merely from these defaults.
F2's sealed loss/margin/seed/50epoch/protocol remains unchanged.

Primary code: https://github.com/liuzhaojun-zwd/ProxyTTT/tree/92fb0fa33d74813566e06820e56e8d8f48ca1205
Primary paper: https://ojs.aaai.org/index.php/AAAI/article/download/38337/42299

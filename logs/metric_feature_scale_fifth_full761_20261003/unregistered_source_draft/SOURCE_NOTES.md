# Independent semantic/detail reading — private, unregistered draft

This is implementation preparation, not a reviewed model, passed runtime gate,
launched experiment or successful contribution. Complete the original F3 six-arm
panel and its one CPU report, integrity audit and claim review before choosing
the matched foundation or registering a training plan. Do not alter F3/EV1.

Evidence read from current actual source:

- `semantic_native_evidence.py:25–44` interpolates semantic evidence to512
  candidates, uses semantic keys, and applies a single value projection toS
  orS+D. Detail cannot directly change those attention scores for fixedS/query.
- `native_detail_roles.py:46–74` changes only CNN evidence before the existing
  CNN→Transformer→Mamba bridges; private-source preparation should retain those
  roles/readout initially, rather than adding matching or new losses together.
- `slot_competition_fp32_roles.py:10–20` has the established FP32 attention
  boundary after actual Q/K gradient support failure. The draft preserves it.
- `run_foundation_recipe.py:134–147` builds the author optimizer from
  `model.signal`. Appending a module outside that object would leave it out of
  the optimizer. A future trainer must explicitly cover every intended trainable
  parameter; do not reuse that call unchanged for a new role/detail model.

The standalone source draft uses image-native detail to build its own K/V and
semantic-context queries for Q. It returnsB×3×16×128. The caller must keep the
existing128-token semantic reading and combine at the evidence exit, with one
new output weight initially zero. This initialization only proposes initial
forward equality; actual paired input/forward and full gradient/restore witnesses
are not yet run. It provides no performance preservation guarantee.

Only the detail component is drafted. Full role integration, optimizer, heads,
paired initialization and formal budget are deliberately unset pending the full
F3 foundation judgment. No test dependency, environment rebuild, model execution,
source review or remote deployment has occurred for this draft.

The first comparison must report the added parameters and compute. A positive
native-addition result alone would confound information and capacity; a matched
extra-semantic or reduced-detail control remains required before claiming that
the independent detail content itself explains improvement. Do not assume that
two branches or attention formulas alone establish novelty.

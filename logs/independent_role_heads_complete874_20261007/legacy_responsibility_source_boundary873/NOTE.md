# Legacy responsibility source boundary

This is a bounded local source review, not a new gradient probe or experiment.

V26 `role_modal_responsibility_v26.py:22-40` forms nine expert-by-modality residual embeddings. It detaches fused/global comparisons and difficulty weights, but does not detach slot margins in the penalty. The responsibility loss therefore has a gradient path into these embeddings. These nine groups are not the current sixteen spatial slots.

`train_signal_preserving_v26.py:140-164` includes the enabled responsibility term in the actual total loss and calls backward. `role_modal_responsibility_v26.py:65-91` explicitly diagnoses the same encoder parameter blocks (42 CNN / 54 Transformer / 93 Mamba tensors) and requires finite nonzero auxiliary gradients. Thus merely sending responsibility back to feature formation is not a new distinction from V26. This review does not rerun those historical diagnostics.

V26's detached baseline similarity is used in decomposition diagnostics; this file does not define separate relative-global repair and keep objectives. A later proposal must distinguish spatial selection/shared-private learning and explicit repair/keep responsibility, and compare against existing V26/R2/CIRC and the closest published objective. No novelty is established by this review.

`training_phases.py:31-53` defines CIRC router-only warm-up followed by a joint full registered objective with all parameters trainable except private projections. It is therefore inaccurate to describe every CIRC phase as router-only. This is a static training policy observation; actual historical gradient activity needs its own run evidence and is not claimed here.

No active neural sources, model configuration, thresholds, queues, Git state, or remote files were changed. Publication is deferred until the current neural queue terminates. The first local extraction incorrectly queried a nonexistent job.mode key and raised StopIteration; the actual saved job schema uses phase. This was a local reader error, with no SSH or experiment effect.

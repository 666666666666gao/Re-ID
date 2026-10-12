# MSVR310 text-package M0 failures

Both fixed901 packages completed eight actual M0 optimizer updates, finite/all-parameter activity checks and strict reload with zero recorded output difference. All149 frozen text states remained unchanged. Both then failed the registered independent role-VJP activity assertion at `queue_text_semantic_prior.py:101`. Neither entered formal epoch training and neither has an official retrieval score.

For both packages, the isolated scaled observer recorded positive `output.weight` gradients on all seven noninitial steps and exact zeros for the four `psi` tensors. The actual post-step unscaled task-gradient log recorded positive maxima for all five tensors on those same seven steps. In the pretrained second update, actual `psi` maxima range from approximately5.75e-10 to1.31e-8. These are tiny positive task gradients, not zero and not a proof of useful identity information.

The production training source calls its M0 `autograd.grad` while still inside the outer autocast region; the actual scaled optimizer backward is called after leaving that region. The observer also recomputes the role loss, whereas actual backward differentiates the summed global and role loss. Those are source-visible differences. The existing detach policy blocks the global task from training the text branch, so the zero-versus-positive discrepancy needs an observation-boundary explanation before any claim of branch disconnection. No first divergent operation has been localized and no numerical correction has been executed here.

PyTorch's current AMP example places the scaled `autograd.grad` call outside autocast, divides its returned gradients before use, and recommends running optimizer backward outside autocast. This is relevant implementation guidance, not evidence that the installed runtime has been fixed or that changing one context will guarantee a pass. [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-penalty)

Original failures, source, diagnostic fields and the fixed acceptance gate remain unchanged. The current pilot proceeds to its already registered RGBNT100 endpoints. Any later missing-endpoint decision requires the closed pilot report and a concrete source review; no scale, threshold, loss, seed or model revision is introduced during this campaign.

Closed original failure text archives:

- Pretrained:8 files,21,347 bytes, SHA256 `fc16f155db1fa68b7f5f2304e9713a48b57845527615794ac3ad26f3679fac11`.
- Random:8 files,21,189 bytes, SHA256 `09cc68ad4e9f4e150021b8786cfd506998bc6fd578aa6810825fc8ce61ced479`.

The archives retain the eight-batch receipts and raw traces. Both original900 MSVR probes and both failed901 probes are still retained pending the current lifecycle closure. They are engineering binaries without formal best metrics, not missing retrieval results to be filled in with diagnostic numbers.

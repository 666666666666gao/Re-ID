# V15 Round 1 Review

<details>
<summary>GPT-5.5 xhigh raw review</summary>

Read the full proposal. Bottom line: **REVISE**, not READY. V15 is pointed at the right bottleneck and is materially more relevant than V13/V14 router-only changes or V9 late residual relay, but the proposal needs tighter counterfactual semantics, BN/classifier specification, and contribution framing before it can support the claimed mechanism.

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 8.0 | Directly targets “new identity representation beyond fixed branches/router,” while preserving Signal and avoiding external backbones. |
| Method Specificity | 6.5 | Core CRDE is concrete, but off-path detach, BN/classifier behavior, trainable/frozen boundaries, and stage-3 semantics need exact definition. |
| Contribution Quality | 6.5 | CRDE + matched no-exchange regret is focused, but listing V8 semantic rebranching as a retained method point creates contribution sprawl. |
| Frontier Leverage | 7.0 | Sensible use of frozen CLIP tail as semantic interpreter; no need to force LLM/VLM/Diffusion/RL. |
| Feasibility | 7.0 | Likely feasible on 3090 after M0 capacity, but on/off dual forward and BN state handling are real implementation risks. |
| Validation Focus | 7.5 | Minimal and mostly claim-driven; Q1 is useful as a train-only gate, but cannot itself imply dev ≥65. |
| Venue Readiness | 6.0 | Promising early method, but currently too dependent on precise implementation hygiene and too close to prior exchange/fusion work unless narrowed. |

**OVERALL weighted score: 6.9 / 10**

**Dimension Fixes Below 7**
- **Method Specificity, 6.5, CRITICAL**: `exchange_off` is not yet a valid counterfactual unless it is guaranteed not to update BN running stats, classifier heads, or any shared state. Fix: define forward modes explicitly. `off` should produce pre-BN retrieval embeddings under `torch.no_grad()` or full detach, with edge coefficients zero, no BN/classifier pass, and scalar `R_off.detach()`. Train ID/classifier heads only on `exchange_on`.
- **Method Specificity, 6.5, IMPORTANT**: BN/classifier ownership is ambiguous across V12 fold checkpoints and V8 all-fit Phase-A. Fix: state that each fold/all-fit run uses source-identity-only BN/classifier heads, initialized or reused consistently, never evaluated on held-out classifier labels, and retrieval metrics always use pre-BN embeddings.
- **Contribution Quality, 6.5, IMPORTANT**: The “existing foundation contribution retained” makes the paper look like V8 + CRDE + regret. Fix: demote V8 to prerequisite/background. The sole contribution should be: “delta-only pre-tail expert exchange trained by matched no-exchange retrieval regret.”
- **Venue Readiness, 6.0, IMPORTANT**: Novelty over HFER/FusionReID-style exchange is fragile unless the paper stresses the exact distinction: role-adapter delta only, injected before frozen CLIP tail, receiver private path preserved, no late pooled relay, no router. Fix: delete broad claims about tri-expert fusion and lead with this narrower mechanism.

**Requested Integrity / Scope Checks**
- **V12 checkpoint reuse**: No direct leakage if the proposal truly uses V12 complete-path identity-OOF checkpoints where each fold’s Signal/expert path was trained only on the 94 source identities and evaluated only on the 47 held-out identities. The issue is scope, not leakage: Q1 uses V12 fold-specific complete-path models, while D1 deploys on V8 all-fit Phase-A. Q1 must be described only as a necessary train-only qualification gate, not as evidence that the all-fit deployable model will reach 65.
- **Counterfactual validity**: Currently under-specified. “Edge coefficients zero and detach” is not enough if off-path BN/statistics/classifier state can move. Fix as above: off-path is a frozen pre-BN retrieval comparator, not a second train branch.
- **Classifier/BN mismatch**: Real risk. The proposal must say exactly whether BNNeck/classifiers are reinitialized per fold, reused, frozen, or trained. Since retrieval uses pre-BN embeddings, the cleanest version is: classifier heads are training-only, source-class-local, never part of the counterfactual risk or evaluation.
- **No-exchange detach issue**: Detaching only the risk scalar is acceptable if the off embedding path has no state mutation. Otherwise the “matched no-exchange” baseline can be contaminated by training.
- **Gate support**: Q1 can support “identity-disjoint CRDE beats matched no-exchange on fit folds.” It cannot support “deployable deterministic collaboration clears 65”; only D1 can. Also, “at least four nonzero edges” should be a diagnostic, not a scientific gate, because nonzero weights are not evidence of useful cooperation.

**Simplification Opportunities**
1. Delete V8 semantic rebranching from the contribution list; treat it as the frozen substrate.
2. Make matched retrieval regret part of CRDE training, not a separate supporting contribution.
3. Move “four nonzero edges” from pass/fail gate to diagnostic logging.

**Modernization Opportunities**
NONE for new components. The CLIP-tail-as-frozen-semantic-interpreter framing is the right modern leverage here. Adding DINO, VLM text, diffusion, RL, or LLM planning would be unnatural drift.

**Drift Warning**
NONE at the problem level. The method still attacks the anchored bottleneck. Scope warning: do not let V12 OOF qualification results become a proxy main result for the V8 all-fit deployable dev target.

**Verdict: REVISE**

The core idea is worth refining: pre-tail role-delta exchange is meaningfully distinct from V9’s late pooled orthogonal relay and avoids the failed V13/V14 router bottleneck. It is not READY because the matched counterfactual must be made implementation-proof, and the contribution needs to collapse into one clean mechanism rather than a three-part system story.

</details>

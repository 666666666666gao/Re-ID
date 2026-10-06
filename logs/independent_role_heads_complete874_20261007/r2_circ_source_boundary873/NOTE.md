# R2 and CIRC: existing responsibility mechanisms

Bounded local source review only. No model forward, gradient probe, new experiment, or reconstruction of historical runtime was performed. Source hashes below describe these local raw bytes; remote equivalence of the legacy files was not independently measured in this review.

| Existing mechanism | Actual source behavior | Consequence for a later proposal |
|---|---|---|
| R2 legal multi-positive retrieval | `tools/msvr_cross_scene_smooth_ap.py:35-55` uses same identity / different environment positives, ignores same-identity same-environment candidates, averages approximate AP across each eligible anchor's positives. Different identities remain negatives. | Legal environment filtering and multiple positives are existing controls, not new contributions. |
| R2 candidate-side learning | `tools/train_official_three_dataset_roles.py:272-309` computes gradients for historical refreshed candidate embeddings, recreates their graph, and passes the vector-Jacobian product into the 189 encoder tensors. | Training responsibility upstream and treating both query/candidate sides are already present. Merely adding these paths is not a new mechanism. |
| R2 supported balancing | `tools/msvr_supported_gradient_balance.py:66-96` combines current plus historical ranking gradients with auxiliary gradients per expert role; only supported active steps update the balance state. | A new role gradient coefficient or ordinary ranking objective needs a matched R2 comparison. |
| Generic R2 environment field | `tools/build_official_three_dataset_protocols.py:49-53` deliberately stores camera in the scene field for RGBNT201/100 and actual scene for MSVR310. | Reading protocol.scene in this old R2 entry is intentional; it is not the newer raw logger's MSVR camera/view proxy error. Do not claim a generic R2 filtering bug. |
| CIRC deletion sensitivity | `modeling/trifusion/circ_scoring.py:774-828` compares the uninterrupted full fused network's margin against total/direct/relay interventions, with an optional symmetric reference bank. `:622-690` uses different-camera positive prototypes and identity-negative prototypes. | Deletion-sensitive usefulness, full-network propagation, and query/candidate symmetry have precedents in this project. CIRC's baseline in this target generator is the uninterrupted fused model, not an independently trained global-only vector. |

The CIRC reference margin is similarity to a positive prototype minus the largest negative-identity prototype similarity; this differs from a per-query list of all legal positive instances. This source distinction does not establish novelty or invalidate CIRC's historical results.

Together with the preceding V26 audit, these facts narrow the next research comparison. A potential P3 must explicitly distinguish relative-global repair and keep responsibility and its use in learning spatial selection/shared-private content. It should compare with existing R2/V26/CIRC and the closest direct published objective. No claim is made here that the proposed objectives are new or effective.

Current independent-head training remains unchanged. Publication of this source note is deferred until that queue and its original report are terminal. No early remote query or additional GPU work was performed.

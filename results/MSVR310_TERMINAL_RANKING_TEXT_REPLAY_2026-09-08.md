# Complete terminal ranking text replay

A read-only CLI now extracts the ranking replay from the previously executed full instance-memory Q1 analysis. It does not import training modules, load models or distance arrays, access images, or alter a running experiment. It requires the exact terminal summary SHA, CPU verification status and completed pipeline bindings before analysis.

## Executed validation

The input is the sealed 2026-09-07 instance-memory Q1, with `control` meaning batch-only and `instance_memory` meaning stale expanded history. All six endpoints and five outputs were replayed: 2,069,520 saved ranking positions, 600 queries and 60 identities. Every ranking is a full gallery permutation; only same-identity/same-scene gallery records are filtered. All other distractors remain. AP, Rank-1/5/10, both fixed five-gate sets and the original seed42 10,000-draw identity-cluster bootstrap agree with the archived terminal result.

| Output | Paired mAP gain (pp) | AP improved / declined / unchanged | Rank-1 repaired / new errors |
|---|---:|---:|---:|
| Fused | -0.2161564150 | 258 / 284 / 58 | 20 / 16 |
| CNN | +0.5759355076 | 289 / 272 / 39 | 24 / 13 |
| Transformer | +0.2243257183 | 273 / 273 / 54 | 13 / 20 |
| Mamba | -0.2776803277 | 249 / 310 / 41 | 25 / 12 |

The old FAIL and 0/5 paired conditions remain unchanged. This is verifier validation against real complete data, not another training result, another seed, or new independent generalization evidence. Ruff F also passed.

The evidence folder contains all 3000 query/output changes and all 300 identity/output changes, with hashes in ranking_replay.json. Query records include first/last positive rank and nearest-negative record/identity/scene. Those columns describe saved rankings; they do not establish visual causes. No individual official test identity is used for tuning.

## Current experiment boundary

The fresh-coordinate run remains bound to b4501fa/config32e22d3a. Its `control` means stale-history updates and `fresh_memory` means current-role reencoded-history updates. Its complete Q1 has NOT been read or replayed by this tool. The new CLI will be used only after all six endpoints and original CPU pipeline terminate, alongside the already validated training-text analyzer and fresh-context audit.

Latest actual observation 2026-09-08T02:21:04+08:00: original wrapper192704/Q1193650 alive with exact command lines, two endpoints complete, fold1 control epoch1/20, main disk4,362,149,888 bytes free. No training/config/checkpoint changes or weight deletions occurred. Completion estimate remains around03:40; inspect milestones, not partial metrics.

This is executor-side deterministic text verification, not an independent reviewer. It does not recompute feature distances or gradients. Those remain within the separately receipted remote CPU/runtime scopes. Seed42 and repeatedly used source OOF limitations remain. Three-dataset baseline/SOTA Goal ACTIVE/UNMET.

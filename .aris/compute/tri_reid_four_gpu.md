# Existing four-GPU TriFusion environment

Provider: user-owned SSH gaob@172.19.12.138:2026; key transport, ProxyCommand=none. No password stored.

Warm reuse only; no environment rebuild or package installation. Runtime `/data/gaob/Re-ID/conda-envs/tri_reid/bin/python`, repository `/data/gaob/Re-ID/Trifusion`, existing datasets/protocols/baseline weights at the registered queue paths. torch2.5.1+cu121, numpy1.24.4, mamba-ssm2.2.6.post3, timm1.0.15 from the recovery environment. This is the current host, replacing obsolete single-card AGENTS paths.

October1 recovery witnesses executed real CUDA/Mamba forward/backward on all four cards, followed by six actual production eight-batch M0s and complete training. September30 GPU incident remains archived; kernel cause/admin repair is unavailable. A new ledger records this existing runtime, not import-only readiness or an environment change.

New campaigns must check actual free devices and run their own production-data M0 with real Mamba/AMP, finite loss/gradients, every trainable tensor nonzero at least once, unchanged frozen baseline and strict reload. Dependencies and baseline SHA are pinned by each manifest. Poll240sec; never restart on an observation timeout. Weights and raw matrices remain remote. No W&B or messaging integration.

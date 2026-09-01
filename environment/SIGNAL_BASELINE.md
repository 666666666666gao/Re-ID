# Signal baseline environment

The remote-only comparator environment is pinned to Python 3.10.13,
PyTorch 2.1.1+cu118, torchvision 0.16.1+cu118, and CUDA 11.8. Rebuild it on
the GPU server with either the complete environment file:

```bash
/root/miniconda3/bin/conda env create -f environment/signal_environment.yml
```

or the verified two-step form:

```bash
/root/miniconda3/bin/conda create -n signal python=3.10.13 pip=26.2.1 -y
/root/miniconda3/envs/signal/bin/pip install setuptools==83.0.0 wheel==0.47.0
/root/miniconda3/envs/signal/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r environment/signal_requirements-lock.txt
/root/miniconda3/envs/signal/bin/pip check
```

The runtime lock is the upstream Signal `requirements.txt` at commit `cd1b0a6` with
three training-unrelated entries removed for observed, reproducible reasons:

- `grad-cam`: author-local `file:///media/...` URL; imported only by
  visualization scripts.
- `visdom==0.2.4`: PEP-517 build fails because its build environment cannot
  import `pkg_resources`; Signal does not import `visdom`.
- `ninja==1.11.1.1`: reports unsupported-platform metadata on the server;
  Signal does not import it and no installed package requires it.

The verified remote receipts are under
`artifacts/signal_env_cd1b0a6/`. The final `pip freeze` is byte-identical to
`signal_requirements-lock.txt` and has SHA-256
`f2956e90e3a179eee2a0260dccca56743e249125dace639d4ad710e55e10ad38`.

## Verified development run

The sole seed-42 run used B64/K8 for 50 epochs on the frozen 141-fit/30-dev
identity split. It completed on the remote RTX 3090 with a 13,620 MiB peak
reserved allocation and zero official-test access. Reloading the best
checkpoint produced:

| mAP | Rank-1 | Rank-5 | Rank-10 |
|---:|---:|---:|---:|
| 58.0109 | 57.4545 | 69.9394 | 76.6061 |

The complete 3072D retrieval feature and camera SIE were active. The best
checkpoint is SHA-256
`1f5c200cd43fcbc00b8a0494329519eed3e6f062d9a29d43a0ecdd97ff4966c3`;
the fixed epoch-50 checkpoint is
`9f3a74a75fd5e2d1fa2dff0db011dfcd0360bdd76d75ba7b4a140965dcf15b5c`.
These are held-out-development artifacts. They do not reproduce Signal's
upstream official-test 80.3 mAP / 85.2 Rank-1 result.

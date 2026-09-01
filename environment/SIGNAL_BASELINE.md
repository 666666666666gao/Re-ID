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

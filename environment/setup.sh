#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# The overlay may be recreated on container rebuilds — a venv inside
# it will be lost and all subsequent jobs will fail.
# Use: uv pip install --system <packages>
# Do NOT create a venv (python -m venv, virtualenv, etc.).
#
# torch is the CUDA 12.6 build: the reference proposer is trained to
# convergence (150k steps), which takes minutes on a GPU and many hours on one
# CPU. The CUDA wheel bundles its runtime libraries and also runs on CPU-only
# nodes, where the evaluators run.
#
# The CUDA build and its libraries are ~3.4 GB of wheels. Installed in one
# command they outlast a 10-minute command timeout, and a killed install
# leaves torch's metadata behind without its files. So the pinned set
# (resolved with `uv pip compile` for torch==2.12.1 on the cu126 index,
# Python 3.11, manylinux_2_28) is installed in groups of at most ~0.8 GB, each
# with --reinstall so that a retry really rewrites a partial install, and the
# whole step is skipped once a working CUDA torch is present.

set -e

uv pip install --system numpy

TORCH_INDEX=https://download.pytorch.org/whl/cu126

torch_ok() {
    python - <<'PY'
import sys
try:
    import torch
    ok = (torch.__version__.startswith("2.12.1") and torch.version.cuda is not None
          and torch.backends.cudnn.version() is not None)
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "cudnn", torch.backends.cudnn.version())
except Exception as exc:  # broken or missing install
    print("torch check failed:", repr(exc))
    ok = False
sys.exit(0 if ok else 1)
PY
}

install_group() {
    local attempt
    for attempt in 1 2 3; do
        # `timeout` runs the uv binary itself, with a limit sized for the
        # largest group; hung downloads are still killed and retried.
        if timeout 2400 uv pip install --system --reinstall --no-deps \
                --index-url "$TORCH_INDEX" "$@"; then
            return 0
        fi
        echo "[setup] install group failed (attempt $attempt/3): $*" >&2
    done
    return 1
}

if torch_ok; then
    echo "[setup] CUDA torch already installed"
else
    install_group typing-extensions==4.16.0 setuptools==78.1.0 sympy==1.14.0 mpmath==1.3.0 \
        networkx==3.6.1 jinja2==3.1.6 markupsafe==3.0.3 fsspec==2026.9.0 filelock==4.2.4 \
        cuda-toolkit==12.6.3 cuda-bindings==12.9.7 cuda-pathfinder==1.6.0
    install_group nvidia-cudnn-cu12==9.10.2.21
    install_group nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-runtime-cu12==12.6.77 \
        nvidia-cuda-nvrtc-cu12==12.6.85 nvidia-cuda-cupti-cu12==12.6.80 \
        nvidia-nvjitlink-cu12==12.6.85 nvidia-nvtx-cu12==12.6.77 nvidia-cufile-cu12==1.11.1.6 \
        nvidia-curand-cu12==10.3.7.77
    install_group nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.7.1
    install_group nvidia-cufft-cu12==11.3.0.4 nvidia-cusolver-cu12==11.7.1.2 \
        nvidia-nvshmem-cu12==3.4.5
    install_group nvidia-nccl-cu12==2.29.3 triton==3.7.1
    install_group torch==2.12.1+cu126
    if ! torch_ok; then
        echo "[setup] CUDA torch install is not importable" >&2
        exit 1
    fi
fi

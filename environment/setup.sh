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

set -e

uv pip install --system numpy
uv pip install --system "torch==2.12.1" --index-url https://download.pytorch.org/whl/cu126

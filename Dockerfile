# syntax=docker/dockerfile:1
#
# CUDA image for the EntroGuard project. Builds a Python 3.12 env with uv
# (using uv.lock for reproducibility), then swaps in a CUDA build of torch
# since the default PyPI wheel uv.lock resolves to is CPU-only.
#
# Build (adjust CUDA_TAG to match your driver -- cu121/cu124/cu126/...,
# see https://pytorch.org/get-started/locally/):
#   docker build --build-arg CUDA_TAG=cu124 -t entroguard .
#
# Run (needs the NVIDIA Container Toolkit on the host for --gpus to work;
# without a GPU, drop --gpus all and the code falls back to CPU):
#   docker run --gpus all -p 8000:8000 \
#     -v entro_hf_cache:/root/.cache/huggingface \
#     -v "$(pwd)/checkpoints:/app/checkpoints" \
#     -v "$(pwd)/results:/app/results" \
#     entroguard
#
# Then open http://localhost:8000 -- the web dashboard (app.py) is the
# default CMD. Override it to run a single script instead, e.g.:
#   docker run --gpus all entroguard python train_attacker.py --arch transformer

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Ubuntu 22.04 ships Python 3.10; pyproject.toml requires >=3.12, so pull
# it from deadsnakes instead of building from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Dependency layer first so code-only changes don't invalidate the cache.
COPY pyproject.toml uv.lock ./
RUN uv venv --python 3.12 .venv && \
    uv sync --frozen --no-install-project

# uv.lock resolves torch from PyPI, which is CPU-only -- reinstall it from
# the CUDA wheel index for the same version, GPU build.
ARG CUDA_TAG=cu124
RUN uv pip install --python .venv/bin/python \
        --index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --no-deps --force-reinstall torch

COPY . .
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

# checkpoints/ and results/ are how you get trained weights and reports
# back out of the container; the HF cache is what makes download_data.py
# a one-time cost instead of a per-run one.
VOLUME ["/root/.cache/huggingface", "/app/checkpoints", "/app/results"]

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]

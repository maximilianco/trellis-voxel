# ACE-Step 1.5 (MIT) as a RunPod serverless worker: text to music (instrumental loops and songs).
#
# Follows ACE-Step's own Dockerfile (CUDA 12.8 runtime, Python 3.11, uv sync from its lockfile) at a pinned commit,
# with the main model (turbo DiT, VAE, text encoder, 1.7B 5 Hz LM; 10.1 GB) baked in so a cold worker downloads
# nothing. Needs a 24 GB GPU for the LM and the DiT together, and a host driver for CUDA 12.8.

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common git curl ca-certificates libsndfile1 ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

ARG ACESTEP_COMMIT=ca1e85fe9430179831e6bc6be790c332190a3866
WORKDIR /app
RUN git clone https://github.com/ACE-Step/ACE-Step-1.5.git /app \
    && git checkout "${ACESTEP_COMMIT}" \
    && rm -rf /app/.git

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev --python python3.11 \
    && uv pip install --python /app/.venv/bin/python "runpod>=1.7,<2" requests \
    && uv cache clean

ENV PATH="/app/.venv/bin:${PATH}" \
    ACESTEP_PROJECT_ROOT=/app \
    ACESTEP_CHECKPOINTS_DIR=/app/checkpoints \
    ACESTEP_CONFIG_PATH=acestep-v15-turbo \
    ACESTEP_INIT_LLM=true \
    ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B \
    ACESTEP_LM_BACKEND=pt \
    ACESTEP_DEVICE=cuda \
    ACESTEP_COMMIT=${ACESTEP_COMMIT}

# The weights last, in their own layer: a handler change rebuilds nothing above it.
ARG BAKE_WEIGHTS=1
RUN if [ "${BAKE_WEIGHTS}" = "1" ]; then acestep-download --dir /app/checkpoints && rm -rf /root/.cache/huggingface; fi

ARG GIT_SHA=unknown
ENV WORKER_COMMIT=${GIT_SHA}
COPY handler.py /app/worker_handler.py
CMD ["python", "-u", "/app/worker_handler.py"]

# SkinTokens / TokenRig as a RunPod serverless worker: glb in, rigged glb out.
#
# Stage two of the avatar pipeline. Stage one is Hunyuan3D (deploy/hunyuan3d),
# stage three is the local Luanti conversion (lab/luanti_rig.py).
#
# SkinTokens predicts the skeleton AND the skin weights as one autoregressive
# token sequence over a Qwen3-0.6B backbone, rather than regressing weights per
# vertex. That matters here because the alternative on this project was fitting
# a skeleton to the mesh geometrically -- which works only for a humanoid in a
# known pose, and this pipeline is meant to take whatever the generator makes.
#
# It cannot share the Hunyuan3D image. This is torch 2.7.0/cu128 and Python
# 3.11+; Hunyuan3D-2.1 is torch 2.5.1/cu124 on Python 3.10. Ubuntu 24.04 is
# used here purely because it ships a new enough Python.
#
# VRAM: 14 GB minimum upstream, so a 24 GB endpoint GPU is the sensible floor.
# Unlike the Hunyuan3D worker there is no second stage to fit alongside it.
ARG CUDA_TAG=12.8.1-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    CUDA_HOME=/usr/local/cuda \
    SKINTOKENS_ROOT=/workspace/SkinTokens \
    PYTHONPATH=/workspace/SkinTokens \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV MAX_JOBS=4

# 24.04 marks the system Python as externally managed (PEP 668), so a plain
# `pip install` refuses. A venv is cleaner than --break-system-packages and
# keeps the CUDA extension builds pointed at one interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip git wget \
        ninja-build build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# Torch first: flash-attn compiles against this exact build.
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
RUN git clone https://github.com/VAST-AI-Research/SkinTokens.git ${SKINTOKENS_ROOT}

RUN pip install -r ${SKINTOKENS_ROOT}/requirements.txt 2>&1 \
        | tee /workspace/requirements_build.log | tail -5
RUN pip install runpod huggingface_hub trimesh

# flash-attn is what the Qwen3 backbone attends with. It takes most of the
# build; --no-build-isolation is mandatory because its setup.py imports torch.
RUN pip install flash-attn --no-build-isolation 2>&1 \
        | tee /workspace/flashattn_build.log | tail -5

# Weights ARE baked here, unlike the generator images. TokenRig plus the
# FSQ-CVAE is small next to a 3B shape model, and baking removes the need to
# attach a network volume to this endpoint at all -- which matters because the
# rigger is the stage most likely to be called repeatedly while tuning.
#
# Build with --build-arg BAKE_WEIGHTS=0 to attach `VAST-AI/SkinTokens` as a
# RunPod model instead.
ARG SKINTOKENS_REPO=VAST-AI/SkinTokens
ARG BAKE_WEIGHTS=1
ENV SKINTOKENS_MODEL=${SKINTOKENS_REPO}
ENV BAKE_WEIGHTS=${BAKE_WEIGHTS}

RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        cd ${SKINTOKENS_ROOT} && python download.py --model 2>&1 \
            | tee /workspace/weights_download.log | tail -5 ; \
        du -sh ${SKINTOKENS_ROOT}/experiments || true ; \
    else \
        echo "weights not baked; attach ${SKINTOKENS_REPO} to the endpoint" ; \
    fi

COPY handler.py /workspace/handler.py
CMD ["python", "-u", "/workspace/handler.py"]

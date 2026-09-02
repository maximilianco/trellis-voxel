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
# It cannot share the Hunyuan3D image. This is torch 2.7.0/cu128 on Python
# 3.11; Hunyuan3D-2.1 is torch 2.5.1/cu124 on Python 3.10.
#
# Python 3.11 EXACTLY, and not merely ">= 3.11" as upstream's README says.
# requirements.txt pulls `bpy` -- Blender as an importable module -- and bpy
# only publishes wheels for the one interpreter each Blender embeds: cp311 up
# to bpy 5.0.1, then cp313 from 5.1. There has never been a cp312 wheel. So the
# obvious reading of ">= 3.11", which is Ubuntu 24.04's stock 3.12, fails the
# build outright with "No matching distribution found for bpy>=4.2" -- which is
# precisely how this image failed the first time. 3.13 would also resolve (to
# bpy 5.2.1) but puts flash-attn and torch on much less travelled wheels.
#
# Hence jammy plus deadsnakes rather than noble: 22.04 ships 3.10, and
# deadsnakes is the reliable way to get 3.11 next to it.
#
# VRAM: 14 GB minimum upstream, so a 24 GB endpoint GPU is the sensible floor.
# Unlike the Hunyuan3D worker there is no second stage to fit alongside it.
ARG CUDA_TAG=12.8.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX" \
    CUDA_HOME=/usr/local/cuda \
    SKINTOKENS_ROOT=/workspace/SkinTokens \
    PYTHONPATH=/workspace/SkinTokens \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# sm_120 is in the arch list because RunPod's 48 GB tier now serves RTX PRO 6000
# Blackwell, and picking a GPU by capacity alone is how a worker ends up unable
# to run the image. torch 2.7/cu128 supports Blackwell out of the box, but
# flash-attn is compiled here, and compiled against 8.0-9.0 it would fail on
# that card exactly as if torch itself were too old -- the endpoint sits in
# IN_QUEUE while every worker fails its fitness check, which reads like a
# capacity problem and is not one.
#
# Only this image can carry sm_120. The Hunyuan3D worker is on CUDA 12.4, whose
# nvcc does not know the architecture at all, and its torch 2.5.1 stops at
# sm_90; that endpoint has to be given Ada or Ampere instead.
ENV MAX_JOBS=4

# A venv, not the system interpreter: it keeps every extension build pointed at
# 3.11 rather than jammy's stock 3.10, without fighting the distro packages.
#
# The X, GL, Wayland, xkb and dbus libraries are bpy's, not ours. Blender links
# against them even in background mode, and a missing one does not fail the
# build -- it fails at `import bpy` inside demo.py's bpy_server subprocess, on a
# running GPU worker, where the parent only reports "bpy_server failed to
# start" and throws the real error away. That is exactly how this image failed
# the first time it ran a job, so the set below is deliberately generous.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv git wget \
        ninja-build build-essential libglib2.0-0 \
        libgl1 libegl1 libgles2 libglu1-mesa \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxext6 libxt6 \
        libxrandr2 libxinerama1 libxcursor1 libsm6 libice6 \
        libxkbcommon0 libxkbcommon-x11-0 \
        libwayland-client0 libwayland-cursor0 libwayland-egl1 \
        libdbus-1-3 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel \
    && python -c "import sys; assert sys.version_info[:2]==(3,11), sys.version"

# Torch first: flash-attn compiles against this exact build.
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
RUN git clone https://github.com/VAST-AI-Research/SkinTokens.git ${SKINTOKENS_ROOT}

RUN pip install -r ${SKINTOKENS_ROOT}/requirements.txt 2>&1 \
        | tee /workspace/requirements_build.log | tail -40

# bpy is held to the 4.5 LTS line rather than left on upstream's `>=4.2`.
#
# That constraint was written before Blender 5.0 existed, so pip now resolves it
# to bpy 5.0.1 -- a major version past anything SkinTokens was developed
# against, and the release where a good deal of the Python API moved. Since the
# failure being chased is bpy_server dying the instant it starts, running the
# Blender the authors actually targeted removes one of the two candidate causes.
# 4.5.13 is the newest cp311 build in the 4.x series.
RUN pip install "bpy==4.5.13" \
    && python -c "import bpy; print('bpy', bpy.app.version_string)"

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

# Which commit built this image. Debugging the paint stage cost a full round
# of wrong conclusions because the endpoint was serving an image older than the
# one being reasoned about, and nothing in the response said so. The probe
# reports this, so "is the deployed image the one I just pushed" is a fact
# rather than an assumption.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

COPY handler.py /workspace/handler.py
CMD ["python", "-u", "/workspace/handler.py"]

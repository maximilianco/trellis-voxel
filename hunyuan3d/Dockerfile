# Hunyuan3D-2.1 image-to-3D as a RunPod serverless worker.
#
# Why this exists alongside the TRELLIS workers rather than replacing them:
# TRELLIS bakes its texture out of the multiview diffusion, and the result
# measures flat -- a 2048 albedo off TRELLIS.2 carries a mean gradient of
# 1.16/255, and reconstructing it from a 128px thumbnail costs only 4.2/255.
# That is why generated avatars read as low resolution no matter how large the
# texture is asked to be; the pixels are there and the detail is not.
# Hunyuan3D-2.1 runs a separate PBR paint stage (Hunyuan3D-Paint-v2-1, 2B)
# over the generated shape instead, which is the part TRELLIS has no equivalent
# of.
#
# It cannot share the TRELLIS image: this is built against torch 2.5.1/cu124
# and the SkinTokens rigger against torch 2.7.0/cu128, and every compiled
# extension is linked to one exact torch build.
#
# VRAM, from the upstream README -- read this before picking an endpoint GPU:
#   shape only            10 GB
#   texture only          21 GB
#   shape + texture       29 GB
# So a 24 GB card CANNOT do both in one pass. Either give the endpoint a 48 GB
# GPU (L40S / A6000 / A100), or send want_texture=false and paint separately.
# handler.py's probe reports the card and warns when it is too small.
ARG CUDA_TAG=12.4.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    CUDA_HOME=/usr/local/cuda \
    HY3D_ROOT=/workspace/Hunyuan3D-2.1 \
    PYTHONPATH=/workspace/Hunyuan3D-2.1:/workspace/Hunyuan3D-2.1/hy3dshape:/workspace/Hunyuan3D-2.1/hy3dpaint \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Without pipefail a failed pip inside a `| tee | tail` pipeline exits 0 and the
# broken image ships. This bit the TRELLIS build already.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# An unbounded ninja on a 4-core runner gets OOM-killed, which surfaces as a
# bare "Killed" rather than a compiler error.
ENV MAX_JOBS=4

# Python 3.11, not jammy's stock 3.10, and the reason is bpy.
#
# requirements.txt pins `bpy==4.0`, and that version no longer exists on PyPI:
# every bpy still published is cp311 (4.2.0 through 5.0.1) or cp313 (5.1+).
# There is no cp310 build left at all, so on 3.10 the pin is unsatisfiable and
# pip fails with "Could not find a version that satisfies the requirement
# bpy==4.0" -- the same class of failure the rigger image hit, for the same
# reason: bpy ships one wheel per interpreter that Blender embeds, and old ones
# get withdrawn.
#
# 3.11 keeps torch 2.5.1/cu124 and every other pin here on wheels that exist
# (pymeshlab 2022.2.post3 and open3d 0.18.0 both publish cp311).
#
# The X and GL libraries are bpy's. Importing Blender links against them even
# headless, and a missing one surfaces at `import bpy` in a running job rather
# than at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv git wget ninja-build \
        build-essential libglib2.0-0 libjpeg-dev libeigen3-dev \
        libgl1 libegl1 libgles2 libx11-6 libxi6 libxxf86vm1 libxfixes3 \
        libxrender1 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel \
    && python -c "import sys; assert sys.version_info[:2]==(3,11), sys.version"

# Torch first: custom_rasterizer and the mesh painter both compile against it.
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git ${HY3D_ROOT}

# Upstream's requirements need two edits before they will resolve at all.
#
#   bpy==4.0        withdrawn from PyPI; 4.2.0 is the oldest still published
#                   and is cp311, which is why this image is on 3.11.
#   --extra-index-url
#                   the file points pip at Tencent and Aliyun mirrors. From a
#                   GitHub runner those are slow at best, and an extra index is
#                   a dependency-confusion foothold besides: any name resolved
#                   there outranks nothing and silently shadows PyPI. Everything
#                   pinned here exists on PyPI proper.
#
# Edited with sed rather than by vendoring a copy of the file, so upstream's
# pins keep flowing through and only the two broken lines are overridden.
RUN sed -i -e 's|^--extra-index-url.*$||' \
           -e 's|^bpy==4\.0$|bpy==4.2.0|' ${HY3D_ROOT}/requirements.txt \
    && grep -nE 'bpy|extra-index' ${HY3D_ROOT}/requirements.txt || true

# basicsr is sdist-only and its setup.py imports torch, so with PEP 517 build
# isolation -- where torch does not exist -- it dies with "No module named
# 'torch'". Installing it first, unisolated, means the pin is already satisfied
# when the requirements file asks for it.
RUN pip install --no-build-isolation basicsr==1.4.2 2>&1 \
        | tee /workspace/basicsr_build.log | tail -20

# tail -40, not -5. The first failure here was a one-line pip resolver error
# that the previous `tail -5` truncated away entirely, which turned a plainly
# diagnosable build into a guess. The full log stays in the image regardless.
RUN pip install -r ${HY3D_ROOT}/requirements.txt 2>&1 \
        | tee /workspace/requirements_build.log | tail -40
RUN pip install runpod huggingface_hub accelerate onnxruntime

# The two native extensions. These are NOT optional: custom_rasterizer is what
# projects the painted views back onto the mesh, and without the mesh painter
# the texture stage has no rasteriser at all. Let them fail loudly.
RUN cd ${HY3D_ROOT}/hy3dpaint/custom_rasterizer \
    && pip install --no-build-isolation -e . 2>&1 \
       | tee /workspace/rasterizer_build.log | tail -5

RUN cd ${HY3D_ROOT}/hy3dpaint/DifferentiableRenderer \
    && bash compile_mesh_painter.sh 2>&1 \
       | tee /workspace/painter_build.log | tail -5

# The paint pipeline upscales its generated views with RealESRGAN and expects
# the checkpoint on disk at a fixed path. Fetching it at runtime means every
# cold worker pays for it, and the download is not on a persisted volume.
RUN mkdir -p ${HY3D_ROOT}/hy3dpaint/ckpt \
    && wget -q -O ${HY3D_ROOT}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    && echo "RealESRGAN baked: $(du -h ${HY3D_ROOT}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth | cut -f1)"

# Background removal. The paint stage wants a cut-out subject, and rembg
# otherwise pulls 176 MB of u2net into /root/.u2net on every cold start.
RUN pip install rembg \
    && mkdir -p /root/.u2net \
    && wget -q -O /root/.u2net/u2net.onnx \
       https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
    && echo "u2net baked: $(du -h /root/.u2net/u2net.onnx | cut -f1)"

# Weights are NOT baked by default. Shape (3.3B) plus paint (2B) is well over
# 20 GB; attach `tencent/Hunyuan3D-2.1` to the endpoint as a RunPod model
# instead and the worker loads it from /runpod-volume/huggingface-cache.
# handler.py detects that mount at runtime.
ARG HY3D_REPO=tencent/Hunyuan3D-2.1
ARG BAKE_WEIGHTS=0
ENV HY3D_MODEL=${HY3D_REPO}
ENV BAKE_WEIGHTS=${BAKE_WEIGHTS}

RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['HY3D_MODEL'], local_dir='/workspace/weights')" ; \
        echo "weights baked into image" ; \
    else \
        echo "weights not baked; attach the model to the endpoint instead" ; \
    fi

COPY handler.py /workspace/handler.py
CMD ["python", "-u", "/workspace/handler.py"]

# TRELLIS image-to-3D as a RunPod serverless worker.
#
# Why this is worth doing: TRELLIS is internally voxel-based. Its first stage
# ("sparse structure") decodes to a 64^3 occupancy grid before any mesh or
# gaussian decoder runs -- see ss_dec_conv3d_16l8 in the model repo. So the thing
# we actually want is not a by-product of a mesh export, it is the pipeline's own
# intermediate representation. That makes image -> voxels a short path.
#
# The dependency list below is TRELLIS' own setup.sh --basic set rather than a
# hand-picked subset. Leaving out easydict alone makes `import trellis` fail with
# ModuleNotFoundError, which is indistinguishable from "TRELLIS is not installed"
# unless the error text is printed.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    CUDA_HOME=/usr/local/cuda \
    PYTHONPATH=/workspace/TRELLIS \
    ATTN_BACKEND=xformers \
    SPCONV_ALGO=native

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git wget ninja-build \
        build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# Torch first: every extension below compiles against this exact build.
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# TRELLIS setup.sh --basic, plus the serverless SDK.
RUN pip install \
        numpy==1.26.4 pillow imageio imageio-ffmpeg scipy tqdm einops ninja \
        easydict opencv-python-headless trimesh xatlas pyvista pymeshfix igraph \
        transformers==4.46.3 safetensors huggingface_hub accelerate \
        onnxruntime rembg runpod

# Large, and only used by mesh post-processing: never fail the build on it.
RUN pip install open3d || echo "open3d unavailable (mesh post-processing disabled)"

# Attention and sparse convolution: required by the sparse-structure stage.
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install spconv-cu120

# Optional; ATTN_BACKEND falls back to xformers, and this is slow to build.
RUN pip install flash-attn==2.6.3 --no-build-isolation || \
        echo "flash-attn unavailable; using ATTN_BACKEND=xformers"

# utils3d at the revision TRELLIS pins.
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Mesh/radiance-field renderers. The voxel path never calls these, so a failure
# here must not sink the image.
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html || \
        echo "kaolin unavailable (mesh export disabled)" ; \
    pip install git+https://github.com/NVlabs/nvdiffrast.git || \
        echo "nvdiffrast unavailable (mesh export disabled)" ; \
    pip install git+https://github.com/JeffreyXiang/diffoctreerast.git || \
        echo "diffoctreerast unavailable (radiance-field export disabled)"

# The gaussian rasterizer comes from mip-splatting -- which is where TRELLIS'
# own setup.sh takes it from, not from a TRELLIS submodule. Per-voxel colour
# depends on it; shape does not.
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ || \
        echo "gaussian rasterizer unavailable (shape works, colours will not)"

WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /workspace/TRELLIS
# setup.sh still references extensions/vox2seq, but that directory no longer
# exists in the repo (checked against the git tree) and nothing imports it.

# Weights are NOT baked in by default.
#
# Runpod caches HuggingFace models host-side: attach the model to the endpoint
# and the worker loads it from /runpod-volume/huggingface-cache with no image
# bloat, no network volume, and faster cold starts than either. handler.py
# detects that mount at runtime.
#
# Build with --build-arg BAKE_WEIGHTS=1 to embed them instead.
ARG TRELLIS_REPO=microsoft/TRELLIS-image-large
ARG BAKE_WEIGHTS=0
ENV TRELLIS_MODEL=${TRELLIS_REPO}
ENV BAKE_WEIGHTS=${BAKE_WEIGHTS}

RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        python -c "import os; from huggingface_hub import snapshot_download; repo=os.environ['TRELLIS_MODEL']; ignore=['*_1024_*'] if 'TRELLIS.2' in repo else None; snapshot_download(repo, local_dir='/workspace/weights/'+repo.split('/')[-1], ignore_patterns=ignore)" ; \
        if echo "$TRELLIS_MODEL" | grep -q "TRELLIS.2" ; then \
            python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS-image-large', local_dir='/workspace/weights/TRELLIS-image-large', allow_patterns=['pipeline.json','ckpts/ss_dec_conv3d_16l8_fp16*'])" ; \
        fi ; \
        echo "weights baked into image" ; \
    else \
        echo "weights not baked; attach the model to the endpoint instead" ; \
    fi

COPY handler.py /workspace/handler.py
CMD ["python", "-u", "/workspace/handler.py"]

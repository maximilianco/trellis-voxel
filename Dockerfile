# TRELLIS image-to-3D as a RunPod serverless worker.
#
# Why this is worth doing: TRELLIS is internally voxel-based. Its first stage
# ("sparse structure") decodes to a 64^3 occupancy grid before any mesh or
# gaussian decoder runs -- see ss_dec_conv3d_16l8 in the model repo. So the thing
# we actually want is not a by-product of a mesh export, it is the pipeline's own
# intermediate representation. That makes image -> voxels a short path.
#
# Build notes: the heavy custom CUDA extensions (flash-attn, kaolin, spconv,
# nvdiffrast, diffoctreerast, gaussian rasterizer) are the fragile part of any
# TRELLIS install. They are installed in dependency order and the optional
# renderers are allowed to fail, because the voxel path does not need them.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    CUDA_HOME=/usr/local/cuda \
    # TRELLIS reads these at import time.
    ATTN_BACKEND=xformers \
    SPCONV_ALGO=native \
    HF_HOME=/workspace/hf \
    TRELLIS_MODEL=microsoft/TRELLIS-image-large

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git wget ninja-build \
        build-essential libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# Torch first: every extension below compiles against this exact build.
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
# This is TRELLIS' own setup.sh --basic set, not a guess. Omitting easydict
# alone makes `import trellis` raise ModuleNotFoundError, which looks identical
# to "TRELLIS is not installed" unless the error message is printed.
RUN pip install         numpy==1.26.4 pillow imageio imageio-ffmpeg scipy tqdm einops ninja         easydict opencv-python-headless trimesh xatlas pyvista pymeshfix igraph         transformers==4.46.3 safetensors huggingface_hub accelerate         onnxruntime rembg runpod
# open3d is large and used only by mesh post-processing; never fail the build on it.
RUN pip install open3d || echo "open3d unavailable (mesh post-processing disabled)"

# Attention and sparse convolution: required by the sparse-structure stage.
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install spconv-cu120

# flash-attn is optional (ATTN_BACKEND falls back to xformers) and slow to build,
# so a failure here must not sink the image.
RUN pip install flash-attn==2.6.3 --no-build-isolation || \
    echo "flash-attn unavailable; continuing with ATTN_BACKEND=xformers"

# utils3d at the revision TRELLIS pins.
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Renderers. Only the mesh/gaussian *export* paths need these; the voxel path
# does not, so they are best-effort.
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html || \
        echo "kaolin unavailable (mesh export disabled)" ; \
    pip install git+https://github.com/NVlabs/nvdiffrast.git || \
        echo "nvdiffrast unavailable (mesh export disabled)" ; \
    pip install git+https://github.com/JeffreyXiang/diffoctreerast.git || \
        echo "diffoctreerast unavailable (radiance-field export disabled)"

WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /workspace/TRELLIS

# The gaussian rasterizer comes from mip-splatting, which is where TRELLIS'
# own setup.sh takes it from. It is what makes per-voxel colour available.
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting &&     pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ ||         echo "gaussian rasterizer unavailable (shape works, colours will not)"
# setup.sh still references extensions/vox2seq, but that directory no longer
# exists in the repo (verified against the git tree) and nothing imports it.

ENV PYTHONPATH=/workspace/TRELLIS:$PYTHONPATH

# Weights are NOT baked in by default.
#
# Runpod caches HuggingFace models host-side: create the endpoint with
#   runpodctl serverless create ... --model-reference microsoft/TRELLIS-image-large
# and the worker loads from the standard HF cache with no image bloat, no network
# volume, and faster cold starts than either. Baking 3.3 GB into the image would
# make every first pull slower for no benefit.
#
# Set BAKE_WEIGHTS=1 to embed them anyway (useful if you would rather not depend
# on the host cache).
ARG TRELLIS_REPO=microsoft/TRELLIS-image-large
ARG BAKE_WEIGHTS=0
ENV TRELLIS_MODEL=${TRELLIS_REPO}

RUN if [ "$BAKE_WEIGHTS" = "1" ]; then       python -c "import os; from huggingface_hub import snapshot_download; repo=os.environ['TRELLIS_MODEL']; name=repo.split('/')[-1]; ignore=['*_1024_*'] if 'TRELLIS.2' in repo else None; snapshot_download(repo, local_dir='/workspace/weights/'+name, ignore_patterns=ignore); snapshot_download('microsoft/TRELLIS-image-large', local_dir='/workspace/weights/TRELLIS-image-large', allow_patterns=['pipeline.json','ckpts/ss_dec_conv3d_16l8_fp16*']) if 'TRELLIS.2' in repo else None" &&       echo "weights baked";     else echo "weights not baked; use --model-reference at endpoint creation"; fi

# TRELLIS.2 needs the TRELLIS-1 sparse-structure decoder too; when using
# --model-reference, pass both repos.
COPY handler.py /workspace/handler.py
CMD ["python", "-u", "/workspace/handler.py"]

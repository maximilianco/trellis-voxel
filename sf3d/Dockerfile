FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS dependencies
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/stable-fast-3d \
    PYTHONUNBUFFERED=1 \
    USE_CUDA=1 USE_NATIVE_ARCH=0 MAX_JOBS=2 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    HF_HOME=/workspace/model-cache \
    U2NET_HOME=/workspace/rembg-cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev git build-essential ninja-build \
    ca-certificates libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir pip==24.2 setuptools==69.5.1 wheel==0.44.0
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124
ARG SF3D_COMMIT=ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2
RUN git clone https://github.com/Stability-AI/stable-fast-3d.git /workspace/stable-fast-3d \
    && git -C /workspace/stable-fast-3d checkout "${SF3D_COMMIT}" \
    && rm -rf /workspace/stable-fast-3d/.git
WORKDIR /workspace/stable-fast-3d
COPY constraints.txt /workspace/constraints.txt
RUN pip install --no-cache-dir --no-build-isolation -c /workspace/constraints.txt -r requirements.txt \
    && pip install --no-cache-dir -c /workspace/constraints.txt runpod==1.7.13
# Cache public background-removal and auxiliary encoders; gated SF3D weights
# are downloaded using a runtime secret, never included in a public image.
RUN python - <<'PY'
import rembg, open_clip, torch
from sf3d.models.tokenizers.dinov2 import Dinov2Model
from texture_baker import TextureBaker
import uv_unwrapper
assert torch.__version__.startswith('2.4.1') and torch.version.cuda == '12.4'
rembg.new_session('u2net', providers=['CPUExecutionProvider'])
Dinov2Model.from_pretrained('facebook/dinov2-large')
open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
print('Compiled extensions and public model caches verified')
PY

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/stable-fast-3d PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/model-cache U2NET_HOME=/workspace/rembg-cache \
    SF3D_MODEL_ID=stabilityai/stable-fast-3d \
    SF3D_MODEL_REVISION=f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778 \
    SF3D_TEXTURE_RESOLUTION=1024 OMP_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 ca-certificates libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=dependencies /workspace /workspace
ARG WORKER_REVISION=development
ENV SF3D_WORKER_REVISION=${WORKER_REVISION}
COPY handler.py /workspace/sf3d-handler.py
WORKDIR /workspace/stable-fast-3d
LABEL org.opencontainers.image.revision=${WORKER_REVISION}
CMD ["python", "-u", "/workspace/sf3d-handler.py"]

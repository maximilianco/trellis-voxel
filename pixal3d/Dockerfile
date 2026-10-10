# Extend the last successfully published TRELLIS.2 CUDA build, leaving its
# endpoint untouched. The existing FlexGEMM/CuMesh/O-Voxel stack is reused.
ARG BASE_IMAGE=ghcr.io/maximilianco/trellis-voxel:v2-01e1b90f4d897f0db04289ae1fa20977b8bd784e
FROM ${BASE_IMAGE} AS dependencies
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PIXAL3D_COMMIT=f7cf38429b0bd264f1995f0f8743a88b1c728b94
ARG MOGE_COMMIT=74fbce054ebed49800de42d0ad0e83495065719a
ARG NAF_COMMIT=37f2dfc180f2de53d98bd601109c0da0dd6b0f43
ENV PYTHONPATH=/workspace/Pixal3D:/workspace/TRELLIS.2 \
    ATTN_BACKEND=sdpa \
    PIXAL3D_LOW_VRAM=true \
    PIXAL3D_MODEL_RESOLUTION=1024 \
    PIXAL3D_FRONT_ROTATION=true \
    PIXAL3D_MODEL_REVISION=b0cb2e1b794cab9aa0ac38a95d794a4d9337437f \
    MAX_JOBS=2 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
RUN git clone https://github.com/TencentARC/Pixal3D.git /workspace/Pixal3D \
    && git -C /workspace/Pixal3D checkout "${PIXAL3D_COMMIT}"
COPY torch-constraints.txt /workspace/torch-constraints.txt

# Keep compiled extensions bound to torch 2.6/cu124. Only replace the Python
# dependencies Pixal3D requires. Gradio is unnecessary for a serverless worker.
RUN sed -e '/^git+https:\/\/github.com\/microsoft\/MoGe.git$/d' -e '/^gradio$/d' \
        /workspace/Pixal3D/requirements.txt > /workspace/pixal-requirements.txt \
    && pip install -c /workspace/torch-constraints.txt -r /workspace/pixal-requirements.txt \
    && pip install -c /workspace/torch-constraints.txt \
        "git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}" \
        https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# NAF supports the legacy QK/AV interface. Use the official binary matched
# exactly to our Python 3.10 / torch 2.6 / CUDA 12.4 ABI, without replacing torch.
# No CPU-only fallback or lengthy source compilation in an image builder.
RUN pip install --no-deps --only-binary=:all: \
        natten==0.17.5+torch260cu124 -f https://whl.natten.org

# Bake pinned NAF source into the image, avoiding moving hub code at runtime.
RUN git clone https://github.com/valeoai/NAF.git /workspace/NAF \
    && git -C /workspace/NAF checkout "${NAF_COMMIT}"
# Its released weights remain model/naf_release.pth, cached by TORCH_HOME.
RUN NAF_COMMIT="${NAF_COMMIT}" python - <<'PY'
import os
from pathlib import Path
path=Path('/workspace/Pixal3D/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py')
text=path.read_text()
needle='"valeoai/NAF", "naf", pretrained=True, device=device, trust_repo=True'
assert text.count(needle)==1, 'Upstream NAF loader changed; review before building'
path.write_text(text.replace(needle, '"/workspace/NAF", "naf", pretrained=True, device=device, source="local", trust_repo=True'))
PY

RUN python - <<'PY'
import importlib, torch
assert torch.__version__.startswith('2.6.0'), torch.__version__
assert torch.version.cuda=='12.4', torch.version.cuda
for name in ['inference','pixal3d','flex_gemm','cumesh','o_voxel','natten','moge','rembg']:
    importlib.import_module(name)
from transformers import DINOv3ViTModel
from natten import libnatten
from natten.functional import na2d_qk, na2d_av
torch.hub.load('/workspace/NAF','naf',source='local',pretrained=False,device='cpu')
print('Pixal3D imports and compiled NATTEN kernel library verified; GPU smoke test still required')
PY

# Keep the worker revision out of dependency layers so handler fixes reuse them.
FROM dependencies AS runtime
ARG WORKER_REVISION=development
ENV PIXAL3D_WORKER_REVISION=${WORKER_REVISION}
COPY handler.py /workspace/pixal3d-handler.py
LABEL org.opencontainers.image.revision=${WORKER_REVISION}
CMD ["python", "-u", "/workspace/pixal3d-handler.py"]

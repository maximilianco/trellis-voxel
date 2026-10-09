# Extend the last successfully published TRELLIS.2 CUDA build, leaving its
# endpoint untouched. The existing FlexGEMM/CuMesh/O-Voxel stack is reused.
ARG BASE_IMAGE=ghcr.io/maximilianco/trellis-voxel:v2-01e1b90f4d897f0db04289ae1fa20977b8bd784e
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PIXAL3D_COMMIT=f7cf38429b0bd264f1995f0f8743a88b1c728b94
ARG MOGE_COMMIT=74fbce054ebed49800de42d0ad0e83495065719a
ARG NAF_COMMIT=37f2dfc180f2de53d98bd601109c0da0dd6b0f43
ARG WORKER_REVISION=development
ENV PYTHONPATH=/workspace/Pixal3D:/workspace/TRELLIS.2 \
    ATTN_BACKEND=sdpa \
    PIXAL3D_LOW_VRAM=true \
    PIXAL3D_MODEL_RESOLUTION=1024 \
    PIXAL3D_FRONT_ROTATION=true \
    PIXAL3D_MODEL_REVISION=b0cb2e1b794cab9aa0ac38a95d794a4d9337437f \
    PIXAL3D_WORKER_REVISION=${WORKER_REVISION} \
    MAX_JOBS=2 \
    NATTEN_N_WORKERS=2 \
    NATTEN_CUDA_ARCH="8.0;8.6;8.9" \
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
        https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl \
        cmake==3.30.9

# Explicit architectures are essential: a CPU-only image builder would
# otherwise silently install NATTEN without its GPU kernels.
RUN pip install -c /workspace/torch-constraints.txt natten==0.21.0 --no-build-isolation \
        2>&1 | tee /workspace/natten_build.log | tail -30

# Upstream uses a moving torch.hub source; pin the NAF code too. Its released
# weights remain the upstream model/naf_release.pth asset, cached by TORCH_HOME.
RUN NAF_COMMIT="${NAF_COMMIT}" python - <<'PY'
import os
from pathlib import Path
path=Path('/workspace/Pixal3D/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py')
text=path.read_text()
needle='"valeoai/NAF", "naf"'
assert text.count(needle)==1, 'Upstream NAF loader changed; review before building'
path.write_text(text.replace(needle, '"valeoai/NAF:'+os.environ['NAF_COMMIT']+'", "naf"'))
PY

COPY handler.py /workspace/pixal3d-handler.py
RUN python - <<'PY'
import importlib, torch
assert torch.__version__.startswith('2.6.0'), torch.__version__
assert torch.version.cuda=='12.4', torch.version.cuda
for name in ['inference','pixal3d','flex_gemm','cumesh','o_voxel','natten','moge','rembg']:
    importlib.import_module(name)
from transformers import DINOv3ViTModel
from natten import _C
print('Pixal3D imports and compiled NATTEN kernel library verified; GPU smoke test still required')
PY
LABEL org.opencontainers.image.revision=${WORKER_REVISION}
CMD ["python", "-u", "/workspace/pixal3d-handler.py"]

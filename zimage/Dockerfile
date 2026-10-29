# Z-Image-Turbo (Tongyi-MAI, Apache 2.0) as a RunPod serverless worker: text to image in a few seconds, for concept
# pictures (sheets of four objects for SF3D) and game images. The weights (about 20 GB) download at the first start
# into the container disk (40 GB) at a pinned revision; FlashBoot keeps a started worker's state.

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1
RUN pip install --no-cache-dir "diffusers==0.42.1" "transformers>=4.51,<5" "accelerate>=1.2" "safetensors>=0.4" "huggingface_hub[hf_xet]>=0.34" "runpod>=1.7,<2" "pillow>=10"

ARG GIT_SHA=unknown
ENV WORKER_COMMIT=${GIT_SHA}
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]

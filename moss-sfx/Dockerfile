# MOSS-SoundEffect v2.0 (OpenMOSS, Apache 2.0) as a RunPod serverless worker: text to sound effect, 48 kHz, up to
# 30 s. The weights (about 10 GB) download at the first start into the container disk (30 GB) at a pinned revision;
# FlashBoot keeps a started worker's state. Python 3.12 (the package asks for it) with torch 2.9 on CUDA 12.8.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 HF_XET_HIGH_PERFORMANCE=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.9.0+cu128" "torchaudio==2.9.0+cu128"
# The pipeline package at a pinned commit, with its own pins (transformers 4.57, diffusers 0.37); the hub within
# what transformers 4 accepts.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
    "moss-soundeffect-v2 @ git+https://github.com/OpenMOSS/MOSS-TTS@4cf2dab88fc8711aef93089b3b75c0ae07373e47#subdirectory=moss_soundeffect_v2" \
    "huggingface_hub[hf_xet]<1" "runpod>=1.7,<2"

ARG GIT_SHA=unknown
ENV WORKER_COMMIT=${GIT_SHA}
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]

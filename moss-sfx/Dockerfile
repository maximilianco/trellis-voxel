# MOSS-SoundEffect v2.0 (OpenMOSS, Apache 2.0) as a RunPod serverless worker: text to sound effect, 48 kHz, up to
# 30 s. The weights (about 10 GB) download at the first start into the container disk (30 GB) at a pinned revision;
# FlashBoot keeps a started worker's state. Python 3.12 (the package asks for it) with torch 2.9 on CUDA 12.8.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 HF_XET_HIGH_PERFORMANCE=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --root-user-action=ignore --extra-index-url https://download.pytorch.org/whl/cu128 "torch==2.9.0+cu128" "torchaudio==2.9.0+cu128"
RUN pip install --no-cache-dir --root-user-action=ignore "runpod>=1.7,<2" "huggingface_hub[hf_xet]<1"
# Both packages without their own lists, then what their code imports: descript-audiotools pins protobuf below 3.20
# (for an old tensorboard), which RunPod's telemetry can't share (pip found no way, 2026-10-29); the current tensorboard
# takes the current protobuf. The pipeline's Gradio demo stays out.
RUN pip install --no-cache-dir --root-user-action=ignore --no-deps "descript-audiotools==0.7.2" \
    "moss-soundeffect-v2 @ git+https://github.com/OpenMOSS/MOSS-TTS@4cf2dab88fc8711aef93089b3b75c0ae07373e47#subdirectory=moss_soundeffect_v2"
RUN pip install --no-cache-dir --root-user-action=ignore --extra-index-url https://download.pytorch.org/whl/cu128 \
    argbind pyloudnorm importlib-resources scipy julius ffmpy ipython rich matplotlib librosa pystoi torch-stoi flatten-dict markdown2 randomname tensorboard \
    "torch==2.9.0+cu128" "torchaudio==2.9.0+cu128" "numpy==1.26.4" "einops==0.8.2" "pillow==12.2.0" "tqdm==4.67.3" "safetensors==0.7.0" \
    "transformers==4.57.1" "diffusers==0.37.1" "ftfy==6.3.1" "regex==2026.4.4" "soundfile==0.13.1" "imageio==2.37.3"

ARG GIT_SHA=unknown
ENV WORKER_COMMIT=${GIT_SHA}
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]

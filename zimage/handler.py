"""RunPod serverless worker: Z-Image-Turbo (Tongyi-MAI, Apache 2.0), text to image.

Concept pictures for the image-to-3D workers (a sheet of four objects, or one) and game images, a few seconds each:
an 8-step distilled 6B model. Weights download from Hugging Face at the worker's first start (pinned revision; about
20 GB, too large to bake into an image GitHub's runners can build) and stay loaded.

Input:
  {"prompt": "...", "width": 1024, "height": 1024, "steps": 9, "seed": -1, "format": "jpeg"}
  {"prompts": ["...", "..."]}  several pictures in one job (one load, back to back)
  {"probe": true}  versions, GPU, whether the model is loaded; no picture
Output:
  {"images": [{"image_b64", "format", "seed", "seconds"}], "gpu", "offload", "load_seconds", "commit", "model"}
"""

import base64
import io
import os
import random
import subprocess
import time

import runpod
import torch

MODEL = os.environ.get("ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
REVISION = os.environ.get("ZIMAGE_REVISION", "f332072aa78be7aecdf3ee76d5c247082da564a6")
CACHE = os.environ.get("ZIMAGE_CACHE", "/models/z-image-turbo")
COMMIT = os.environ.get("WORKER_COMMIT", "unknown")
PIPE = None
STATE = {"load_seconds": None, "offload": None}


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as error:  # noqa: BLE001 - a probe reports, never fails
        return f"unknown ({error})"


def load():
    """The pipeline, once: downloaded at the first start, on the GPU (or offloaded module by module on a card under 30 GB)."""
    global PIPE
    if PIPE is not None:
        return PIPE
    started = time.time()
    from huggingface_hub import snapshot_download
    from diffusers import ZImagePipeline

    local = snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=CACHE,
        allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    pipe = ZImagePipeline.from_pretrained(local, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    if vram < 30:
        # The 4B text encoder and the 6B transformer don't fit a 24 GB card together: each moves in when it runs.
        pipe.enable_model_cpu_offload()
        STATE["offload"] = True
    else:
        pipe.to("cuda")
        STATE["offload"] = False
    PIPE = pipe
    STATE["load_seconds"] = round(time.time() - started, 1)
    print(f"[worker] Z-Image ready in {STATE['load_seconds']} s (offload {STATE['offload']}, {vram:.0f} GB)", flush=True)
    return PIPE


def picture(pipe, prompt, width, height, steps, seed, image_format):
    seed = seed if seed >= 0 else random.randrange(2**31)
    started = time.time()
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        # Turbo is distilled with guidance built in.
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    buffer = io.BytesIO()
    if image_format == "png":
        image.save(buffer, format="PNG")
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
    return {"image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"), "format": "png" if image_format == "png" else "jpeg", "seed": seed, "seconds": round(time.time() - started, 2)}


def handler(job):
    job_input = job.get("input") or {}
    if job_input.get("probe"):
        return {"ready": PIPE is not None, "gpu": gpu_name(), "load_seconds": STATE["load_seconds"], "offload": STATE["offload"], "commit": COMMIT, "model": f"{MODEL}@{REVISION[:12]}"}
    prompts = job_input.get("prompts") or ([job_input["prompt"]] if job_input.get("prompt") else [])
    prompts = [str(p) for p in prompts if str(p).strip()][:8]
    if not prompts:
        return {"error": "prompt (or prompts, up to 8) is required"}
    width = max(256, min(2048, int(job_input.get("width", 1024)) // 16 * 16))
    height = max(256, min(2048, int(job_input.get("height", 1024)) // 16 * 16))
    steps = max(1, min(30, int(job_input.get("steps", 9))))
    seed = int(job_input.get("seed", -1))
    image_format = "png" if str(job_input.get("format", "jpeg")).lower() == "png" else "jpeg"
    try:
        pipe = load()
        images = [picture(pipe, prompt, width, height, steps, seed if seed < 0 else seed + i, image_format) for i, prompt in enumerate(prompts)]
    except Exception as error:  # noqa: BLE001 - the caller gets the reason
        return {"error": f"{type(error).__name__}: {error}"}
    return {"images": images, "gpu": gpu_name(), "offload": STATE["offload"], "load_seconds": STATE["load_seconds"], "commit": COMMIT, "model": f"{MODEL}@{REVISION[:12]}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

"""RunPod serverless worker: MOSS-SoundEffect v2.0 (OpenMOSS, Apache 2.0), text to sound effect.

Sound effects for games where recordings fall short (a creature's call, a spell, a machine nobody recorded): a 1.3B
DiT with flow matching, a DAC VAE and a Qwen3 text encoder, 48 kHz, up to 30 s. Weights download from Hugging Face at
the worker's first start (pinned revision; about 10 GB) and stay loaded. torch.compile stays off (TORCHDYNAMO_DISABLE):
its first call compiles for minutes, longer than a serverless job should wait.

Input:
  {"prompt": "...", "seconds": 4, "steps": 50, "cfg_scale": 4.0, "seed": -1, "format": "mp3"}
  {"prompts": ["...", "..."]}  several sounds in one job (one load, back to back)
  {"probe": true}  versions, GPU, whether the model is loaded; no sound
Output:
  {"sounds": [{"audio_b64", "format", "seconds", "seed", "took"}], "gpu", "load_seconds", "commit", "model"}
"""

import base64
import os
import random
import subprocess
import tempfile
import time

import runpod
import torch

MODEL = os.environ.get("MOSS_SFX_MODEL", "OpenMOSS-Team/MOSS-SoundEffect-v2.0")
REVISION = os.environ.get("MOSS_SFX_REVISION", "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d")
CACHE = os.environ.get("MOSS_SFX_CACHE", "/models/moss-soundeffect-v2")
COMMIT = os.environ.get("WORKER_COMMIT", "unknown")
PIPE = None
STATE = {"load_seconds": None}


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as error:  # noqa: BLE001 - a probe reports, never fails
        return f"unknown ({error})"


def load():
    """The pipeline, once: downloaded at the first start, on the GPU in bfloat16."""
    global PIPE
    if PIPE is not None:
        return PIPE
    started = time.time()
    from huggingface_hub import snapshot_download
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    local = snapshot_download(MODEL, revision=REVISION, local_dir=CACHE, token=os.environ.get("HF_TOKEN") or None)
    PIPE = MossSoundEffectPipeline.from_pretrained(local, torch_dtype=torch.bfloat16, device="cuda")
    STATE["load_seconds"] = round(time.time() - started, 1)
    print(f"[worker] MOSS-SoundEffect ready in {STATE['load_seconds']} s", flush=True)
    return PIPE


def sound(pipe, prompt, seconds, steps, cfg_scale, seed, audio_format):
    seed = seed if seed >= 0 else random.randrange(2**31)
    torch.manual_seed(seed)
    started = time.time()
    audio = pipe(prompt=prompt, seconds=seconds, num_inference_steps=steps, cfg_scale=cfg_scale)
    with tempfile.TemporaryDirectory() as folder:
        wav = os.path.join(folder, "sound.wav")
        pipe.save_audio(audio, wav)
        path = wav
        if audio_format == "mp3":
            path = os.path.join(folder, "sound.mp3")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "160k", path], check=True, timeout=60)
        with open(path, "rb") as file:
            data = file.read()
    return {"audio_b64": base64.b64encode(data).decode("ascii"), "format": audio_format, "seconds": seconds, "seed": seed, "took": round(time.time() - started, 2)}


def handler(job):
    job_input = job.get("input") or {}
    if job_input.get("probe"):
        return {"ready": PIPE is not None, "gpu": gpu_name(), "load_seconds": STATE["load_seconds"], "commit": COMMIT, "model": f"{MODEL}@{REVISION[:12]}"}
    prompts = job_input.get("prompts") or ([job_input["prompt"]] if job_input.get("prompt") else [])
    prompts = [str(p) for p in prompts if str(p).strip()][:6]
    if not prompts:
        return {"error": "prompt (or prompts, up to 6) is required"}
    seconds = max(0.5, min(30.0, float(job_input.get("seconds", 4))))
    steps = max(10, min(150, int(job_input.get("steps", 50))))
    cfg_scale = max(1.0, min(10.0, float(job_input.get("cfg_scale", 4.0))))
    seed = int(job_input.get("seed", -1))
    audio_format = "wav" if str(job_input.get("format", "mp3")).lower() == "wav" else "mp3"
    try:
        pipe = load()
        sounds = [sound(pipe, prompt, seconds, steps, cfg_scale, seed if seed < 0 else seed + i, audio_format) for i, prompt in enumerate(prompts)]
    except Exception as error:  # noqa: BLE001 - the caller gets the reason
        return {"error": f"{type(error).__name__}: {error}"}
    return {"sounds": sounds, "gpu": gpu_name(), "load_seconds": STATE["load_seconds"], "commit": COMMIT, "model": f"{MODEL}@{REVISION[:12]}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
